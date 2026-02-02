import cv2
import os
import math
import torch

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from shapely.geometry import Polygon, box

from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry
from YoloKit.Config import COLOR_DICT, PHOTI_CLASS_MAP
from YoloKit.Data import TileData, ResizePadData


def tile_image(
    img: NDArray,
    tile_size: int = 512,
    overlap: float = 0.8,
) -> list[TileData]:
    stride = max(1, int(tile_size * (1.0 - overlap)))
    H, W = img.shape[:2]

    y_range = int(H - tile_size + 1)
    x_range = int(W - tile_size + 1)

    assert y_range > 0 and x_range > 0
    assert stride > 0, "overlap too high -> stride becomes 0"

    tile_id = 0
    tiles: list[TileData] = []

    for y0 in range(0, y_range, stride):
        for x0 in range(0, x_range, stride):
            tiles.append(
                TileData(
                    tile_id,
                    x0,
                    y0,
                    tile_size,
                    img[y0 : y0 + tile_size, x0 : x0 + tile_size],
                )
            )
            tile_id += 1

    return tiles


def is_tile_empty(tile_mask: NDArray, min_white_ratio: float = 0.01) -> bool:
    white = tile_mask > 0
    white_ratio = np.count_nonzero(white) / white.size

    return bool(white_ratio < min_white_ratio)


def resize_and_pad(
    img_name: str,
    img: NDArray,
    max_w: int = 2048,
    max_h: int = 2048,
    tile_size: int = 512,
    overlap: float = 0.8,
    snap_threshold_px: int = 78,
    max_patch_rows: int = 2,
    debug: bool = False,
) -> tuple[NDArray, ResizePadData]:
    H, W = img.shape[:2]

    # vertical overlap in pixels (matches your tiling overlap)
    patch_vertical_overlap_px = int(round(tile_size * overlap))

    s = compute_downscale(
        w=W,
        h=H,
        max_w=max_w,
        max_h=max_h,
        patch_size=tile_size,
        patch_vertical_overlap_px=patch_vertical_overlap_px,
        snap_extra_patch_row_threshold_px=snap_threshold_px,
        max_patch_rows=max_patch_rows,
    )

    new_w = int(round(W * s))
    new_h = int(round(H * s))

    if debug:
        print(f"Scale: s={s:.6f}  orig={W}x{H} → resized={new_w}x{new_h}")

    img_resized = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    )

    pad_w_min = max(0, tile_size - new_w)
    pad_h_min = max(0, tile_size - new_h)

    padded_w = new_w + pad_w_min
    padded_h = new_h + pad_h_min

    pad_w_tile = (tile_size - (padded_w % tile_size)) % tile_size
    pad_h_tile = (tile_size - (padded_h % tile_size)) % tile_size

    pad_w = pad_w_min + pad_w_tile
    pad_h = pad_h_min + pad_h_tile

    img_padded = cv2.copyMakeBorder(
        img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
    )

    final_w = new_w + pad_w
    final_h = new_h + pad_h

    if debug:
        print(f"Padding: final={final_w}x{final_h}  pad_w={pad_w} pad_h={pad_h}")

    meta = ResizePadData(
        img_name=img_name,
        orig_w=W,
        orig_h=H,
        scale=s,
        resized_w=new_w,
        resized_h=new_h,
        pad_w=pad_w,
        pad_h=pad_h,
        padded_w=final_w,
        padded_h=final_h,
    )

    return img_padded, meta


def tile_image_and_labels(
    img: NDArray,
    mask: NDArray,
    instances: list[tuple[str, Polygon]],
    base_name: str,
    out_img_dir: str | Path,
    out_mask_dir: str | Path,
    out_lbl_dir: str | Path,
    tile_size: int = 512,
    class_map: dict[str, int] = PHOTI_CLASS_MAP,
    overlap: float = 0.8,
    min_white_ratio: float = 0.005,
):
    stride = int(tile_size * (1 - overlap))
    H, W = img.shape[:2]

    y_range = H - tile_size + 1
    x_range = W - tile_size + 1
    assert y_range > 0 and x_range > 0
    assert stride > 0, "overlap too high -> stride becomes 0"

    tile_id = 0
    for y0 in range(0, y_range, stride):
        for x0 in range(0, x_range, stride):

            tile_img = img[y0 : y0 + tile_size, x0 : x0 + tile_size]
            tile_mask = mask[y0 : y0 + tile_size, x0 : x0 + tile_size]

            if is_tile_empty(tile_mask, min_white_ratio=min_white_ratio):
                continue

            tile_box = box(x0, y0, x0 + tile_size, y0 + tile_size)

            lbl_path = Path(out_lbl_dir) / f"{base_name}_{tile_id:04d}.txt"
            kept_any = False

            with open(lbl_path, "w", encoding="utf8") as f:
                for cid, poly in instances:
                    inter: BaseGeometry = poly.intersection(tile_box)
                    if inter.is_empty:
                        continue

                    # flatten polygons
                    geoms: list[Polygon]
                    if isinstance(inter, Polygon):
                        geoms = [inter]
                    elif isinstance(inter, MultiPolygon):
                        geoms = list(inter.geoms)
                    else:
                        geoms = []

                    for g in geoms:
                        coords = []
                        for x, y in np.array(g.exterior.coords):
                            coords.append(f"{(x - x0) / tile_size:.6f}")
                            coords.append(f"{(y - y0) / tile_size:.6f}")
                        if len(coords) >= 6:  # at least 3 points
                            class_idx = class_map[str(cid)]
                            f.write(str(class_idx) + " " + " ".join(coords) + "\n")
                            kept_any = True

            if not kept_any:
                try:
                    os.remove(lbl_path)
                except OSError:
                    pass
                continue

            cv2.imwrite(
                str(Path(out_img_dir) / f"{base_name}_{tile_id:04d}.png"), tile_img
            )
            cv2.imwrite(
                str(Path(out_mask_dir) / f"{base_name}_{tile_id:04d}.png"), tile_mask
            )
            tile_id += 1

    return tile_id


def compute_downscale(
    w: int,
    h: int,
    max_w: int,
    max_h: int,
    patch_size: int,
    patch_vertical_overlap_px: int = 78,
    snap_extra_patch_row_threshold_px: int = 78,
    max_patch_rows: int = 2,
) -> float:
    """
    Compute a resize scale factor for patch-based inference of line detection.

    Pipeline logic:
      1) Downscale to fit within (max_w, max_h) (never upscale in this step).
      2) Ensure at least one full patch in height (may upscale).
      3) Snap height *down* if it barely crosses a patch-row boundary (works for any row count).
      4) Optionally cap the number of patch rows by shrinking height to the maximum allowed.

    Definitions (vertical tiling with overlap):
      stride_y = patch_size - patch_vertical_overlap_px
      Row boundaries happen at: patch_size + k * stride_y   (k >= 0)
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")

    if patch_size <= 0:
        raise ValueError(f"Invalid patch_size: {patch_size}")

    if patch_vertical_overlap_px < 0 or patch_vertical_overlap_px >= patch_size:
        raise ValueError(
            f"patch_vertical_overlap_px must be in [0, patch_size-1], got {patch_vertical_overlap_px}"
        )

    stride_y = patch_size - patch_vertical_overlap_px  # vertical step between rows

    # -----------------------------
    # Step 1) Fit within max box (no upscaling)
    # -----------------------------
    scale_to_max_w = max_w / float(w)
    scale_to_max_h = max_h / float(h)
    s = min(scale_to_max_w, scale_to_max_h, 1.0)

    scaled_h = h * s

    # -----------------------------
    # Step 2) Ensure at least one patch in height
    # -----------------------------
    if scaled_h < patch_size:
        s = patch_size / float(h)
        scaled_h = patch_size

    # -----------------------------
    # Step 3) Snap down if we're just barely above ANY row boundary
    #
    # Boundaries: H = patch_size + k * stride_y
    # If scaled_h is in (boundary, boundary + threshold], snap down to boundary.
    # -----------------------------
    if snap_extra_patch_row_threshold_px > 0:
        if scaled_h > patch_size:
            excess = scaled_h - patch_size

            # k is the largest integer such that boundary(k) <= scaled_h
            k = int(math.floor(excess / float(stride_y)))
            boundary_h = patch_size + k * stride_y

            extra_px = scaled_h - boundary_h
            if 0.0 < extra_px <= float(snap_extra_patch_row_threshold_px):
                scaled_h = boundary_h
                s = scaled_h / float(h)

    # -----------------------------
    # Step 4) Cap patch rows (soft cap)
    #
    # Max height allowed for R rows: patch_size + (R - 1) * stride_y
    # -----------------------------
    if max_patch_rows is not None and max_patch_rows > 0:
        max_allowed_h = patch_size + (max_patch_rows - 1) * stride_y
        if scaled_h > max_allowed_h:
            scaled_h = max_allowed_h
            s = scaled_h / float(h)

    return s


def instances_to_mask(
    instances: list, width: int, height: int, include_class_ids: dict[str, int] | None
):
    mask = np.zeros((height, width), dtype=np.uint8)

    for cid, poly in instances:
        if include_class_ids is not None and cid not in include_class_ids.keys():
            continue
        pts = np.array(poly.exterior.coords).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def instances_to_color_mask(
    instances: list,
    width: int,
    height: int,
    class_colors: dict | None,
    background: tuple = (0, 0, 0),
):
    """
    Render instance polygons into a single RGB mask for visualization.

    Args:
        instances: list of (class_id, shapely.Polygon)
        width, height: output image size
        class_colors: dict {class_id: (B, G, R)} or (R,G,B) - OpenCV uses BGR
        background: background color (B, G, R)

    Returns:
        RGB uint8 image (H, W, 3)
    """
    if class_colors is None:
        class_colors = COLOR_DICT

    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask[:] = background

    for class_id, poly in instances:
        if poly.is_empty or not poly.is_valid:
            continue

        color = class_colors.get(class_id, (255, 255, 255))  # fallback: white

        # exterior
        pts = np.array(poly.exterior.coords).astype(np.int32)
        cv2.fillPoly(mask, [pts], color)

        # handle holes (rare but possible)
        for interior in poly.interiors:
            hole_pts = np.array(interior.coords).astype(np.int32)
            cv2.fillPoly(mask, [hole_pts], background)

    return mask


def mask_to_polygons(
    mask: NDArray,
    min_area: int = 10,
    epsilon: float = 1.5,
):
    """
    Convert a single binary mask to polygons.

    Returns a list of polygons (some masks may fragment).
    """
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        approx = cv2.approxPolyDP(cnt, epsilon, True)
        poly = [(float(x), float(y)) for [[x, y]] in approx]
        polygons.append(poly)

    return polygons


def masks_to_contours(merged_masks: list[NDArray]):
    """
    merged_masks: list[H×W] or (M, H, W)
    returns: list[np.ndarray] contours (N_i, 2)
    """
    contours = []

    for mask in merged_masks:
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(cnts) == 0:
            continue

        # take largest connected component
        contour = max(cnts, key=cv2.contourArea)
        contour = contour.squeeze(1)  # (N, 2)

        contours.append(contour)

    return contours

def tile_to_padded(poly, tile: TileData):
    return [(x + tile.x0, y + tile.y0) for x, y in poly]


def unpad(poly, meta: ResizePadData):
    return [(min(x, meta.resized_w - 1), min(y, meta.resized_h - 1)) for x, y in poly]


def resized_to_original(poly, meta: ResizePadData):
    inv = 1.0 / meta.scale
    return [(x * inv, y * inv) for x, y in poly]


def reproject_polygon(
    poly_tile: NDArray,
    tile: TileData,
    meta: ResizePadData,
):
    poly = tile_to_padded(poly_tile, tile)
    poly = unpad(poly, meta)
    poly = resized_to_original(poly, meta)
    return poly


def scale_polygons(instances: list[tuple[str, Polygon]], scale_factor: float):
    scaled_instances = []

    for class_name, poly in instances:
        coords = np.asarray(poly.exterior.coords, dtype=np.float32)
        coords[:, 0] *= scale_factor
        coords[:, 1] *= scale_factor

        p2 = Polygon(coords)

        if p2.is_valid and not p2.is_empty:
            scaled_instances.append((class_name, p2))

    return scaled_instances


def preds_to_polygons(preds):
    page_polys = []

    for p in preds:
        coords = [(x + p["x0"], y + p["y0"]) for x, y in p["polygon"].exterior.coords]
        page_polys.append(Polygon(coords))

    return page_polys


# ----------- PostProcessing ----------------------------
def sort_by_y(global_masks, y_centers):
    order = np.argsort(y_centers)
    return global_masks[order], y_centers[order]


def collect_global_line_masks(
    results, tile_data: list[TileData], page_shape, class_name="line"
):
    H_page, W_page = page_shape[:2]

    global_masks_list: list[NDArray] = []
    y_centers_list: list[NDArray] = []

    for res, tile in zip(results, tile_data):

        if res.masks is None:
            continue

        masks = res.masks.data.cpu().numpy()  # (N, h, w)
        boxes = res.boxes.xyxy.cpu().numpy()  # (N, 4)
        classes = res.boxes.cls.cpu().numpy()
        names = res.names

        line_id = [k for k, v in names.items() if v == class_name][0]
        idx = np.where(classes == line_id)[0]

        if len(idx) == 0:
            continue

        masks = masks[idx]
        boxes = boxes[idx]

        y_center = (boxes[:, 1] + boxes[:, 3]) / 2 + tile.y0
        y_centers_list.append(y_center)

        M, h, w = masks.shape

        tile_global = np.zeros((M, H_page, W_page), dtype=np.uint8)
        tile_global[:, tile.y0 : tile.y0 + h, tile.x0 : tile.x0 + w] = (
            masks > 0.5
        ).astype(np.uint8)

        global_masks_list.append(tile_global)

    if len(global_masks_list) == 0:
        return None, None

    global_masks = np.concatenate(global_masks_list, axis=0)
    y_centers = np.concatenate(y_centers_list, axis=0)

    return global_masks, y_centers


def collect_global_line_masks_gpu(
    results, tile_data, page_shape, class_name="line", device="cuda"
):
    H_page, W_page = page_shape[:2]

    global_masks = []
    y_centers = []

    for res, tile in zip(results, tile_data):

        if res.masks is None:
            continue

        masks = res.masks.data
        boxes = res.boxes.xyxy
        classes = res.boxes.cls
        names = res.names

        line_id = [k for k, v in names.items() if v == class_name][0]
        class_idx = (classes == line_id).nonzero(as_tuple=True)[0]

        if class_idx.numel() == 0:
            continue

        masks = masks[class_idx]
        boxes = boxes[class_idx]

        y_center = (boxes[:, 1] + boxes[:, 3]) / 2 + tile.y0
        y_centers.append(y_center)

        M, h, w = masks.shape

        tile_global = torch.zeros(
            (M, H_page, W_page), dtype=torch.bool, device=masks.device
        )

        tile_global[:, tile.y0 : tile.y0 + h, tile.x0 : tile.x0 + w] = masks > 0.5

        global_masks.append(tile_global)

    if len(global_masks) == 0:
        return None, None

    global_masks = torch.cat(global_masks, dim=0)
    y_centers = torch.cat(y_centers, dim=0)

    return global_masks, y_centers


def collapse_duplicates(y_centers, eps: int =  4):
    """
    Collapses multiple detections of the same line into one.
    """
    y_centers = np.sort(y_centers)
    collapsed = [y_centers[0]]

    for y in y_centers[1:]:
        if abs(y - collapsed[-1]) > eps:
            collapsed.append(y)
        else:
            collapsed[-1] = (collapsed[-1] + y) / 2

    return np.array(collapsed)


def cluster_lines(y_centers, dup_eps=8, spacing_factor=0.4):
    y_centers = np.array(y_centers)

    # 1. collapse duplicates
    collapsed = collapse_duplicates(y_centers, eps=dup_eps)

    if len(collapsed) < 2:
        return [[i] for i in range(len(y_centers))]

    # 2. estimate true spacing
    dy = np.diff(collapsed)
    typical_spacing = np.median(dy)

    # 3. downscale spacing for clustering
    threshold = typical_spacing * spacing_factor

    # 4. cluster original detections
    order = np.argsort(y_centers)
    y_sorted = y_centers[order]

    clusters = []
    current = [order[0]]

    for i in range(1, len(y_sorted)):
        if (y_sorted[i] - y_sorted[i - 1]) < threshold:
            current.append(order[i])
        else:
            clusters.append(current)
            current = [order[i]]

    clusters.append(current)
    return clusters


def merge_line_masks(global_masks, clusters) -> list[NDArray]:
    merged = []

    for cluster in clusters:
        line_mask = np.zeros_like(global_masks[0])
        for idx in cluster:
            line_mask |= global_masks[idx]
        merged.append(line_mask)

    return merged

def optimize_contour(contour, eps: float = 2.0):
    return cv2.approxPolyDP(contour, eps, True)


def mask_to_contour(line_mask: NDArray, optimize: bool = False, eps: float = 2.0):
    cnts, _ = cv2.findContours(
        line_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    
    if optimize:
        cnts = [optimize_contour(x, eps) for x in cnts]

    contour = np.vstack([c.squeeze() for c in cnts])
    return contour


def mask_to_contours(line_mask: np.ndarray, optimize=False, eps=2.0):
    cnts, _ = cv2.findContours(
        line_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if optimize:
        cnts = [optimize_contour(c, eps) for c in cnts]

    # return list of (N_i, 2)
    contours = [c.squeeze(1) for c in cnts if c.shape[0] >= 3]
    return contours


def contour_to_original_space(contour_padded: NDArray, meta: ResizePadData) -> NDArray:
    """
    contour_padded: (N,2) in padded-resized space
    meta: ResizePadData
    """
    s = meta.scale
    contour_orig = contour_padded.copy().astype(np.float32)
    contour_orig[:, 0] /= s
    contour_orig[:, 1] /= s
    return contour_orig


def extract_line_images(
    page_img: NDArray, merged_masks: list[NDArray], background: str = "white"
):
    line_images = []

    if background == "white":
        bg_value = 255
    elif background == "black":
        bg_value = 0
    else:
        raise ValueError("background must be 'white' or 'black'")

    for mask in merged_masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        crop_img = page_img[y_min : y_max + 1, x_min : x_max + 1]
        crop_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

        # ocr_line = crop_img * crop_mask[..., None]
        ocr_line = crop_img.copy()
        ocr_line[crop_mask == 0] = bg_value
        line_images.append(ocr_line)

    return line_images
