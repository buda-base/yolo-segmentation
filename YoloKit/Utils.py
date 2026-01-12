import os
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

import random
import shutil
import yaml
import xml.etree.ElementTree as ET

from dataclasses import asdict
from numpy.typing import NDArray
from pathlib import Path
from shapely.geometry import Polygon, box

from YoloKit.Config import COLOR_DICT, PHOTI_CLASS_MAP, SEMANTIC_TEXTREGION_MAP
from YoloKit.Data import InstanceRecord, PolyData, TileData, ResizePadData


def get_filename(file_path: str) -> str:
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def show_image(
    image: NDArray, cmap: str = "", axis="off", fig_x: int = 8, fix_y: int = 8
) -> None:
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)


def is_pagexml_done(xml_path: str) -> bool:
    """
    Return True if TranskribusMetadata status == 'DONE'
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "TranskribusMetadata":
            status = elem.attrib.get("status", "").upper()
            return status == "DONE"

    return False


def parse_points(points_str: str) -> NDArray:
    pts = []

    for p in points_str.split():
        x, y = p.split(",")
        pts.append((float(x), float(y)))

    return np.array(pts, dtype=np.float32)


def extract_textregion_attribute(elem):
    """
    Returns text region attribute (e.g. 'caption', 'margin', ...)
    or None if the TextRegion should be ignored.
    """
    custom = elem.attrib.get("custom", "")
    custom = custom.lower()

    for key, semantic in SEMANTIC_TEXTREGION_MAP.items():
        if key in custom:
            return semantic

    return None


def load_pagexml_instances(xml_path: str) -> list[tuple[str, Polygon]]:
    """
    Extract semantically meaningful instances from PageXML.
    Returns: list of (class_id, Polygon)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns_uri = root.tag.split("}")[0].strip("{")
    ns = {"pc": ns_uri}

    instances = []

    # TextLines
    for tl in root.findall(".//pc:TextLine", ns):
        coords_el = tl.find("pc:Coords", ns)

        if coords_el is None:
            continue
        pts = parse_points(coords_el.attrib["points"])

        if pts.shape[0] < 3:
            continue
        poly = Polygon(pts)

        if poly.is_valid:
            instances.append(("line", poly))

    # ImageRegion
    for ir in root.findall(".//pc:ImageRegion", ns):
        coords_el = ir.find("pc:Coords", ns)

        if coords_el is None:
            continue
        poly = Polygon(parse_points(coords_el.attrib["points"]))

        if poly.is_valid:
            instances.append(("image", poly))

    # TextRegion
    for tr in root.findall(".//pc:TextRegion", ns):
        attribute = extract_textregion_attribute(tr)

        if attribute is None:
            continue  # ignore generic container regions

        coords_el = tr.find("pc:Coords", ns)
        if coords_el is None:
            continue

        poly = Polygon(parse_points(coords_el.attrib["points"]))

        if poly.is_valid:
            instances.append((attribute, poly))

    return instances


def is_tile_empty(tile_mask: NDArray, min_white_ratio: float = 0.01) -> bool:
    white = tile_mask > 0
    white_ratio = np.count_nonzero(white) / white.size

    return bool(white_ratio < min_white_ratio)


def split_dataset(root: str | Path, val_ratio: float = 0.1):
    root = Path(root)
    img_dir = root / "images"
    lbl_dir = root / "labels"

    train_img = root / "images/train"
    val_img = root / "images/val"
    train_lbl = root / "labels/train"
    val_lbl = root / "labels/val"

    # Make dirs
    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect images
    imgs = list(img_dir.glob("*.png"))
    random.shuffle(imgs)

    split_idx = int(len(imgs) * (1 - val_ratio))
    train_files = imgs[:split_idx]
    val_files = imgs[split_idx:]

    for img_path in train_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        shutil.move(img_path, train_img / img_path.name)
        shutil.move(lbl_path, train_lbl / lbl_path.name)

    for img_path in val_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        shutil.move(img_path, val_img / img_path.name)
        shutil.move(lbl_path, val_lbl / lbl_path.name)


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
                    inter = poly.intersection(tile_box)
                    if inter.is_empty:
                        continue

                    # flatten polygons
                    geoms = (
                        [inter]
                        if inter.geom_type == "Polygon"
                        else (
                            list(inter.geoms)
                            if inter.geom_type == "MultiPolygon"
                            else []
                        )
                    )
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


def resize_and_pad(
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


def process_page(
    xml_path: str,
    img_path: str,
    output_path: str | Path,
    classes: dict[str, int],
    tile_size: int = 512,
    target_width: int = 2048,
    overlap: float = 0.8,
    snap_threshold: float = 0.15,
    debug: bool = False,
):

    if not is_pagexml_done(xml_path):
        if debug:
            print(f"[SKIP] {xml_path} not DONE")
        return

    base = Path(xml_path).stem
    img = cv2.imread(str(img_path))

    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    instances = load_pagexml_instances(xml_path)
    max_patch_rows = 2 * tile_size
    snap_threshold_px = int(tile_size * snap_threshold)

    img_padded, meta = resize_and_pad(
        img,
        max_w=target_width,
        max_h=max_patch_rows,
        tile_size=tile_size,
        overlap=overlap,
        snap_threshold_px=snap_threshold_px,
        max_patch_rows=max_patch_rows,
        debug=debug,
    )

    scaled_instances = scale_polygons(instances, meta.scale)

    output_path = Path(output_path)
    out_img_dir = output_path / "images"
    out_mask_dir = output_path / "masks"
    out_lbl_dir = output_path / "labels"
    debug_dir = output_path / "debug"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # write debug masks to disk
    if debug:
        color_mask = instances_to_color_mask(
            scaled_instances, meta.padded_w, meta.padded_h, class_colors=COLOR_DICT
        )

        cv2.imwrite(
            str(debug_dir / f"{Path(img_path).stem}_instances_rgb.png"), color_mask
        )

    mask = instances_to_mask(
        scaled_instances, meta.padded_w, meta.padded_h, include_class_ids=classes
    )

    _ = tile_image_and_labels(
        img_padded,
        mask,
        scaled_instances,
        base,
        out_img_dir,
        out_mask_dir,
        out_lbl_dir,
        tile_size,
        classes,
        overlap=0.8,
    )


def write_yolo_yaml(dataset_root, class_map, yaml_path=None):
    dataset_root = Path(dataset_root).resolve()

    if yaml_path is None:
        yaml_path = dataset_root / "dataset.yaml"
    else:
        yaml_path = Path(yaml_path)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    names = {int(v): k.lower() for k, v in class_map.items()}

    data = {
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def write_parquet(records: list[InstanceRecord], out_path: str):
    rows = [asdict(r) for r in records]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)


def infer_tiles(model, tiles):
    predictions = []

    for tile in tiles:
        results = model(tile["img"])
        r = results[0]

        if r.masks is None:
            continue

        for poly in r.masks.xy:
            predictions.append(
                {
                    "polygon": Polygon(poly),
                    "x0": tile["x0"],
                    "y0": tile["y0"],
                    "confidence": (
                        float(r.boxes.conf.mean()) if r.boxes is not None else 1.0
                    ),
                }
            )

    return predictions


def preds_to_polygons(preds):
    page_polys = []

    for p in preds:
        coords = [(x + p["x0"], y + p["y0"]) for x, y in p["polygon"].exterior.coords]
        page_polys.append(Polygon(coords))

    return page_polys


# ----------- Drawing functions --------------------------


def draw_yolo_seg_labels(
    img: NDArray,
    label_path: str,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    copy: bool = True,
) -> NDArray:
    """
    Draw YOLO segmentation polygons from a .txt label file onto an image tile.

    Args:
        img: np.ndarray (H, W, 3) – tile image
        label_path: path to YOLO .txt file
        color: BGR color tuple
        thickness: contour thickness
        copy: draw on copy or in-place

    Returns:
        Image with drawn contours
    """
    if copy:
        out = img.copy()
    else:
        out = img

    H, W = img.shape[:2]

    with open(label_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                # need at least class + 3 points
                continue

            coords = list(map(float, parts[1:]))

            # convert normalized coords → pixel coords
            pts = []
            for i in range(0, len(coords), 2):
                x = coords[i] * W
                y = coords[i + 1] * H
                pts.append([int(round(x)), int(round(y))])

            np_pts = np.asarray(pts, dtype=np.int32)
            np_pts = np_pts.reshape((-1, 1, 2))  # OpenCV format

            cv2.drawContours(out, [np_pts], -1, color, thickness)

    return out


def draw_polygons_only(
    image: NDArray,
    polygons: list[list[tuple[float, float]]],
    thickness=2,
):
    overlay = image.copy()

    for poly in polygons:
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(overlay, [pts], True, color, thickness)

    return overlay


def show_prediction_overlay(results, fig_x: int = 12, fig_y: int = 8):
    img = results[0].orig_img.copy()

    if img.ndim == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    # Instance masks: (N, H, W), float in [0,1]
    masks = results[0].masks.data.cpu().numpy()

    overlay = img_rgb.copy()
    alpha = 0.4  # transparency
    mask_color = np.array([0, 255, 0], dtype=np.uint8)  # green

    for m in masks:
        binary = m > 0.5
        overlay[binary] = ((1 - alpha) * overlay[binary] + alpha * mask_color).astype(
            np.uint8
        )

    plt.figure(figsize=(fig_x, fig_y))
    plt.imshow(overlay)
    plt.axis("off")
