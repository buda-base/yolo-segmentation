import cv2
import numpy as np
import xml.etree.ElementTree as ET


from numpy.typing import NDArray
from pathlib import Path
from shapely.geometry import Polygon

from YoloKit.Utils import get_filename
from YoloKit.Config import COLOR_DICT, SEMANTIC_TEXTREGION_MAP
from YoloKit.Processing.image import (
    resize_and_pad,
    scale_polygons,
    instances_to_color_mask,
    instances_to_mask,
    tile_image_and_labels,
)


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


def process_xml_data(
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
    img_name = get_filename(img_path)
    img = cv2.imread(str(img_path))

    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    instances = load_pagexml_instances(xml_path)
    max_patch_rows = 2 * tile_size
    snap_threshold_px = int(tile_size * snap_threshold)

    img_padded, meta = resize_and_pad(
        img_name,
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
