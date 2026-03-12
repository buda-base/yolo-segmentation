import cv2
from datetime import datetime
import json
import numpy as np
import xml.etree.ElementTree as ET


from numpy.typing import NDArray
from pathlib import Path
from shapely.geometry import Polygon

from YoloKit.utils import get_filename
from YoloKit.yolo import bbox_to_yolo, polygon_to_bbox
from YoloKit.config import COLOR_DICT, SEMANTIC_TEXTREGION_MAP
from YoloKit.image import (
    resize_and_pad,
    scale_polygons,
    instances_to_color_mask,
    instances_to_mask,
    tile_image_and_labels,
)


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
ET.register_namespace("", PAGE_NS)
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")


def now_iso():
    return datetime.now().isoformat()


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


def parse_points(points_str: str) -> list[tuple[float, float]]:
    pts = []

    for p in points_str.split():
        x, y = p.split(",")
        pts.append((float(x), float(y)))

    return pts


def yolo_to_bbox(box, width, height):
    """
    YOLO (cx,cy,w,h) -> pixel bbox
    """
    _, cx, cy, bw, bh = box

    cx *= width
    cy *= height
    bw *= width
    bh *= height

    x0 = int(cx - bw / 2)
    y0 = int(cy - bh / 2)
    x1 = int(cx + bw / 2)
    y1 = int(cy + bh / 2)

    return x0, y0, x1, y1


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


def parse_pagexml_to_yolo_bboxes(xml_path: str, class_map: dict):
    """
    Extract region-level YOLO bboxes from PAGE XML.

    Returns:
        List[(class_id, cx, cy, w, h)]
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

    page = root.find(".//pc:Page", ns)
    img_w = int(page.attrib["imageWidth"])
    img_h = int(page.attrib["imageHeight"])

    results = []

    for region in root.findall(".//pc:TextRegion", ns):

        region_type = region.attrib.get("type", None)
        if region_type not in class_map:
            continue

        coords = region.find("pc:Coords", ns)
        if coords is None:
            continue

        pts = parse_points(coords.attrib["points"])
        bbox = polygon_to_bbox(pts)
        cx, cy, w, h = bbox_to_yolo(bbox, img_w, img_h)

        class_id = class_map[region_type]

        results.append((class_id, float(cx), float(cy), float(w), float(h)))

    return results


def parse_pagexml(xml_path, class_map: dict):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

    page = root.find(".//pc:Page", ns)
    img_w = int(page.attrib["imageWidth"])
    img_h = int(page.attrib["imageHeight"])

    boxes = []

    for region in root.findall(".//pc:TextRegion", ns):

        region_type = region.attrib.get("type")
        if region_type not in class_map:
            continue

        coords = region.find("pc:Coords", ns)
        if coords is None:
            continue

        pts = parse_points(coords.attrib["points"])
        bbox = polygon_to_bbox(pts)

        boxes.append((class_map[region_type], bbox))

    return boxes, img_w, img_h


def create_pagexml(image_name, width, height, boxes, class_names):

    pcgts = ET.Element(
        f"{{{PAGE_NS}}}PcGts",
        {
            "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation": f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd"
        },
    )

    # ------------------------------------------------
    # REQUIRED METADATA BLOCK
    # ------------------------------------------------

    metadata = ET.SubElement(pcgts, f"{{{PAGE_NS}}}Metadata")

    creator = ET.SubElement(metadata, f"{{{PAGE_NS}}}Creator")
    creator.text = "Ultralytics converter"

    created = ET.SubElement(metadata, f"{{{PAGE_NS}}}Created")
    created.text = now_iso()

    lastchange = ET.SubElement(metadata, f"{{{PAGE_NS}}}LastChange")
    lastchange.text = now_iso()

    # ------------------------------------------------

    page = ET.SubElement(
        pcgts,
        f"{{{PAGE_NS}}}Page",
        {
            "imageFilename": image_name,
            "imageWidth": str(width),
            "imageHeight": str(height),
        },
    )

    for i, box in enumerate(boxes):

        class_id = box[0]
        label = class_names[str(class_id)]

        x0, y0, x1, y1 = yolo_to_bbox(box, width, height)

        region = ET.SubElement(
            page,
            f"{{{PAGE_NS}}}TextRegion",
            {
                "id": f"region_{i}",
                "type": label,
            },
        )

        coords = ET.SubElement(region, f"{{{PAGE_NS}}}Coords")

        coords.set(
            "points",
            f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}",
        )

    return ET.ElementTree(pcgts)


def convert_ultralytics_to_pagexml(
    ndjson_file: str,
    image_dir: str,
    output_dir: str,
):

    ndjson_path = Path(ndjson_file)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = None

    with open(ndjson_path, "r", encoding="utf-8") as f:

        for line in f:
            record = json.loads(line)

            if record["type"] == "dataset":
                class_names = record["class_names"]
                continue

            if record["type"] != "image":
                continue

            img_name = record["file"]
            width = record["width"]
            height = record["height"]

            boxes = record["annotations"]["boxes"]

            xml_tree = create_pagexml(
                img_name,
                width,
                height,
                boxes,
                class_names,
            )

            xml_path = output_dir / (Path(img_name).stem + ".xml")

            xml_tree.write(
                xml_path,
                encoding="utf-8",
                xml_declaration=True,
            )
