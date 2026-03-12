import numpy as np


def parse_ndjson_bbox(
    record: dict,
) -> tuple[list[tuple], int, int]:
    """
    Converts one Ultralytics NDJSON image record
    into PageXML-like output:

        boxes, img_w, img_h

    where boxes = [(class_id, (x0,y0,x1,y1)), ...]
    """

    img_w = int(record["width"])
    img_h = int(record["height"])

    boxes = []

    ann = record.get("annotations", {})
    box_list = ann.get("boxes", [])

    for entry in box_list:
        # ultralytics format:
        # [class_id, cx, cy, w, h]
        cls, cx, cy, w, h = entry

        bbox = yolo_to_bbox(cx, cy, w, h, img_w, img_h)
        boxes.append((int(cls), bbox))

    return boxes, img_w, img_h


def yolo_to_bbox(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
):
    """
    Convert normalized YOLO bbox to absolute xyxy bbox.
    Input:
        cx, cy, w, h in range [0,1]
    Output:
        (x0, y0, x1, y1) in pixel space
    """
    bw = w * img_w
    bh = h * img_h

    x0 = (cx * img_w) - bw / 2
    y0 = (cy * img_h) - bh / 2
    x1 = x0 + bw
    y1 = y0 + bh

    return (x0, y0, x1, y1)



def polygon_to_bbox(points:list[tuple[float, float]]):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    return x_min, y_min, x_max, y_max


def bbox_to_yolo(
    bbox,
    img_w: int,
    img_h: int,
):
    x_min, y_min, x_max, y_max = bbox

    w = x_max - x_min
    h = y_max - y_min

    cx = x_min + w / 2
    cy = y_min + h / 2

    return (
        cx / img_w,
        cy / img_h,
        w / img_w,
        h / img_h,
    )


def generate_tiles(img_w: int, img_h: int, tile_size=640):
    tiles = []

    for y0 in range(0, img_h, tile_size):
        for x0 in range(0, img_w, tile_size):

            x1 = min(x0 + tile_size, img_w)
            y1 = min(y0 + tile_size, img_h)

            tiles.append({
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            })

    return tiles


def intersect_bbox(a, b):
    """
    bbox format: (x_min, y_min, x_max, y_max)
    """
    x_min = max(a[0], b[0])
    y_min = max(a[1], b[1])
    x_max = min(a[2], b[2])
    y_max = min(a[3], b[3])

    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)


def bbox_to_tile_yolo(bbox, tile, tile_size: int = 640):
    x_min, y_min, x_max, y_max = bbox

    # shift into tile coordinate system
    x_min -= tile["x0"]
    x_max -= tile["x0"]
    y_min -= tile["y0"]
    y_max -= tile["y0"]

    w = x_max - x_min
    h = y_max - y_min

    cx = x_min + w / 2
    cy = y_min + h / 2

    return (
        cx / tile_size,
        cy / tile_size,
        w / tile_size,
        h / tile_size,
    )


def tile_yolo_boxes(
    full_boxes,
    img_w,
    img_h,
    tile_size: int = 640,
):
    """
    full_boxes:
        [(class_id, (x_min, y_min, x_max, y_max)), ...]
    """

    tiles = generate_tiles(img_w, img_h, tile_size)

    tiled_results = []

    for tile in tiles:

        tile_bbox = (
            tile["x0"],
            tile["y0"],
            tile["x1"],
            tile["y1"],
        )

        tile_labels = []

        for class_id, bbox in full_boxes:

            inter = intersect_bbox(bbox, tile_bbox)
            if inter is None:
                continue

            yolo_box = bbox_to_tile_yolo(inter, tile, tile_size)
            tile_labels.append((class_id, *yolo_box))

        tiled_results.append((tile, tile_labels))

    return tiled_results


def non_max_suppression(boxes, scores, iou_thresh=0.5):
    """
    boxes: Nx4 (xyxy)
    scores: N
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def save_yolo_labels(label_path: str, boxes):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")