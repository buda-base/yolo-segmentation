import os
from typing import Mapping
import cv2

import matplotlib.pyplot as plt
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

import random
import shutil
import yaml
import xml.etree.ElementTree as ET

from dataclasses import asdict
from datetime import datetime

from glob import glob
from natsort import natsorted
from itertools import chain
from numpy.typing import NDArray
from pathlib import Path
from uuid import uuid1
import xml.etree.ElementTree as ET

from YoloKit.data import InstanceRecord


def get_utc_time():
    """
    Get current UTC time as a formatted string.

    Returns:
        Current UTC time in ISO format (YYYY-MM-DDTHH:MM:SS)
    """
    utc_time = datetime.now()
    utc_time = utc_time.strftime("%Y-%m-%dT%H:%M:%S")

    return utc_time


def generate_guid(clock_seq: int):
    """
    Generate a UUID with a specific clock sequence.
    
    Args:
        clock_seq: Clock sequence value for UUID generation
        
    Returns:
        Generated UUID
    """
    return uuid1(clock_seq=clock_seq)



def get_filename(file_path: str) -> str:
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def get_directory_images(file_path: str) -> list[str]:
    exts = ["jpg", "jpeg", "png", "tif", "tiff"]
    images = natsorted(
        chain.from_iterable(glob(f"{file_path}/*.{ext}") for ext in exts)
    )

    return images


def create_dir(dir_path: str) -> None:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created output directory: {dir_path}")
    except BaseException as e:
        print(f"Failed to create directory: {e}")


def show_image(
    image: NDArray, cmap: str = "", axis="off", fig_x: int = 8, fig_y: int = 8
) -> None:
    plt.figure(figsize=(fig_x, fig_y))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)


def show_overlay(
    image: NDArray,
    mask: NDArray,
    alpha=0.4,
    axis="off",
    fig_x: int = 24,
    fix_y: int = 13,
):
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha)


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


def write_yolo_yaml(
    dataset_root: str,
    class_map: Mapping[str, int],
    yaml_path: str | None = None,
) -> None:
    dataset_path = Path(dataset_root).resolve()

    if yaml_path is None:
        yaml_file = dataset_path / "dataset.yaml"
    else:
        yaml_file = Path(yaml_path)

    yaml_file.parent.mkdir(parents=True, exist_ok=True)

    names = {int(v): k.lower() for k, v in class_map.items()}

    data = {
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }

    with yaml_file.open("w") as f:
        yaml.dump(data, f, sort_keys=False)


def write_parquet(records: list[InstanceRecord], out_path: str):
    rows = [asdict(r) for r in records]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)


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


def draw_yolo_bboxes(
    image_path: str,
    label_path: str,
    class_map: dict
):
    """
    Visualize YOLO bounding boxes on an image.

    YOLO format:
        class cx cy w h   (normalized)
    """

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    H, W = img.shape[:2]

    if not Path(label_path).exists():
        print("No label file found.")
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        # --- YOLO normalized -> pixel coords ---
        bw = w * W
        bh = h * H

        px = cx * W
        py = cy * H

        x1 = int(px - bw / 2)
        y1 = int(py - bh / 2)
        x2 = int(px + bw / 2)
        y2 = int(py + bh / 2)

        # clamp for safety
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        # color per class
        color = (
            (0, 255, 0) if cls == 1 else
            (255, 0, 0) if cls == 0 else
            (0, 0, 255)
        )

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = class_map.get(cls, str(cls))
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return img

def draw_global_boxes(img: NDArray, boxes: list, classes: list):
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)

        color = (0,255,0) if cls == 1 else (255,0,0)

        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

    return img

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
