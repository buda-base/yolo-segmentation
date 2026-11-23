import argparse
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
import yaml
from PIL import Image
from shapely import GeometryCollection

from tiblined_data import iter_split, line_polygons, stage_dataset, write_page_image
from yolo_kit.yolo import bbox_to_yolo, save_yolo_labels

SPLITS = {"train": "train", "validation": "val", "test": "test"}


def rasterize_polygon(mask, polygon):
    # Rasterize each polygon separately so overlapping lines stay foreground.
    rings = [polygon.exterior, *polygon.interiors]
    contours = [np.rint(ring.coords).astype(np.int32) for ring in rings]
    if not polygon.interiors:
        cv2.fillPoly(mask, contours, 1)
        return

    x0, y0 = contours[0].min(axis=0)
    x1, y1 = contours[0].max(axis=0) + 1
    region = mask[y0:y1, x0:x1]
    layer = np.zeros(region.shape, dtype=np.uint8)
    cv2.fillPoly(layer, [contour - (x0, y0) for contour in contours], 1)
    region |= layer


def export_page(row, image_dir, label_dir, mask_dir, counts):
    stem = row["stem"]
    width, height = row["width"], row["height"]
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes = []
    for parts in line_polygons(row, counts):
        bounds = GeometryCollection(parts).bounds
        boxes.append((0, *bbox_to_yolo(bounds, width, height)))
        for part in parts:
            rasterize_polygon(mask, part)

    write_page_image(row, image_dir)
    Image.fromarray(mask).save(mask_dir / f"{stem}.png")
    save_yolo_labels(label_dir / f"{stem}.txt", boxes)
    counts["pages"] += 1
    counts["lines"] += len(boxes)


def prepare_dataset(source: Path, output: Path, limit: int | None = None):
    groups = {}
    stems = set()
    with stage_dataset(source, output, limit) as (staged, parquet_paths):
        for source_split, split in SPLITS.items():
            image_dir, label_dir, mask_dir = (
                staged / entry / split for entry in ("images", "labels", "masks")
            )
            for directory in (image_dir, label_dir, mask_dir):
                directory.mkdir(parents=True)

            counts = Counter()
            for row in iter_split(
                parquet_paths[source_split], source_split, limit, stems, groups
            ):
                try:
                    export_page(row, image_dir, label_dir, mask_dir, counts)
                except (KeyError, ValueError, OSError, ElementTree.ParseError) as error:
                    raise ValueError(
                        f"Could not export page {row['stem']} in {source_split}: {error}"
                    ) from error
            if not counts["pages"] or not counts["lines"]:
                raise ValueError(f"No usable annotations in {source_split}")
            print(
                f"{source_split}: {counts['pages']} pages, {counts['lines']} lines, "
                f"{counts['repaired']} polygon repairs attempted, {counts['clipped']} clipped, {counts['dropped']} dropped",
                flush=True,
            )

        # Ultralytics resolves dataset paths relative to these YAML files.
        dataset = {split: f"images/{split}" for split in SPLITS.values()}
        dataset["names"] = {0: "line"}
        (staged / "detect.yaml").write_text(
            yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
        )
        # A one-class semantic head uses pixel value 1 for line and 0 for background.
        dataset["masks_dir"] = "masks"
        (staged / "semantic.yaml").write_text(
            yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TiBLINED for YOLO detection and semantic segmentation"
    )
    parser.add_argument("--source", type=Path, default=Path("data/TiBLINED"))
    parser.add_argument(
        "--output", type=Path, help="Destination (default: data/tiblined_yolo)"
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Export only the first N pages of each split for a smoke run",
    )
    args = parser.parse_args()
    output = args.output or Path("data/tiblined_yolo")
    if args.output is None and args.limit_per_split is not None:
        output = output.with_name(f"{output.name}_smoke_{args.limit_per_split}")
    try:
        prepare_dataset(args.source, output, args.limit_per_split)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
