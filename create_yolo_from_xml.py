import argparse
from glob import glob
from pathlib import Path

from natsort import natsorted
from tqdm import tqdm

from YoloKit.Config import PHOTI_CLASS_MAP
from YoloKit.Utils import (
    is_pagexml_done,
    process_page,
    split_dataset,
    write_yolo_yaml
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TinyYOLO PageXML → tiled training data export"
    )

    parser.add_argument(
        "-i", "--images",
        required=True,
        help="Directory containing page images (e.g. *.jpg)"
    )

    parser.add_argument(
        "-x", "--xml",
        required=True,
        help="Directory containing PageXML files"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for tiles"
    )

    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="Image glob pattern (default: *.jpg)"
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size (default: 512)"
    )

    parser.add_argument(
        "--target-width",
        type=int,
        default=2048,
        help="Target resized page width before tiling (default: 2048)"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.8,
        help="Tile overlap fraction (default: 0.8)"
    )

    parser.add_argument(
        "--snap-threshold",
        type=float,
        default=0.15,
        help="Polygon snapping threshold (default: 0.15)"
    )

    parser.add_argument(
        "--no-classes",
        action="store_true",
        help="Disable class mapping (no PHOTI_CLASS_MAP)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    images_dir = Path(args.images)
    xml_dir = Path(args.xml)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = natsorted(glob(str(images_dir / args.pattern)))
    xml_files = natsorted(glob(str(xml_dir / "*.xml")))

    print(f"Images: {len(image_files)}")
    print(f"XML files: {len(xml_files)}")

    if len(image_files) != len(xml_files):
        print("WARNING: image and xml count differs!")

    class_map = None if args.no_classes else PHOTI_CLASS_MAP

    for img, xml in tqdm(zip(image_files, xml_files), total=len(image_files)):
        if not is_pagexml_done(xml):
            continue

        process_page(
            xml_path=xml,
            img_path=img,
            output_path=str(output_path),
            classes=class_map,
            tile_size=args.tile_size,
            target_width=args.target_width,
            overlap=args.overlap,
            snap_threshold=args.snap_threshold,
            debug=args.debug,
        )

        split_dataset(output_path)
        write_yolo_yaml(output_path, yaml_path=f"{output_path}/dataset.yaml", class_map=class_map)


if __name__ == "__main__":
    main()
