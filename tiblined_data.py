from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from shapely import Polygon, box, make_valid

from pagexml import parse_points

SOURCE_SPLITS = ("train", "validation", "test")
COLUMNS = ("stem", "image", "page_xml", "split", "group", "width", "height", "nlines")


def polygon_parts(geometry):
    if isinstance(geometry, Polygon):
        if not geometry.is_empty:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def line_polygons(row, counts):
    width, height = row["width"], row["height"]
    page = ElementTree.fromstring(row["page_xml"]).find("{*}Page")
    if page is None or (
        int(page.attrib["imageWidth"]),
        int(page.attrib["imageHeight"]),
    ) != (width, height):
        raise ValueError("PAGE-XML dimensions disagree with Parquet metadata")
    lines = page.findall(".//{*}TextLine")
    if len(lines) != row["nlines"]:
        raise ValueError("PAGE-XML line count disagrees with Parquet metadata")

    page_bounds = box(0, 0, width - 1, height - 1)
    for line in lines:
        coords = line.find("{*}Coords")
        points = parse_points(coords.get("points", "")) if coords is not None else []
        if len(points) < 3 or not np.isfinite(points).all():
            counts["dropped"] += 1
            continue
        geometry = Polygon(points)
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            counts["repaired"] += 1
        if not page_bounds.covers(geometry):
            geometry = geometry.intersection(page_bounds)
            counts["clipped"] += 1
        parts = list(polygon_parts(geometry))
        if not parts:
            counts["dropped"] += 1
            continue

        yield parts


def write_page_image(row, image_dir):
    stem = row["stem"]
    width, height = row["width"], row["height"]
    image_bytes = row["image"]["bytes"]
    with Image.open(BytesIO(image_bytes)) as image:
        if image.size != (width, height):
            raise ValueError("Image dimensions disagree with Parquet metadata")
        if image.format == "JPEG" and image.getexif().get(274, 1) == 1:
            image.load()
            name = f"{stem}.jpg"
            (image_dir / name).write_bytes(image_bytes)
        else:
            if image.mode not in {"1", "L", "RGB"}:
                image = image.convert("RGB")
            image.info.clear()
            name = f"{stem}.png"
            image.save(image_dir / name)
    return name


@contextmanager
def stage_dataset(source: Path, output: Path, limit: int | None):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Use a new output directory."
        )
    if output.is_relative_to(source):
        raise ValueError(
            "Use a separate output directory beside the TiBLINED snapshot."
        )
    if limit is not None and limit < 1:
        raise ValueError("The per-split limit must be positive.")

    parquet_paths = {
        split: source / "data" / f"{split}.parquet" for split in SOURCE_SPLITS
    }
    for path in parquet_paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing TiBLINED split: {path}. Check --source or run download_tiblined.py."
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tiblined-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        yield staged, parquet_paths
        staged.rename(output)


def iter_split(path, split, limit, stems, groups):
    with pq.ParquetFile(path) as parquet:
        count = 0
        for batch in parquet.iter_batches(batch_size=8, columns=COLUMNS):
            remaining = batch.num_rows if limit is None else limit - count
            for row in batch.slice(0, min(batch.num_rows, remaining)).to_pylist():
                stem, group = row["stem"], row["group"]
                if not stem or not all(
                    c.isascii() and (c.isalnum() or c in "-_") for c in stem
                ):
                    raise ValueError(f"Invalid page stem in {split}: {stem!r}")
                if stem.casefold() in stems:
                    raise ValueError(f"Duplicate page stem: {stem}")
                stems.add(stem.casefold())
                if (
                    row["split"] != split
                    or not group
                    or groups.setdefault(group, split) != split
                ):
                    raise ValueError(
                        f"Inconsistent dataset split or holdout group for {stem}"
                    )
                yield row
                count += 1
            if limit is not None and count >= limit:
                break
