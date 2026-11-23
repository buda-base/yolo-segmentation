import argparse
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from filelock import FileLock
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
LINE_SCHEMA = pa.struct(
    [
        ("line_index", pa.int32()),
        ("x_min", pa.int32()),
        ("y_min", pa.int32()),
        ("x_max", pa.int32()),
        ("y_max", pa.int32()),
        ("confidence", pa.float32()),
    ]
)
PAGE_SCHEMA = pa.schema(
    [
        ("image_path", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("status", pa.string()),
        ("error", pa.string()),
        ("lines", pa.list_(LINE_SCHEMA)),
    ]
)


@dataclass(frozen=True)
class InferenceOptions:
    checkpoint: Path = Path("Models/best.pt")
    img_size: int = 1024
    confidence: float = 0.38
    iou: float = 0.5
    device: str = "0"
    fp32: bool = False
    decode_workers: int = 8

    @property
    def precision(self):
        return 32 if self.fp32 or self.device == "cpu" else 16


def image_paths(root: Path, manifest: Path | None = None) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image root is not a directory: {root}")
    if manifest is None:
        paths = []

        def raise_walk_error(error):
            raise error

        for directory, subdirs, filenames in os.walk(root, onerror=raise_walk_error):
            subdirs.sort()
            parent = Path(directory).relative_to(root)
            paths.extend(
                (parent / name).as_posix()
                for name in sorted(filenames)
                if Path(name).suffix.lower() in IMAGE_SUFFIXES
            )
    else:
        paths = []
        for line in manifest.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            path = Path(os.path.abspath(root / line))
            if (
                not path.is_relative_to(root)
                or path.suffix.lower() not in IMAGE_SUFFIXES
            ):
                raise ValueError(
                    f"Manifest entry must be a supported image under {root}: {line!r}"
                )
            paths.append(path.relative_to(root).as_posix())
        if len({os.path.normcase(path) for path in paths}) != len(paths):
            raise ValueError("The input manifest contains duplicate image paths.")
    if not paths:
        raise ValueError(f"No supported images found under {root}")
    return paths


def read_page(path: Path) -> np.ndarray:
    # Use the same orientation-aware OpenCV decode for inference and preview.
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError("OpenCV could not decode the image")
    return image


def prepare_page(root, image_path, img_size):
    row = {
        "image_path": image_path,
        "width": None,
        "height": None,
        "status": "ok",
        "error": None,
        "lines": [],
    }
    try:
        image = read_page(root / image_path)
    except (OSError, ValueError, cv2.error) as error:
        row.update(status="decode_error", error=str(error))
        return row, None, None

    height, width = image.shape[:2]
    row.update(width=width, height=height)
    gain = img_size / max(height, width)
    resized_w = max(1, round(width * gain))
    resized_h = max(1, round(height * gain))
    left = (img_size - resized_w) // 2
    top = (img_size - resized_h) // 2
    if (resized_w, resized_h) != (width, height):
        image = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
    if resized_w != img_size or resized_h != img_size:
        image = cv2.copyMakeBorder(
            image,
            top,
            img_size - resized_h - top,
            left,
            img_size - resized_w - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
    return row, image, (gain, left, top)


def format_lines(boxes, width, height, transform):
    gain, left, top = transform
    boxes = boxes[np.isfinite(boxes).all(axis=1)]
    xyxy = (boxes[:, :4] - (left, top, left, top)) / gain
    xyxy = np.clip(xyxy, 0, (width, height, width, height))
    valid = (xyxy[:, 0] < xyxy[:, 2]) & (xyxy[:, 1] < xyxy[:, 3])
    xyxy, scores = xyxy[valid], boxes[valid, 4]
    order = np.lexsort((xyxy[:, 0], (xyxy[:, 1] + xyxy[:, 3]) / 2))
    rounded = np.column_stack((np.floor(xyxy[:, :2]), np.ceil(xyxy[:, 2:]))).astype(
        np.int32
    )
    return [
        {
            "line_index": index,
            "x_min": x0,
            "y_min": y0,
            "x_max": x1,
            "y_max": y1,
            "confidence": score,
        }
        for index, ((x0, y0, x1, y1), score) in enumerate(
            zip(rounded[order].tolist(), scores[order].tolist(), strict=True)
        )
    ]


class LineBoxPredictor:
    def __init__(self, options: InferenceOptions):
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionPredictor

        class BoxPredictor(DetectionPredictor):
            def pre_transform(self, images):
                return images  # Worker threads already resized and padded the batch.

            def postprocess(self, preds, img, orig_imgs, **kwargs):
                if self.args.task == "segment":
                    preds = preds[0]  # Unwrap the segmentation output before box NMS.
                return super().postprocess(preds, img, orig_imgs, **kwargs)

        if not options.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {options.checkpoint}")
        if options.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use a CUDA-enabled PyTorch build or select device='cpu'."
            )
        self.yolo = YOLO(str(options.checkpoint.resolve()))
        if self.yolo.task not in {"segment", "detect"}:
            raise ValueError(
                f"The {self.yolo.task!r} checkpoint does not produce per-instance boxes."
            )
        line_classes = [
            index for index, label in self.yolo.names.items() if label == "line"
        ]
        if len(line_classes) != 1:
            raise ValueError(
                f"Expected one class named 'line', found {self.yolo.names}."
            )
        stride = int(self.yolo.model.stride.max())
        if options.img_size < stride or options.img_size % stride:
            raise ValueError(
                f"Image size must be a positive multiple of model stride {stride}."
            )
        if options.decode_workers < 1 or not (
            0 <= options.confidence <= 1 and 0 <= options.iou <= 1
        ):
            raise ValueError(
                "Use positive decode workers and confidence and IoU thresholds in [0, 1]."
            )
        cv2.setNumThreads(1)
        self.options = options
        self.predict_args = {
            "predictor": BoxPredictor,
            "task": self.yolo.task,
            "classes": line_classes,
            "imgsz": options.img_size,
            "conf": options.confidence,
            "iou": options.iou,
            "rect": False,
            "device": options.device,
            "quantize": options.precision,
            "verbose": False,
            "save": False,
        }

    def predict(self, prepared):
        valid = [
            (row, image, transform)
            for row, image, transform in prepared
            if image is not None
        ]
        if valid:
            results = self.yolo.predict(
                [image for _, image, _ in valid], **self.predict_args
            )
            counts = [len(result.boxes) for result in results]
            boxes = (
                torch.cat([result.boxes.data for result in results])
                .float()
                .cpu()
                .numpy()
            )
            offset = 0
            for (row, _, transform), count in zip(valid, counts, strict=True):
                row["lines"] = format_lines(
                    boxes[offset : offset + count],
                    row["width"],
                    row["height"],
                    transform,
                )
                offset += count
        return [row for row, _, _ in prepared]

    def batches(self, root, paths, batch_size, pool):
        chunks = iter(
            paths[i : i + batch_size] for i in range(0, len(paths), batch_size)
        )

        def submit(chunk):
            return [
                pool.submit(prepare_page, root, path, self.options.img_size)
                for path in chunk
            ]

        pending = deque(
            submit(chunk) for chunk in (next(chunks, []), next(chunks, [])) if chunk
        )
        while pending:
            prepared = [future.result() for future in pending.popleft()]
            yield self.predict(prepared)
            if chunk := next(chunks, []):
                pending.append(submit(chunk))


def benchmark(
    root, options, batch_sizes=(4, 8, 16, 24, 32), sample_size=128, manifest=None
):
    """Measure decode plus prediction throughput on evenly spaced source files."""
    root = Path(root).resolve()
    paths = image_paths(root, manifest)
    if sample_size < 1 or not batch_sizes or min(batch_sizes) < 1:
        raise ValueError("Sample size and candidate batch sizes must be positive.")
    sample = [
        paths[i]
        for i in np.linspace(0, len(paths) - 1, min(sample_size, len(paths)), dtype=int)
    ]
    candidates = sorted({min(size, len(sample)) for size in batch_sizes})
    model = LineBoxPredictor(options)
    stats = []
    with ThreadPoolExecutor(max_workers=options.decode_workers) as pool:
        # Read the same files once before timing to give each batch size the same disk-cache conditions.
        for path in sample:
            try:
                with (root / path).open("rb") as stream:
                    while stream.read(1 << 20):
                        pass
            except OSError:
                continue  # The timed decode accounts for unreadable files.
        for size in candidates:
            try:
                # Warm up the model at the exact shape that is about to be timed.
                list(model.batches(root, sample[:size], size, pool))
                start = perf_counter()
                pages = sum(
                    row["status"] == "ok"
                    for batch in model.batches(root, sample, size, pool)
                    for row in batch
                )
                elapsed = perf_counter() - start
            except torch.cuda.OutOfMemoryError:
                print(f"Batch {size}: CUDA out of memory", flush=True)
                torch.cuda.empty_cache()
                break
            stats.append(
                {
                    "batch_size": size,
                    "pages": pages,
                    "seconds": elapsed,
                    "pages_per_second": pages / elapsed,
                }
            )
            print(
                f"Batch {size}: {pages / elapsed:.1f} pages/s ({pages}/{len(sample)} decoded)",
                flush=True,
            )
    if not stats or not max(candidate["pages"] for candidate in stats):
        raise RuntimeError("No benchmark candidate processed any pages.")
    return stats


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def export_line_boxes(
    root, output, options, batch_size=16, pages_per_shard=512, manifest=None, limit=None
):
    root, output = Path(root).resolve(), Path(output).resolve()
    if batch_size < 1 or pages_per_shard < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size, shard size and image limit must be positive.")
    if output == root or output.is_relative_to(root):
        raise ValueError("Place the output outside the source image directory.")
    profile = {
        "schema_version": 1,
        "ultralytics_version": version("ultralytics"),
        "root": str(root),
        "checkpoint_sha256": sha256(options.checkpoint),
        "img_size": options.img_size,
        "confidence": options.confidence,
        "iou": options.iou,
        "precision": options.precision,
        "pages_per_shard": pages_per_shard,
        "limit": limit,
        "manifest_sha256": sha256(manifest) if manifest is not None else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".run.lock", timeout=0):
        state = output / "run.json"
        shards = output / "pages"
        if state.exists():
            config = json.loads(state.read_text(encoding="utf-8"))
            if config["profile"] != profile:
                raise ValueError(
                    "This output belongs to different inputs or inference settings. Use a new directory."
                )
            paths = config["images"]
            if not root.is_dir():
                raise FileNotFoundError(f"Image root is not a directory: {root}")
        else:
            if shards.exists() and any(shards.glob("*.parquet")):
                raise ValueError(
                    "Parquet shards exist without run.json. Use a new output directory."
                )
            paths = image_paths(root, manifest)[:limit]
            partial = state.with_suffix(".tmp")
            partial.write_text(
                json.dumps({"profile": profile, "images": paths}, ensure_ascii=False),
                encoding="utf-8",
            )
            partial.replace(state)

        shards.mkdir(exist_ok=True)
        model = None
        with (
            ThreadPoolExecutor(max_workers=options.decode_workers) as pool,
            tqdm(total=len(paths), unit="page") as progress,
        ):
            for shard_index, start in enumerate(range(0, len(paths), pages_per_shard)):
                target = shards / f"part-{shard_index:06d}.parquet"
                selected = paths[start : start + pages_per_shard]
                if target.exists():
                    with pq.ParquetFile(target) as parquet:
                        if parquet.metadata.num_rows != len(selected):
                            raise ValueError(
                                f"Unexpected page count in completed shard: {target}"
                            )
                else:
                    if model is None:
                        model = LineBoxPredictor(options)
                    rows = []
                    for batch in model.batches(root, selected, batch_size, pool):
                        rows.extend(batch)
                        progress.update(len(batch))
                    partial = target.with_suffix(".tmp")
                    pq.write_table(
                        pa.Table.from_pylist(rows, schema=PAGE_SCHEMA),
                        partial,
                        compression="zstd",
                    )
                    partial.replace(target)
                    continue
                progress.update(len(selected))
    return shards


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO line boxes as resumable Parquet shards"
    )
    parser.add_argument("source", type=Path, help="Root directory of the page images")
    parser.add_argument(
        "--manifest", type=Path, help="Optional text file with one image path per line"
    )
    parser.add_argument("--output", type=Path, default=Path("runs/line_boxes"))
    parser.add_argument("--checkpoint", type=Path, default=Path("Models/best.pt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.38)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pages-per-shard", type=int, default=512)
    parser.add_argument(
        "--limit", type=int, help="Process only the first N pages into a trial output"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare batch sizes without writing any output",
    )
    parser.add_argument("--sample-size", type=int, default=128)
    args = parser.parse_args()
    options = InferenceOptions(
        args.checkpoint,
        args.img_size,
        args.confidence,
        args.iou,
        args.device,
        args.fp32,
        args.decode_workers,
    )
    if args.benchmark:
        stats = benchmark(
            args.source, options, sample_size=args.sample_size, manifest=args.manifest
        )
        print(
            f"Best tested batch size: {max(stats, key=lambda item: item['pages_per_second'])['batch_size']}"
        )
    else:
        shards = export_line_boxes(
            args.source,
            args.output,
            options,
            args.batch_size,
            args.pages_per_shard,
            args.manifest,
            args.limit,
        )
        print(shards)


if __name__ == "__main__":
    main()
