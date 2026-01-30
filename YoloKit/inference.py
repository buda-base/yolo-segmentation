import os
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import random

from glob import glob
from natsort import natsorted

from numpy.typing import NDArray
from tqdm import tqdm
from ultralytics import YOLO
from YoloKit.Processing.image import (
    cluster_lines,
    collect_global_line_masks,
    contour_to_original_space,
    extract_line_images,
    mask_to_contour,
    merge_line_masks,
    resize_and_pad,
    tile_image
)
from YoloKit.utils import create_dir, get_filename
from YoloKit.data import TileData

line_parquet_scheme = {
    "image_name": str,
    "line_id": int,
    "contour_x": list[float],
    "contour_y": list[float],
    "bbox": list[float],  # [x_min, y_min, x_max, y_max]
    "area": float,
    "n_points": int,
}


class YoloInference:
    def __init__(self, model_checkpoint: str, task: str = "segment"):
        self.model = YOLO(model_checkpoint, task=task)

    def _has_results(self, results) -> bool:
        return (
            results is not None and len(results) > 0
        )  # revise this to check for actual mask objects

    def _preprocess_image_cpu(image_path: str) -> tuple[str, NDArray]:
        pass

    def _post_process_cpu(
        self, results: list, tile_data: list[TileData], image: NDArray
    ):
        global_masks, y_centers_raw = collect_global_line_masks(
            results, tile_data, page_shape=image.shape, class_name="line"
        )

        order = np.argsort(y_centers_raw)
        global_masks = global_masks[order]
        y_centers_raw = y_centers_raw[order]

        if len(y_centers_raw) == 0:
            return []

        clusters = cluster_lines(y_centers_raw, dup_eps=8, spacing_factor=0.5)
        return merge_line_masks(global_masks, clusters)

    def _post_process_gpu():
        # TODO
        pass

    def _run_prediction(self, tile_data: list[TileData]) -> tuple:
        img_tiles = [t.img for t in tile_data]
        return self.model(img_tiles, verbose=False)

    def _prediction_to_parquet(
        self, image_name: str, image_path: str, post_process_mode: str = "cpu"
    ):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_padded, meta = resize_and_pad(image_name, img)

        tile_data = tile_image(img_padded, overlap=0.2)
        results = self._run_prediction(tile_data)

        merged_masks = self._post_process_cpu(results, tile_data)

        records = []

        if len(merged_masks) == 0:
            return records

        if merged_masks is None:
            return records

        for i, line_mask in enumerate(merged_masks):
            contour_padded = mask_to_contour(line_mask)
            contour_orig = contour_to_original_space(contour_padded, meta)

            xs = contour_orig[:, 0]
            ys = contour_orig[:, 1]

            record = {
                "image_name": image_name,
                "line_id": i,
                "contour_x": xs.tolist(),
                "contour_y": ys.tolist(),
                "bbox": [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ],
                "area": float(cv2.contourArea(contour_orig.astype(np.float32))),
                "n_points": len(xs),
            }
            records.append(record)

        return records

    """
    Public interface to:
    1) generate ocr_lines based on an image path. This should be called from a loop in the OCRPipeline class
    2) just generate parquet data from an input directory
    3) generate preview masks, mostly useful to run debug passes on various directories
    """

    def get_ocr_lines(self, image_path: str, post_process_mode: str = "cpu"):
        image_name = get_filename(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_padded, meta = resize_and_pad(image_name, img)

        tile_data = tile_image(img_padded, overlap=0.2)
        results = self._run_prediction(tile_data)

        # TODO: handle cpu and gpu post_process_mode
        merged_masks = self._post_process_cpu(results, tile_data)

        if len(merged_masks) == 0:
            return []

        ocr_lines = extract_line_images(img_padded, merged_masks, background="white")

        return ocr_lines

    def generate_parquet_data(self, directory: str, out_dir: str):
        images = natsorted(glob(f"{directory}/*.jpg"))

        if not os.path.isdir(out_dir):
            create_dir(out_dir)

        for image_path in tqdm(images, total=len(images)):
            image_name = get_filename(image_path)
            records = self._prediction_to_parquet(image_name, image_path)

            if len(records) > 0:
                df = pd.DataFrame(records)
                out_file = f"{out_dir}/{image_name}.parquet"
                pq.write_table(pa.Table.from_pandas(df), out_file)

    def generate_debug_output(self, directory: str, out_dir: str, alpha: float = 0.4):
        images = natsorted(glob(f"{directory}/*.jpg"))

        cached_colors = []
        for i in range(40):
            color = [
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ]
            cached_colors.append(color)

        if not os.path.isdir(out_dir):
            create_dir(out_dir)

        for image_path in tqdm(images, total=len(images)):
            image_name = get_filename(image_path)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_padded, meta = resize_and_pad(image_name, img)

            tile_data = tile_image(img_padded, overlap=0.2)
            results = self._run_prediction(tile_data)

            # TODO: handle cpu and gpu post_process_mode
            merged_masks = self._post_process_cpu(results, tile_data, img_padded)

            if len(merged_masks) == 0:
                continue

            preview_img = img.copy()
            preview_mask = np.zeros(img.shape, dtype=np.uint8)

            for idx, m_mask in enumerate(merged_masks):
                contour_padded = mask_to_contour(m_mask)
                contour_orig = contour_to_original_space(contour_padded, meta)
                contour_orig = contour_orig.astype(np.int32)

                cv2.drawContours(
                    preview_mask,
                    [contour_orig],
                    -1,
                    color=cached_colors[idx],
                    thickness=-1,
                )

            cv2.addWeighted(preview_mask, alpha, preview_img, 1 - alpha, 0, preview_img)

            out_file = f"{out_dir}/{image_name}_prev.jpg"
            cv2.imwrite(out_file, preview_img)
