from dataclasses import dataclass
from numpy.typing import NDArray
from shapely.geometry import Polygon


@dataclass
class ResizePadData:
    orig_w: int
    orig_h: int
    scale: float
    resized_w: int
    resized_h: int
    pad_w: int
    pad_h: int
    padded_w: int
    padded_h: int


@dataclass
class TileData:
    tile_id: int
    x0: int
    y0: int
    tile_size: int
    img: NDArray


@dataclass
class PolyData:
    img_name: str
    polygon: Polygon


@dataclass
class TilePrediction:
    tile_id: int
    class_id: int
    confidence: float
    mask: NDArray


@dataclass
class InstanceRecord:
    image_id: str
    instance_id: int
    class_id: int
    confidence: float
    polygon: list[tuple[float, float]]
    tile_id: int
    tile_x0: int
    tile_y0: int
    scale: float
