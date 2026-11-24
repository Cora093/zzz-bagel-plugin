"""有效窗口表在出生点、底图边缘和非标准裁图上的定位回归。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from bagel.vision.map_locator import (
    MIN_WINDOW_ENERGY,
    MapLocator,
    MultiMapLocator,
    _valid_windows,
    load_map_locators,
)


def test_recorded_spawn_coordinates() -> None:
    """五张出生画面的全层搜索和随后局部追踪都应回到已知位置。"""
    directory = Path(__file__).resolve().parents[1] / 'docs/materials/spawn_points'
    layers = load_map_locators(map_id='雅努斯幻境')
    for record in json.loads((directory / 'positions.json').read_text(encoding='utf-8')):
        expected = record['hits'][0]
        with Image.open(directory / record['frame']) as image:
            minimap = np.asarray(image.convert('RGB'))[206:407, 1533:1734]
        locator = MultiMapLocator(layers)
        for _ in range(2):
            hit = locator.locate(minimap)
            assert hit is not None
            assert (hit[0].name, hit[2]) == (expected['layer'], tuple(expected['center']))


def test_valid_windows_matches_brute_force() -> None:
    """逐窗口计算方差，核对分批边界两侧的有效性。"""
    texture = np.zeros((96, 36), np.float32)
    texture[70:80, 16:28] = np.random.default_rng(3).normal(size=(10, 12))
    windows = np.lib.stride_tricks.sliding_window_view(texture.astype(np.float64), (21, 17))
    expected = windows.var(axis=(2, 3)) * 21 * 17 > MIN_WINDOW_ENERGY
    np.testing.assert_array_equal(_valid_windows(texture, 21, 17), expected)


@pytest.mark.parametrize(('height', 'width'), [(201, 201), (85, 110)])
def test_search_matches_integral_reference(tmp_path: Path, height: int, width: int) -> None:
    """对照旧积分图的坐标和分数，包括贴边、纯色补边和非标准窗口。"""
    rng = np.random.default_rng(17)
    pixels = np.zeros((320, 360, 4), np.uint8)
    pixels[:, :250, 3] = 255
    pixels[:80, :80, :3] = rng.integers(256, size=(80, 80, 3), dtype=np.uint8)
    pixels[200:, 150:240, :3] = rng.integers(256, size=(120, 90, 3), dtype=np.uint8)
    path = tmp_path / 'layer.png'
    Image.fromarray(pixels).save(path)
    locator = MapLocator('测试层', path, origin=(70, 90))
    base = locator._texture
    sums, squares = cv2.integral2(base, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
    total = (sums[height:, width:] - sums[:-height, width:]
             - sums[height:, :-width] + sums[:-height, :-width])
    energy = (squares[height:, width:] - squares[:-height, width:]
              - squares[height:, :-width] + squares[:-height, :-width])
    valid = energy - total * total / (height * width) > MIN_WINDOW_ENERGY
    # 将候选窗口左上角换为地图上的圆心坐标。
    dx = locator.origin[0] + width // 2 - locator._pad
    dy = locator.origin[1] + height // 2 - locator._pad
    for top, left in [(110, 110), (60, 30), (300, 250)]:
        query = base[top:top + height, left:left + width].copy()
        # 左上角这个单点，201×201 能覆盖纹理，85×110 完全落在补边内。
        for x, y, radius in [(0, 0, 0), (0, 0, None), (30, 60, 80), (250, 300, 10)]:
            if radius is None:
                center = None
                x1, y1, x2, y2 = 0, 0, valid.shape[1], valid.shape[0]
            else:
                center = (x + dx, y + dy)
                x1, y1 = max(0, x - radius), max(0, y - radius)
                x2, y2 = min(valid.shape[1], x + radius + 1), min(valid.shape[0], y + radius + 1)
            region = base[y1:y2 + height - 1, x1:x2 + width - 1]
            scores = cv2.matchTemplate(region, query, cv2.TM_CCOEFF_NORMED)
            scores[~valid[y1:y2, x1:x2] | ~np.isfinite(scores)] = -np.inf
            _, score, _, point = cv2.minMaxLoc(scores)
            expected = (score, (point[0] + x1 + dx, point[1] + y1 + dy)) if np.isfinite(score) else None
            assert locator.search_with_texture(query, center, radius) == expected
