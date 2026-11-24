"""有效窗口表在出生点、底图边缘和非标准裁图上的定位回归。"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bagel.vision.map_locator import MapLocator, MultiMapLocator, load_map_locators


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


@pytest.mark.parametrize(('height', 'width', 'offset'), [(201, 201, 0), (85, 110, 80)])
def test_clipped_window_and_blank_region(tmp_path: Path, height: int, width: int, offset: int) -> None:
    """偏移底图的贴边位置能命中，纯色区域和遮挡帧不能命中。"""
    pixels = np.zeros((320, 360, 4), np.uint8)
    pixels[:, :, 3] = 255
    pixels[:80, :80, :3] = np.random.default_rng(17).integers(256, size=(80, 80, 3), dtype=np.uint8)
    path = tmp_path / 'layer.png'
    Image.fromarray(pixels).save(path)
    locator = MapLocator('测试层', path, origin=(70, 90))
    # 两种尺寸的搜索范围均延伸到补边外。
    texture = locator._texture[offset:offset + height, offset:offset + width].copy()
    center = (70 + offset + width // 2 - locator._pad, 90 + offset + height // 2 - locator._pad)
    hit = locator.search_with_texture(texture, center, 100)
    assert hit is not None and hit[1] == center
    assert locator.search_with_texture(texture, (410, 390), 3) is None
    assert locator.search_with_texture(np.zeros_like(texture)) is None
