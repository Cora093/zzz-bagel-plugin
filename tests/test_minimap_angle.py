"""镜头朝向识别在朝向素材上的回归。"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bagel.vision.minimap_angle import analyze

DIRECTORY = Path(__file__).resolve().parents[1] / 'docs/materials/minimap_heading'
FRAMES: list[dict] = json.loads((DIRECTORY / 'frames.json').read_text(encoding='utf-8'))
KNOWN_MISREADS = {'none_elevator_fade.webp', 'none_dark_particles.webp'}
"""文档「已知限制」里记录的两张误读图，本测试不要求它们返回 None。"""


def _read(file: str) -> float | None:
    with Image.open(DIRECTORY / file) as image:
        return analyze(np.asarray(image.convert('RGB')))[2]


def _error(angle: float, reference: float) -> float:
    return abs((angle - reference + 180.0) % 360.0 - 180.0)


def test_fitted_references_keep_baseline() -> None:
    """有拟合参考方向的 50 张图与换严门前一致：yellowbg_234 越过污染线，但两门读数差不到 15 度，不换门。"""
    errors = []
    for record in FRAMES:
        if record['reference'] is None or 'fit' not in record:
            continue
        angle = _read(record['file'])
        assert angle is not None, record['file']
        errors.append(_error(angle, record['reference']))
    assert len(errors) == 50
    assert float(np.mean(errors)) <= 1.25
    assert max(errors) <= 7.8


@pytest.mark.parametrize(
    'record', [r for r in FRAMES if r['scene'] == 'poster'], ids=lambda r: r['file'],
)
def test_yellow_poster_background(record: dict) -> None:
    """黄色海报整片落进黄门时，换严门后仍读到扇区方向。参考方向是估计值，留 10 度余量。"""
    angle = _read(record['file'])
    assert angle is not None
    assert _error(angle, record['reference']) <= 10.0


@pytest.mark.parametrize(
    'record',
    [r for r in FRAMES if r['reference'] is None and r['file'] not in KNOWN_MISREADS],
    ids=lambda r: r['file'],
)
def test_no_minimap_returns_none(record: dict) -> None:
    """没有小地图的画面读不到朝向。"""
    assert _read(record['file']) is None
