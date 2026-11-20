"""用仓库内的实机裁剪图测识别，用可控帧序列测选行。"""

from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from one_dragon.base.geometry.rectangle import Rect

from bagel.vision.prompt_list import (
    PromptFrame,
    PromptRow,
    PromptSelection,
    highlight_center,
)

from .prompt_frames import ObservedFrame


def archived_screen(name: str) -> np.ndarray:
    """将归档裁剪图放回原来的 1080p 坐标。"""
    path = Path(__file__).resolve().parents[1] / 'docs' / 'materials' / 'keycard' / f'{name}.webp'
    crop = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    screen = np.zeros((1080, 1920, 3), np.uint8)
    screen[430:760, 1260:1650] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return screen


@pytest.mark.parametrize(('name', 'expected_y'), [
    ('prompt_aligned', 595), ('prompt_pickup_over_safe', 555),
    ('prompt_pickup_over_box', 515), ('prompt_searched', 675),
    ('prompt_done_zero', None), ('prompt_pickup_above', None),
])
def test_archived_highlight(name: str, expected_y: int | None) -> None:
    center = highlight_center(archived_screen(name))
    if expected_y is None:
        assert center is None
    else:
        assert center == pytest.approx(expected_y, abs=2)


def test_one_side_or_two_highlights_have_no_unique_selection() -> None:
    screen = archived_screen('prompt_aligned')
    screen[430:530] = screen[545:645].copy()
    assert highlight_center(screen) is None
    screen[:, 1610:] = 0
    assert highlight_center(screen) is None


@pytest.mark.parametrize(('name', 'labels', 'kinds'), [
    ('prompt_aligned', {595: '汽车旅馆二层1/1'}, ['card_reader']),
    ('prompt_searched', {675: '公文包'}, ['searched']),
    ('prompt_pickup_over_safe', {555: '诡异药粉', 634: '机械保险'}, ['pickup', 'mechanical_safe']),
    ('prompt_pickup_over_box', {515: '诡异药粉', 594: '纳米纤维×3', 672: '公文包'},
     ['pickup', 'pickup', 'box']),
])
def test_visible_rows_use_real_icons_and_row_text(name: str, labels: dict[int, str], kinds: list[str]) -> None:
    ctx = Mock()
    ctx.screen_loader.get_area.return_value.pc_rect = Rect(1324, 430, 1640, 760)

    def read_crop(*, image: np.ndarray, rect: Rect) -> list[Mock]:
        """固定文字以隔离 OCR 模型，仍以执行器实际请求的行位置取词。"""
        return [Mock(data=text) for y, text in labels.items() if rect.y1 < y < rect.y2]

    ctx.ocr_service.get_ocr_result_list.side_effect = read_crop
    frame = PromptFrame(ctx, archived_screen(name), 1)
    assert [row.kind for row in frame.rows] == kinds
    assert [row.text for row in frame.rows] == list(labels.values())
    assert frame.selected in frame.rows


@pytest.mark.parametrize('name', ['prompt_aligned', 'prompt_done_zero', 'prompt_holding', 'prompt_pickup_above',
                                  'prompt_pickup_over_box', 'prompt_pickup_over_safe', 'prompt_searched'])
def test_row_icon_lookup_matches_full_area_scan(name: str) -> None:
    """在 _read 会查的行（选中行与上下两行）上，按行只扫图标列的结论须与整区扫描一致。"""
    ctx = Mock()
    ctx.screen_loader.get_area.return_value.pc_rect = Rect(1324, 430, 1640, 760)
    full, near = PromptFrame(ctx, archived_screen(name), 1), PromptFrame(ctx, archived_screen(name), 1)
    center = full.center or 595
    for icon in ('searched_icon', 'card_key_icon', 'interact_icon'):
        for y in (round(center + k * 78.5) for k in (-2, -1, 0, 1, 2)):
            assert near.icon_near(icon, y) == any(abs(y - iy) <= 12 for iy in full.icons(icon)), (icon, y)


@pytest.mark.parametrize(('row', 'elapsed', 'stable'), [
    (PromptRow(559, 'pickup', '纳米纤维×4'), 1.5, True),
    (PromptRow(560, 'pickup', '纳米纤维×4'), 0.1, False),
    (PromptRow(555, 'pickup', '纳米纤维×4'), 1.51, False),
    (PromptRow(555, 'pickup', '纳米纤维×3'), 0.1, False),
    (PromptRow(555, 'searched', '纳米纤维×4'), 0.1, False),
    (None, 0.1, False),
])
def test_selection_needs_same_item_in_two_fresh_frames(row: PromptRow | None, elapsed: float, stable: bool) -> None:
    selection = PromptSelection()
    first = ObservedFrame(10, PromptRow(555, 'pickup', '纳米纤维×4'))
    assert not selection.observe(first)
    assert not selection.observe(first)
    assert selection.observe(ObservedFrame(10 + elapsed, row)) is stable


def test_input_invalidates_every_frame_up_to_the_input_time() -> None:
    selection = PromptSelection()
    row = PromptRow(595, 'box', '公文包')
    assert not selection.observe(ObservedFrame(1, row))
    assert selection.observe(ObservedFrame(1.1, row))
    selection.reset(1.3)
    for stamp in (1.1, 1.3, 1.4):
        assert not selection.observe(ObservedFrame(stamp, row))
    assert selection.observe(ObservedFrame(1.5, row))


def test_scroll_one_step_and_search_hidden_rows_only_in_full_window() -> None:
    upper = PromptRow(515, 'mechanical_safe', '机械保险箱')
    middle = PromptRow(595, 'searched', '公文包')
    lower = PromptRow(675, 'electronic_safe', '电子保险箱')
    frame = ObservedFrame(1, middle, upper, lower)
    assert frame.step_toward('mechanical_safe') == -1
    assert frame.step_toward('electronic_safe') == 1
    assert frame.step_toward('card_reader') == 1
    assert ObservedFrame(2, lower, middle).step_toward('card_reader') is None
    assert ObservedFrame(3, lower).step_toward('electronic_safe') == 0
    assert ObservedFrame(4, PromptRow(675, 'box', '')).step_toward('box') is None
