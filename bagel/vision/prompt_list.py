"""1080p 交互提示列表。高亮框定位选中行，图标与文字共同确定条目。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from time import perf_counter
from typing import TYPE_CHECKING

import cv2
import numpy as np
from one_dragon.base.geometry.rectangle import Rect
from one_dragon.utils import cv2_utils, str_utils
from one_dragon.utils.i18_utils import gt

from .template import load_template

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from zzz_od.context.zzz_context import ZContext


ICON_ROW_TOLERANCE: int = 12
"""图标行心与条目行心的最大纵向差（像素）。"""

ICON_COLUMN: tuple[int, int] = (1324, 1404)
"""按行找图标时的横向范围（整屏 x，左闭右开）。

归档提示截图里三种图标的左边缘都在 1336–1338，模板宽 52，右边缘约 1390。
"""


def highlight_center(screen: MatLike) -> int | None:
    """找同时点亮的左右圆弧。背景零星同色或多个候选时不认选中行。"""
    crop = screen[430:760, 1320:1631]
    if crop.shape[:2] != (330, 311):
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (24, 100, 120), (43, 255, 255))
    left = np.count_nonzero(mask[:, :17], axis=1) >= 2
    right = np.count_nonzero(mask[:, -17:], axis=1) >= 2
    ys = np.flatnonzero(left & right)
    bands = np.split(ys, np.flatnonzero(np.diff(ys) > 2) + 1)
    centers = [430 + int((b[0] + b[-1]) // 2) for b in bands if 30 <= len(b) <= 68]
    return centers[0] if len(centers) == 1 else None


def _container_kind(text: str) -> str:
    """容器图标相同，电子与机械保险箱允许名称缺一个字。"""
    if not text or any(word in text for word in ('药粉', '拾取')):
        return 'unknown'
    for kind, title in (('electronic_safe', '电子保险箱'), ('mechanical_safe', '机械保险箱')):
        if str_utils.find_by_lcs(gt(title, 'game'), text, 0.7):
            return kind
    return 'door' if gt('门', 'game') in text else 'box'


@dataclass(frozen=True)
class PromptRow:
    """整屏行心、交互类型和可见文字。"""

    y: int
    kind: str
    text: str

    def matches(self, other: PromptRow | None) -> bool:
        """允许高亮边缘有少量像素抖动。"""
        return (other is not None and self.kind == other.kind
                and self.text == other.text and abs(self.y - other.y) <= 4)


class PromptFrame:
    """同一截图只读一次。移动时只需要选中行，到点时才展开可见列表。"""

    on_cost: Callable[[str, float], None] | None = None
    """计时回调：段名和耗时（秒），只用于日志。"""

    def __init__(
        self, ctx: ZContext, screen: MatLike, stamp: float,
        on_cost: Callable[[str, float], None] | None = None,
    ) -> None:
        self.ctx: ZContext = ctx
        self.screen: MatLike = screen
        self.stamp: float = stamp
        self.on_cost = on_cost
        start = perf_counter()
        self.center: int | None = highlight_center(screen)
        self._cost('高亮', start)
        self._icons: dict[str, list[int]] = {}
        self._icons_near: dict[tuple[str, int], bool] = {}

    def _cost(self, name: str, start: float) -> None:
        if self.on_cost is not None:
            self.on_cost(name, perf_counter() - start)

    def icons(self, name: str) -> list[int]:
        """缓存某种图标的全部行心，也供长按期间查询被遮挡的条目。"""
        if name not in self._icons:
            start = perf_counter()
            rows: list[int] = []
            area = self.ctx.screen_loader.get_area('贝果-局内', '交互图标')
            template = load_template(name)
            if area is not None and template is not None:
                crop = cv2_utils.crop_image_only(self.screen, area.pc_rect)
                hits = cv2_utils.match_template(crop, template[0], 0.8, mask=template[1],
                                                only_best=False, ignore_inf=True)
                rows = sorted(area.pc_rect.y1 + hit.center.y for hit in hits)
            self._icons[name] = rows
            self._cost('图标', start)
        return self._icons[name]

    def icon_near(self, name: str, y: int) -> bool:
        """某行附近有没有这种图标。只匹配该行的图标列，比整区 ``icons`` 快一个数量级。

        Args:
            name: 图标模板 id。
            y: 条目行心（整屏）。
        """
        if name in self._icons:
            return any(abs(y - iy) <= ICON_ROW_TOLERANCE for iy in self._icons[name])
        key = (name, y)
        if key not in self._icons_near:
            start = perf_counter()
            found = False
            template = load_template(name)
            if template is not None:
                half = template[0].shape[0] // 2
                top = y - ICON_ROW_TOLERANCE - half
                bottom = y + ICON_ROW_TOLERANCE + template[0].shape[0] - half + 1
                crop = self.screen[max(top, 0):bottom, ICON_COLUMN[0]:ICON_COLUMN[1]]
                hits = cv2_utils.match_template(crop, template[0], 0.8, mask=template[1],
                                                only_best=False, ignore_inf=True)
                found = any(abs(max(top, 0) + hit.center.y - y) <= ICON_ROW_TOLERANCE for hit in hits)
            self._icons_near[key] = found
            self._cost('图标', start)
        return self._icons_near[key]

    def text_at(self, y: int, right: int = 1575) -> str:
        """保留可能在第二行的物品数量或持卡数。"""
        start = perf_counter()
        hits = self.ctx.ocr_service.get_ocr_result_list(
            image=self.screen, rect=Rect(1400, y - 31, right, y + 31),
        )
        self._cost('OCR', start)
        return ''.join(str_utils.remove_whitespace(hit.data) for hit in hits)

    def _read(self, y: int) -> PromptRow | None:
        if not 460 <= y <= 730:
            return None
        for icon, kind in (('searched_icon', 'searched'), ('card_key_icon', 'card_reader'),
                           ('interact_icon', 'container')):
            if self.icon_near(icon, y):
                text = self.text_at(y, 1620 if kind == 'card_reader' else 1575)
                if kind == 'container':
                    kind = _container_kind(text)
                return PromptRow(y, kind, text)
        text = self.text_at(y)
        if len(re.findall(r'[一-鿿]', text)) < 2:
            return None
        ambiguous = any(c in text for c in ('箱', '柜', '包', '门', '医疗站', '垃圾桶')) or re.search(r'\d/\d', text)
        return PromptRow(y, 'unknown' if ambiguous else 'pickup', text)

    @cached_property
    def selected(self) -> PromptRow | None:
        return self._read(self.center) if self.center is not None else None

    @cached_property
    def rows(self) -> list[PromptRow]:
        if self.center is None:
            return []
        rows = [self.selected] if self.selected else []
        for direction in (-1, 1):
            for step in (1, 2):
                row = self._read(round(self.center + direction * step * 78.5))
                if row is None:
                    break
                rows.append(row)
        return sorted(rows, key=lambda row: row.y)

    def step_toward(self, kind: str) -> int | None:
        """返回单格方向，0 表示目标已选中，None 表示当前没有可执行动作。"""
        selected = self.selected
        if selected is None:
            return None
        if selected.kind == kind:
            return 0 if selected.text else None
        for row in self.rows:
            if row.kind == kind:
                return 1 if row.y > selected.y else -1
        # 三行视窗可能只露出列表的一部分，向下滚一格再找。
        return 1 if len(self.rows) >= 3 else None


class PromptSelection:
    """每次新截图观察一次。输入以后，必须从下一张新图重新判稳。"""

    def __init__(self) -> None:
        self.stamp: float = float('-inf')
        self.previous: PromptRow | None = None
        self.stable: bool = False

    def reset(self, stamp: float) -> None:
        self.stamp = stamp
        self.previous = None
        self.stable = False

    def observe(self, frame: PromptFrame) -> bool:
        if frame.stamp > self.stamp:
            row = frame.selected
            self.stable = (row is not None and row.matches(self.previous)
                           and frame.stamp - self.stamp <= 1.5)
            self.previous = row
            self.stamp = frame.stamp
        return self.stable and frame.stamp == self.stamp
