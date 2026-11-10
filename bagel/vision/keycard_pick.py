"""仓库挑门禁卡用到的画面几何与识别。

坐标取自用户录屏（1920×1080，城郊幻境高危备战仓库），录屏是有损编码；
相似度和行比对阈值按实机清点拼图标定。
挑卡时图标模板只给候选格排序，挑哪张以点开浮窗后 OCR 到的标题为准；
按保留张数清点时只看图标，不点开。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from one_dragon.base.geometry.point import Point
from one_dragon.base.geometry.rectangle import Rect
from one_dragon.utils import str_utils

from .slots import slot_occupied
from .template import load_template

if TYPE_CHECKING:
    from cv2.typing import MatLike

GRID_XS: tuple[int, ...] = (1120, 1224, 1328, 1432, 1536, 1640, 1744)
"""仓库网格 7 列的格心横坐标。"""
GRID_RECT: Rect = Rect(1062, 268, 1802, 872)
"""仓库网格可见区域；滚动后行的纵坐标会变，行心由 ``grid_rows`` 现找。"""
GRID_SCROLL_POS: Point = Point(1432, 570)
"""在网格上滚动鼠标滚轮的位置。"""
PAGE_CENTERS: tuple[Point, ...] = tuple(Point(x, 143) for x in (1100, 1211, 1321, 1432, 1542, 1653, 1763))
"""仓库分页页签 1–7 的中心；录屏里 6、7 页是锁定的。"""
FILTER_ALL_POS: Point = Point(1100, 50)
"""网格上方类型筛选的「全部」。"""
FILTER_KEYCARD_POS: Point = Point(1655, 50)
"""类型筛选的「门禁卡」（门形图标），在「贵重物品」与「其他」之间。"""
LOCK_CONFIRM_RECT: Rect = Rect(880, 635, 1090, 700)
"""点锁定分页弹出「可通过以下途径解锁仓储枢纽容量」，这是它的「确认」按钮。"""

TOOLTIP_TITLE_RECT: Rect = Rect(605, 228, 862, 268)
"""点开物品后浮窗的标题（物品名）。浮窗位置固定，不随点的格子移动。"""
TOOLTIP_TYPE_RECT: Rect = Rect(605, 268, 760, 298)
"""浮窗标题下的类型标签，门禁卡显示「门禁卡」。"""
TOOLTIP_DISMISS_POS: Point = Point(1350, 239)
"""仓库网格标题栏的空白处；点它关掉物品浮窗，不触发别的操作。浮窗开着时滚轮滚不动网格。"""
TOOLTIP_SAFE_BUTTON_RECT: Rect = Rect(800, 718, 988, 766)
"""浮窗的「放入安全箱」按钮（快捷键 S）。"""

_ROW_SAT_MIN: int = 110
_ROW_VAL_MIN: int = 90
"""卡片底色（黄、红、蓝、紫）的饱和度与亮度下限；格子空时是暗灰，不会过线。"""
_ROW_FILL_MIN: float = 0.05
"""一行像素里卡片底色的占比下限；一行只有一张卡时约 0.1。"""
_ROW_HEIGHT_RANGE: tuple[int, int] = (60, 100)
"""完整一行卡片底色的高度范围；被滚动截掉一部分的行不在范围内，跳过。"""
_ROW_CENTER_OFFSET: int = -2
"""卡片底色中线到格心的偏移；录屏里底色 y≈404–481，格心取 442。"""
_ICON_MARGIN: int = 10
"""图标排序时在模板四周多留的搜索边距，容许行心估计有几像素偏差。"""
_ICON_BLUR: float = 1.5
"""整张卡面比对前的高斯模糊；不模糊时同款卡因所在行不同，相似度在 0.93–0.99 间浮动。"""
_LETTER_RECT: tuple[int, int, int, int] = (12, 4, 32, 28)
"""卡面左上的字母徽标在模板里的范围 (x1, y1, x2, y2)，各房间的卡在这里区分 A–E。"""
_LETTER_BLUR: float = 1.0
_LETTER_MARGIN: int = 6
_STRIP_HALF_HEIGHT: int = 24
"""比较滚动前后行内容时，每行取格心上下各这么多像素；滚到顶边的行会被裁掉一截，所以只取中段。"""
_STRIP_SHIFT: int = 8
"""比较两行时容许的上下错位；被裁掉一截的行，行心估计会偏几像素。"""
_STRIP_DIFF_MAX: float = 6.5
"""两行灰度图错位对齐后的平均差不超过它，算同一行。实机同一行 2–6，相邻不同行 7 以上。"""


@dataclass(frozen=True)
class GridCell:
    """当前画面里一个有物品的格子。"""

    center: Point
    score: float
    """与目标卡图标的相似度；没有模板时为 0。"""


def grid_rows(screen: MatLike) -> list[int]:
    """找出网格里完整显示的卡片行，返回各行格心的纵坐标。

    按每一行像素中卡片底色的占比分段，高度合适的段才算完整一行。
    整行都是空格时找不到，这样的行也没有可挑的卡。
    """
    crop = screen[GRID_RECT.y1:GRID_RECT.y2, GRID_RECT.x1:GRID_RECT.x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] >= _ROW_SAT_MIN) & (hsv[:, :, 2] >= _ROW_VAL_MIN)
    filled = mask.mean(axis=1) >= _ROW_FILL_MIN
    rows: list[int] = []
    start: int | None = None
    for index, value in enumerate([*filled.tolist(), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            height = index - start
            if _ROW_HEIGHT_RANGE[0] <= height <= _ROW_HEIGHT_RANGE[1]:
                rows.append(GRID_RECT.y1 + (start + index) // 2 + _ROW_CENTER_OFFSET)
            start = None
    return rows


def grid_cells(screen: MatLike, template_id: str | None) -> list[GridCell]:
    """列出当前画面里有物品的格子，按与目标卡图标的相似度从高到低排序。

    Args:
        screen: 整屏截图，RGB。
        template_id: 目标卡图标模板；None 或模板缺失时不排序，按从上到下、从左到右。
    """
    loaded = load_template(template_id) if template_id is not None else None
    raw = loaded[0] if loaded is not None else None
    cells: list[GridCell] = []
    for y in grid_rows(screen):
        for x in GRID_XS:
            center = Point(x, y)
            if not slot_occupied(screen, center):
                continue
            cells.append(GridCell(center, _icon_score(screen, center, raw)))
    if raw is not None:
        cells.sort(key=lambda cell: -cell.score)
    return cells


def _icon_score(screen: MatLike, center: Point, raw: MatLike | None) -> float:
    """格子图标与模板的相似度：整张卡面与字母徽标两项取较低者。

    整张卡面分得开底色和图案不同的卡，但分不开只差字母的卡（如 B 卡与 D 卡同为橙色跑人）；
    字母徽标分得开字母，分不开同字母不同图案的卡。两项都高才算同款。
    """
    if raw is None:
        return 0.0
    h, w = raw.shape[:2]
    origin = Point(center.x - w // 2, center.y - h // 2)
    whole = _match(screen, raw, origin, (0, 0, w, h), _ICON_MARGIN, _ICON_BLUR)
    letter = _match(screen, raw, origin, _LETTER_RECT, _LETTER_MARGIN, _LETTER_BLUR)
    return min(whole, letter)


def _match(screen: MatLike, raw: MatLike, origin: Point, part: tuple[int, int, int, int],
           margin: int, blur: float) -> float:
    """模板的 ``part`` 部分在格子对应位置四周 ``margin`` 像素内的最大相关系数，两边先同样模糊。"""
    x1, y1, x2, y2 = part
    template = raw[y1:y2, x1:x2]
    left = origin.x + x1 - margin
    top = origin.y + y1 - margin
    if left < 0 or top < 0:
        return 0.0
    region = screen[top:top + (y2 - y1) + 2 * margin, left:left + (x2 - x1) + 2 * margin]
    if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
        return 0.0
    template = cv2.GaussianBlur(template, (0, 0), blur)
    region = cv2.GaussianBlur(region, (0, 0), blur)
    return float(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED).max())


def row_strips(screen: MatLike, rows: list[int]) -> list[MatLike]:
    """各行网格的灰度图（上下多留 ``_STRIP_SHIFT`` 像素供错位对齐），用来认出滚动前后重复出现的行。"""
    half = _STRIP_HALF_HEIGHT + _STRIP_SHIFT
    strips: list[MatLike] = []
    for y in rows:
        crop = screen[y - half:y + half, GRID_RECT.x1:GRID_RECT.x2]
        strips.append(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32))
    return strips


def scroll_overlap(before: list[MatLike], after: list[MatLike]) -> int:
    """滚动前最后 k 行与滚动后最前 k 行一一相同，返回最大的 k；没有重叠为 0。

    滚动不按整行对齐，前后两屏同一行的行心可差十几像素，所以按各自的行心取条带再比。
    相邻几行内容完全相同时可能多认重叠、少数几张卡；少数只会让清点偏保守。
    """
    for k in range(min(len(before), len(after)), 0, -1):
        if all(_strip_same(a, b) for a, b in zip(before[-k:], after[:k], strict=True)):
            return k
    return 0


def _strip_same(a: MatLike, b: MatLike) -> bool:
    """a 的中段在 b 里上下错位不超过 ``_STRIP_SHIFT`` 时，最小平均差不超过阈值。"""
    if a.shape != b.shape:
        return False
    height = 2 * _STRIP_HALF_HEIGHT
    core = a[_STRIP_SHIFT:_STRIP_SHIFT + height]
    diff = min(float(np.abs(core - b[dy:dy + height]).mean()) for dy in range(2 * _STRIP_SHIFT + 1))
    return diff <= _STRIP_DIFF_MAX


def grid_signature(screen: MatLike) -> MatLike:
    """网格区域的缩略灰度图，用来判断滚动或翻页后画面有没有变。"""
    crop = screen[GRID_RECT.y1:GRID_RECT.y2, GRID_RECT.x1:GRID_RECT.x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (gray.shape[1] // 8, gray.shape[0] // 8), interpolation=cv2.INTER_AREA)


def grid_changed(before: MatLike, after: MatLike) -> bool:
    """两张网格缩略图的平均差超过阈值才算画面变了。"""
    return float(np.abs(before.astype(np.float32) - after.astype(np.float32)).mean()) >= 3.0


def title_matches(title: str, name: str) -> bool:
    """浮窗标题与房间名一致：允许 OCR 错漏一个字，不允许多出别的房间名。"""
    compact = ''.join(title.split())
    if not compact:
        return False
    common = str_utils.longest_common_subsequence_length(name, compact)
    return common >= len(name) - 1 and len(compact) <= len(name) + 2
