"""仓库挑门禁卡用到的画面几何与识别。

坐标取自用户录屏（1920×1080，城郊幻境高危备战仓库），录屏是有损编码；
相似度阈值按实机清点拼图标定；列表每条滚动命令正好滚一行，按「命令数」给行编号，画面比对只用来发现命令丢了。
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
_TAB_HALF: int = 24
"""判断页签状态时，取页签中心上下左右各这么多像素。"""
_TAB_GOLD_MIN: int = 400
"""页签区域里金色像素不少于它，算当前选中页。实机选中页 846–1338，其余页 0。"""
_TAB_BRIGHT_MIN: int = 100
"""页签区域里亮像素（灰度 ≥ 200）不少于它，算可以翻到的页。实机白色页码 210–260，
筛选后没有内容的暗色页码和锁定页 0；选中页的金色圆也有约 120。"""
TAB_SELECTED: str = 'selected'
TAB_OPEN: str = 'open'
TAB_CLOSED: str = 'closed'
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
_ROW_FILL_MIN: float = 0.025
"""一行像素里卡片底色的占比下限；末页只剩一张白色卡面时，图案遮住底色，
实机逐行最低约 0.032。取 0.025 保留余量，避免把完整卡片切成多个不足 60 px 的短段。"""
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
ROW_PITCH: int = 122
"""相邻两行卡片的行心间距；实机行心 321、443、565、687、809。"""
TOP_ROW_Y: int = 321
"""列表在顶部时第 1 行的行心纵坐标。"""
ARROW_RECT: Rect = Rect(1414, 856, 1454, 872)
"""网格正下方居中的 ▼ 提示区域；列表还能向下滚时显示，到底后淡出消失。"""
_ARROW_GRAY_MIN: int = 200
ARROW_PIXELS_MIN: int = 100
"""区域里亮度不低于 ``_ARROW_GRAY_MIN`` 的像素数不少于它，算「还能向下滚」。
实机还能滚时约 288，到底后淡出，途中约 40，最后为 0。"""
_EFFECT_CELL_HALF: tuple[int, int] = (42, 40)
"""判断滚动效果时每格取格心左右、上下各这么多像素。"""
_EFFECT_CHANGED_MIN: float = 1.0
"""滚动前后同一格的平均灰度差超过它，算画面变了。游戏截图无损，列表不动时实测为 0。"""
_EFFECT_ROW_DIFFER_MIN: float = 4.0
"""相邻两行同一列的 6×6 缩略图平均差不低于它，算两张不同的卡。
实机同款卡 1.5–2.3（上下边缘调暗、亚像素偏移），不同的卡 5.8 以上（含只差底色或字母的卡）。"""
PAGE_ROWS_MAX: int = 8
"""一页最多 8 行（仓库总容量 280 = 5 页 × 56 格，每行 7 格）；一页最多向下滚 ``PAGE_ROWS_MAX - 5`` 次。"""
VISIBLE_ROWS: int = 5
"""网格一屏能完整显示的行数。"""


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


def more_below(screen: MatLike) -> bool:
    """网格下方的 ▼ 提示还在，说明列表还能向下滚。"""
    return arrow_pixels(screen) >= ARROW_PIXELS_MIN


def arrow_pixels(screen: MatLike) -> int:
    """▼ 提示区域里的亮像素数，写进日志供标定。"""
    crop = screen[ARROW_RECT.y1:ARROW_RECT.y2, ARROW_RECT.x1:ARROW_RECT.x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return int((gray >= _ARROW_GRAY_MIN).sum())


def rows_same(before: list[int], after: list[int], tolerance: int = 1) -> bool:
    """两次截图里各完整行的行心纵坐标一一相同（容许 ``tolerance`` 像素），说明列表没在动。"""
    return len(before) == len(after) and all(
        abs(a - b) <= tolerance for a, b in zip(before, after, strict=True))


def row_phase(rows: list[int]) -> float | None:
    """各行行心纵坐标对行距取余的圆周平均，范围 [0, ROW_PITCH)；没有完整行返回 None。

    所有行等距排列，所以它代表列表滚到了哪个位置（以一个行距为周期）。
    """
    if not rows:
        return None
    angles = np.asarray(rows, dtype=np.float64) * (2 * np.pi / ROW_PITCH)
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float(mean % (2 * np.pi) * ROW_PITCH / (2 * np.pi))


def phase_delta(before: float, after: float) -> float:
    """两个相位之差，折到 [-ROW_PITCH/2, ROW_PITCH/2)。"""
    return (after - before + ROW_PITCH / 2) % ROW_PITCH - ROW_PITCH / 2


def _effect_cells(screen: MatLike, rows: list[int]) -> list[list[MatLike]]:
    """各完整行、各列（跳过滚轮光标所在的列，光标和 ▼ 提示都在这一列）的格子灰度图。"""
    half_w, half_h = _EFFECT_CELL_HALF
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY).astype(np.float32)
    xs = [x for x in GRID_XS if abs(x - GRID_SCROLL_POS.x) > half_w]
    return [[gray[y - half_h:y + half_h, x - half_w:x + half_w] for x in xs] for y in rows]


def scroll_effect(before: MatLike, after: MatLike) -> str:
    """判断一条向下滚动命令生效没有：``moved``（滚了）、``still``（没滚）、``unknown``（分不出）。

    - 滚动后任何一格变了就是滚了：游戏截图无损，列表不动时画面逐像素不变。
    - 画面没变，而滚动前相邻两行有不同的卡：真滚了一行画面一定会变，所以是命令没生效。
    - 画面没变、可见各行又全是同款卡：滚没滚画面都一样，分不出。
    两张截图都要在列表停稳后截取。
    """
    rows = grid_rows(after)
    if not rows or grid_rows(before) != rows:
        return 'moved' if rows else 'unknown'
    cells_before = _effect_cells(before, rows)
    cells_after = _effect_cells(after, rows)
    changed = max(float(np.abs(b - a).mean())
                  for row_b, row_a in zip(cells_before, cells_after, strict=True)
                  for b, a in zip(row_b, row_a, strict=True))
    if changed > _EFFECT_CHANGED_MIN:
        return 'moved'
    thumbs = [[cv2.resize(cell, (6, 6), interpolation=cv2.INTER_AREA) for cell in row] for row in cells_before]
    for upper, lower in zip(thumbs, thumbs[1:], strict=False):
        if any(float(np.abs(a - b).mean()) >= _EFFECT_ROW_DIFFER_MIN for a, b in zip(upper, lower, strict=True)):
            return 'still'
    return 'unknown'


class RowTracker:
    """给本页每一行一个绝对行号：每条生效的向下滚动命令正好滚一行（实机探针 1–8 格都是 122 px）。

    行的相位（行心纵坐标对行距取余）因此固定不变，不能用来发现滚动；行号只靠命令数记账，
    相位只用来核对行没有错位。
    """

    def __init__(self, rows: list[int]) -> None:
        """在列表顶部创建：第 1 个完整行记为第 0 行。"""
        phase = row_phase(rows)
        if phase is None:
            raise ValueError('列表里没有完整行，无法建立滚动记账')
        self.origin: int = rows[0]
        self.phase: float = phase
        self.offset: int = 0
        """累计向下滚动的行数。"""

    def advance(self) -> None:
        """记一次生效的向下滚动。"""
        self.offset += 1

    def aligned(self, rows: list[int], tolerance: float = 6.0) -> bool:
        """停稳后的行心相位仍与顶部一致；不一致说明滚动不是整行，记账不可信。"""
        phase = row_phase(rows)
        return phase is None or abs(phase_delta(self.phase, phase)) <= tolerance

    def row_index(self, y: int) -> int:
        """当前画面里行心纵坐标为 ``y`` 的那一行的绝对行号。"""
        return self.offset + round((y - self.origin) / ROW_PITCH)


def page_tab_states(screen: MatLike) -> list[str]:
    """读分页页签 1–7 的状态：``TAB_SELECTED`` 当前页、``TAB_OPEN`` 可翻到、``TAB_CLOSED`` 暗色或锁定。

    类型筛选后，没有该类物品的页页码变暗；未解锁的页显示锁图标。两者都翻不到内容。
    """
    hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    states: list[str] = []
    for center in PAGE_CENTERS:
        y1, y2 = center.y - _TAB_HALF, center.y + _TAB_HALF
        x1, x2 = center.x - _TAB_HALF, center.x + _TAB_HALF
        h, sat, val = cv2.split(hsv[y1:y2, x1:x2])
        gold = int(((h >= 15) & (h <= 35) & (sat >= 80) & (val >= 120)).sum())
        bright = int((gray[y1:y2, x1:x2] >= 200).sum())
        if gold >= _TAB_GOLD_MIN:
            states.append(TAB_SELECTED)
        elif bright >= _TAB_BRIGHT_MIN:
            states.append(TAB_OPEN)
        else:
            states.append(TAB_CLOSED)
    return states


def title_matches(title: str, name: str) -> bool:
    """浮窗标题与房间名一致：允许 OCR 错漏一个字，不允许多出别的房间名。"""
    compact = ''.join(title.split())
    if not compact:
        return False
    common = str_utils.longest_common_subsequence_length(name, compact)
    return common >= len(name) - 1 and len(compact) <= len(name) + 2
