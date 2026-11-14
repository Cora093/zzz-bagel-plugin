from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from one_dragon.base.geometry.point import Point
from one_dragon.utils import cv2_utils

from ..defs import PLUGIN_DIR
from .slots import slot_occupied

if TYPE_CHECKING:
    from cv2.typing import MatLike

# 品质从高到低；未对上的底色为未知，最低，有空格时仍可装入。
QUALITY_RANK: dict[str, int] = {'Z': 5, 'S': 4, 'A': 3, 'B': 2, 'C': 1, '?': 0}
ITEM_TYPE_OTHER: str = '其他'
ITEM_TYPE_CURRENCY: str = '金币'
ITEM_TYPE_BLUEPRINT: str = '蓝图'
ITEM_TYPE_MATERIAL: str = '材料'
# 特殊奖励专指使用切面钻石角标的菲林、零号业绩，不代表全部撤离奖励。
ITEM_TYPE_SPECIAL_REWARD: str = '特殊奖励'
QUALITY_UNKNOWN: str = '?'

# 同组、同品质时的类型顺序；两种蓝图开关状态共用。
TYPE_RANK: dict[str, int] = {
    ITEM_TYPE_SPECIAL_REWARD: 7,
    '贵重物品': 6,
    ITEM_TYPE_BLUEPRINT: 5,
    '门禁卡': 4,
    '战术棱镜': 3,
    '装备': 2,
    '战术道具': 1,
    ITEM_TYPE_OTHER: 0,
    ITEM_TYPE_MATERIAL: -1,
    ITEM_TYPE_CURRENCY: -2,
}

ACTION_FILL: str = 'fill'
ACTION_SWAP: str = 'swap'
ACTION_STACK: str = 'stack'
"""同一种物品已经有一格，直接叠到那一格上，不占新的安全箱格子。"""

# 相对已校准格心，40x36 搜索窗完整覆盖 24x24 / 24x26 模板并保留边缘余量。
# 宽度比高度多 4：第 1 列格心 x=1353，金币圆形角标右缘在 x≈1335，窗口右缘原为 1333，
# 角标被截掉 2 像素，得分从 0.90 掉到 0.53，金币堆被读成「其他」而一直待确认。
_BADGE_LEFT: int = 56
_BADGE_TOP: int = 64
_BADGE_SEARCH_W: int = 40
_BADGE_SEARCH_H: int = 36
_BADGE_MATCH_MIN: float = 0.75
# OpenCV HSV：H 0–180。红=Z、金=S、紫=A、蓝=B、绿=C。
_QUALITY_S_MIN: float = 40.0
_QUALITY_V_MIN: float = 40.0
# 品质只读左右底色，避开中央图案、左上角标和下方数量/等级文字。
# 每条 6x48；外侧内缩两像素，减少格心误差导致取到黑边的机会。
_QUALITY_REGIONS: tuple[tuple[int, int, int, int], ...] = (
    (-36, -28, -30, 20),
    (30, -28, 36, 20),
)

_BADGE_FILES: tuple[tuple[str, str], ...] = (
    (ITEM_TYPE_BLUEPRINT, 'blueprint'),
    (ITEM_TYPE_CURRENCY, 'currency'),
    ('战术道具', 'tactic'),
    ('战术棱镜', 'prism_bell'),
    ('战术棱镜', 'prism_char'),
    ('装备', 'equip'),
    ('装备', 'backpack'),
    ('贵重物品', 'valuable'),
    (ITEM_TYPE_SPECIAL_REWARD, 'valuable_reward'),
    ('门禁卡', 'keycard'),
    # 历史文件 other 也是白色箱子材料角标，不能因匹配到不同实拍而改变优先级。
    ('材料', 'other'),
    ('材料', 'material'),
)


@dataclass(frozen=True)
class BagelSlotMark:
    """已占用格子上读到的品质与类型。"""

    index: int
    center: Point
    quality: str
    item_type: str

    @property
    def quality_rank(self) -> int:
        """品质排序值，未知为 0。"""
        return QUALITY_RANK.get(self.quality, 0)

    @property
    def type_rank(self) -> int:
        """类型排序值；未识别按其他。"""
        return TYPE_RANK.get(self.item_type, 0)

    def priority(self, blueprint_priority: bool = True) -> tuple[int, int, int]:
        """先分组，再比品质和类型；开启时蓝图独立分组且只比品质。"""
        if self.item_type == ITEM_TYPE_CURRENCY:
            return -1, -1, self.type_rank
        if blueprint_priority and self.item_type == ITEM_TYPE_BLUEPRINT:
            return 2, self.quality_rank, 0
        if self.item_type == ITEM_TYPE_MATERIAL:
            return 0, self.quality_rank, self.type_rank
        return 1, self.quality_rank, self.type_rank


@dataclass(frozen=True)
class StoreChoice:
    """下一件拖拽：填入空安全箱，或与箱内更差的一格对换。"""

    kind: str
    source_index: int
    dest_index: int
    mark: BagelSlotMark


def is_strictly_better(
    left: BagelSlotMark,
    right: BagelSlotMark,
    blueprint_priority: bool = True,
) -> bool:
    """按同一开关比较两件物品，优先级相等则不替换。"""
    return left.priority(blueprint_priority) > right.priority(blueprint_priority)


def identify_quality(screen: MatLike, center: Point) -> str:
    """用左右底色色带的 HSV 中位数判断 Z/S/A/B/C；对不上则为未知。

    Args:
        screen: 整屏 RGB。
        center: 格子中心。
    """
    regions: list[MatLike] = []
    for left, top, right, bottom in _QUALITY_REGIONS:
        x1, y1 = int(center.x) + left, int(center.y) + top
        x2, y2 = int(center.x) + right, int(center.y) + bottom
        if x1 < 0 or y1 < 0 or x2 > screen.shape[1] or y2 > screen.shape[0]:
            return QUALITY_UNKNOWN
        regions.append(screen[y1:y2, x1:x2])
    hsv = cv2.cvtColor(np.concatenate(regions, axis=1), cv2.COLOR_RGB2HSV)
    hue, sat, value = (float(value) for value in np.median(hsv.reshape(-1, 3), axis=0))
    if sat < _QUALITY_S_MIN or value < _QUALITY_V_MIN:
        return QUALITY_UNKNOWN
    if hue <= 12 or hue >= 165:
        return 'Z'
    if 16 <= hue <= 42:
        return 'S'
    if 120 <= hue <= 160:
        return 'A'
    if 90 <= hue <= 119:
        return 'B'
    if 50 <= hue <= 85:
        return 'C'
    return QUALITY_UNKNOWN


def identify_type(screen: MatLike, center: Point) -> str:
    """在格心左上搜索角标模板；低于阈值视为其他。

    Args:
        screen: 整屏 RGB。
        center: 格子中心。
    """
    crop = _badge_search_crop(screen, center)
    if crop is None:
        return ITEM_TYPE_OTHER
    best_type = ITEM_TYPE_OTHER
    best_score = -1.0
    for item_type, template in _badge_templates():
        if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
            continue
        score = float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED).max())
        if score > best_score:
            best_score = score
            best_type = item_type
    if best_score < _BADGE_MATCH_MIN:
        return ITEM_TYPE_OTHER
    return best_type


def inspect_occupied(
    screen: MatLike,
    centers: tuple[Point, ...],
    indices: Sequence[int] | None = None,
) -> list[BagelSlotMark]:
    """只识别已占用格子的品质和类型。

    Args:
        screen: 整屏 RGB。
        centers: 结果格、安全箱或背包格心。
        indices: 只识别这些下标；None 表示按灰度方差自行判断哪些格子有物品。
    """
    marks: list[BagelSlotMark] = []
    for index, center in enumerate(centers):
        if indices is not None:
            if index not in indices:
                continue
        elif not slot_occupied(screen, center):
            continue
        marks.append(
            BagelSlotMark(
                index=index,
                center=center,
                quality=identify_quality(screen, center),
                item_type=identify_type(screen, center),
            ),
        )
    return marks


def choose_store_action(
    results: Sequence[BagelSlotMark],
    safes: Sequence[BagelSlotMark],
    safe_empty: Sequence[int],
    blueprint_priority: bool = True,
    stack_target: Callable[[BagelSlotMark], int | None] | None = None,
) -> StoreChoice | None:
    """用同一优先级选择非金币候选和箱内目标；满箱时仅严格更优才对换。

    传了 ``stack_target`` 时，先按同一优先级逐个找能叠放的格子；叠放不占新
    格子，优先于填空和对换。找不到才走原来的填空或对换。
    """
    candidates = [mark for mark in results if mark.item_type != ITEM_TYPE_CURRENCY]
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda mark: (*mark.priority(blueprint_priority), -mark.index),
        reverse=True,
    )
    if stack_target is not None:
        for mark in ordered:
            destination = stack_target(mark)
            if destination is not None:
                return StoreChoice(ACTION_STACK, mark.index, destination, mark)
    candidate = ordered[0]
    if safe_empty:
        return StoreChoice(ACTION_FILL, candidate.index, safe_empty[0], candidate)
    if safes:
        worst = min(
            safes,
            key=lambda mark: (*mark.priority(blueprint_priority), mark.index),
        )
        if is_strictly_better(candidate, worst, blueprint_priority):
            return StoreChoice(ACTION_SWAP, candidate.index, worst.index, candidate)
    return None


def _badge_search_crop(screen: MatLike, center: Point) -> MatLike | None:
    """裁出格心左上的角标搜索窗。"""
    x1 = int(center.x) - _BADGE_LEFT
    y1 = int(center.y) - _BADGE_TOP
    x2 = x1 + _BADGE_SEARCH_W
    y2 = y1 + _BADGE_SEARCH_H
    if x1 < 0 or y1 < 0 or x2 > screen.shape[1] or y2 > screen.shape[0]:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(screen.shape[1], x2)
        y2 = min(screen.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return screen[y1:y2, x1:x2]


@lru_cache(maxsize=1)
def _badge_templates() -> tuple[tuple[str, MatLike], ...]:
    """从资源目录加载角标模板；缺文件的类型跳过。"""
    loaded: list[tuple[str, MatLike]] = []
    directory = _badge_dir()
    for item_type, stem in _BADGE_FILES:
        image = cv2_utils.read_image(str(directory / f'{stem}.png'))
        if image is None:
            continue
        loaded.append((item_type, image))
    return tuple(loaded)


def _badge_dir() -> Path:
    """角标模板目录；模板随插件分发，不再回退工作目录里的旧副本。"""
    return PLUGIN_DIR / 'assets' / 'game_data' / 'bagel' / 'item_badges'


# 叠放判断只比格心附近的图标，避开左上角标（约 -56,-64）和底部数量文字（约 +31）。
# 结果格外框 91x109、安全箱外框 97x116，图标在安全箱里约大 6%，比对时按几档比例缩放。
STACK_ICON_HALF: int = 26
STACK_WINDOW_HALF: int = 34
_STACK_SCALES: tuple[float, ...] = (0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02)
STACK_MATCH_MIN: float = 0.9
"""同一种物品的最低相关度；实测同一件 0.95–0.98，外形相近但颜色不同的物品约 0.72。"""
STACK_VERIFY_HALF: int = 22
"""叠放核对的裁剪半边长，格心 44 像素见方，避开左上角标和底部数量文字。"""
_STACK_SAME_MAX_DIFF: float = 10.0
"""同一格拖拽前后图标的最大平均差；同一件物品接近 0，换成别的物品会远大于它。"""


def icon_patch(screen: MatLike, center: Point, half: int = STACK_ICON_HALF) -> MatLike | None:
    """裁出格心周围的图标区；越界返回 None。

    Args:
        screen: 整屏 RGB。
        center: 格心。
        half: 裁剪半边长；叠放比对用 STACK_ICON_HALF，安全箱搜索窗用 STACK_WINDOW_HALF。
    """
    x1, y1 = int(center.x) - half, int(center.y) - half
    x2, y2 = int(center.x) + half, int(center.y) + half
    if x1 < 0 or y1 < 0 or x2 > screen.shape[1] or y2 > screen.shape[0]:
        return None
    return screen[y1:y2, x1:x2].copy()


def icon_match_score(item_icon: MatLike, safe_window: MatLike) -> float:
    """结果格图标在安全箱格子里的最高相关度，取三个颜色通道里最低的一个。

    两种格子的外框尺寸不同，图标在安全箱里约大 6%；按几档比例缩放后取最高分，
    避免固定比例对不上。只看灰度会让颜色不同、外形相近的两件物品也拿到高分
    （实测外形相近的黄针剂和银罐子灰度 0.92），所以每个通道单独比、取最差通道；
    同一种物品三个通道都在 0.95 以上，看似相近的不同物品会有一两个通道明显偏低。

    Args:
        item_icon: 结果格图标裁剪。
        safe_window: 安全箱格子的图标搜索窗。
    """
    if item_icon is None or safe_window is None or item_icon.size == 0 or safe_window.size == 0:
        return -1.0
    item = item_icon.astype(np.float32)
    safe = safe_window.astype(np.float32)
    best: list[float] = [-1.0, -1.0, -1.0]
    for scale in _STACK_SCALES:
        size = (
            max(item.shape[1], round(safe.shape[1] * scale)),
            max(item.shape[0], round(safe.shape[0] * scale)),
        )
        scaled = cv2.resize(safe, size, interpolation=cv2.INTER_AREA)
        for channel in range(3):
            score = float(cv2.matchTemplate(
                scaled[:, :, channel], item[:, :, channel], cv2.TM_CCOEFF_NORMED,
            ).max())
            if score > best[channel]:
                best[channel] = score
    return min(best)


def icons_same(before: MatLike, after: MatLike) -> bool:
    """同一格拖拽前后的图标是否没变。

    只用来区分「叠放」和「对换」：叠放后图标还是同一件，数量文字在下方、
    不在裁剪范围内；对换后图标会换成另一件物品。
    """
    if before is None or after is None or before.shape != after.shape:
        return False
    return float(np.mean(np.abs(before.astype(np.float32) - after.astype(np.float32)))) <= _STACK_SAME_MAX_DIFF


def find_stack_target(
    screen: MatLike,
    mark: BagelSlotMark,
    safes: Sequence[BagelSlotMark],
    blocked: set[tuple[int, int]],
) -> int | None:
    """在安全箱里找能叠放 ``mark`` 的格子；找不到返回 None。

    只在材料里找，类型和品质都要相同，再用图标相关度确认，必须过 STACK_MATCH_MIN：
    不同物品拖到一起会变成对换，宁可占一个空格也不赌。搜查结果格和局内背包格
    都用这一个函数：两处的图标和安全箱里的比例都落在 _STACK_SCALES 覆盖的范围内。

    Args:
        screen: 整屏 RGB，源格和安全箱都在这张画面里。
        mark: 待拖的源格。
        safes: 安全箱里已占用的格子。
        blocked: 试过不能叠放的（源格, 安全箱格）组合，跳过不再试。
    """
    if mark.item_type != ITEM_TYPE_MATERIAL:
        return None
    icon = icon_patch(screen, mark.center)
    if icon is None:
        return None
    best_index: int | None = None
    best_score: float = STACK_MATCH_MIN
    for safe_mark in safes:
        if safe_mark.item_type != mark.item_type or safe_mark.quality != mark.quality:
            continue
        if (mark.index, safe_mark.index) in blocked:
            continue
        window = icon_patch(screen, safe_mark.center, STACK_WINDOW_HALF)
        score = icon_match_score(icon, window)
        if score >= best_score:
            best_score = score
            best_index = safe_mark.index
    return best_index


def stack_icon_unchanged(before: MatLike, after: MatLike, center: Point) -> bool:
    """叠放核对：安全箱目标格拖拽前后图标没变。

    只比格心 STACK_VERIFY_HALF 见方：再大就会带到底部数量文字，叠放后数量变化
    会被误判成图标变了，进而被当成对换。
    """
    before_icon = icon_patch(before, center, STACK_VERIFY_HALF)
    after_icon = icon_patch(after, center, STACK_VERIFY_HALF)
    return before_icon is not None and after_icon is not None and icons_same(before_icon, after_icon)

