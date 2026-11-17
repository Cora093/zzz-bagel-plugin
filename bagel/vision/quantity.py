"""读物品格下方的数量文字。

搜索结果、安全箱和背包的卡片大小相近，数量文字的中线都在格心下方约 39 像素，
三处共用一套偏移。数字区只有一行，所以用单行识别（不经文字检测）：检测模型
框不住单个窄的「1」，按区域 OCR 时「1」几乎总是读成空字符串。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from one_dragon.utils.log_utils import log

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from one_dragon.base.geometry.point import Point
    from zzz_od.context.zzz_context import ZContext

QUANTITY_TOP_OFFSET: int = 24
"""数量文字区上沿相对格心的纵向偏移。"""
QUANTITY_BOTTOM_OFFSET: int = 54
"""数量文字区下沿相对格心的纵向偏移。"""
QUANTITY_HALF_WIDTH: int = 45
"""数量文字区半宽，与卡片同宽，不越过相邻格。"""

_CHINESE_DIGITS: dict[str, str] = {
    '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
}
"""单行识别偶尔把数字读成中文，逐字换回。"""
_LEVEL_PATTERN: re.Pattern[str] = re.compile(r'^lv', re.IGNORECASE)


def parse_quantity(text: str) -> int | None:
    """把数量文字解析成件数；读不出时返回 None，不猜。

    - 纯数字（含逐字的中文数字）：就是件数；
    - 以「Lv」开头：战术棱镜等显示的是等级，不叠放，按 1 件；
    - 其余（空、乱码、光标压字读出的「1~」）：未知。
    """
    compact = text.replace(' ', '')
    if _LEVEL_PATTERN.match(compact):
        return 1
    digits = ''.join(_CHINESE_DIGITS.get(char, char) for char in compact)
    if not digits.isdigit() or int(digits) <= 0:
        return None
    return int(digits)


def read_quantity(ctx: ZContext, screen: MatLike, center: Point) -> int | None:
    """读一格物品的数量；读不出或不在画面内返回 None。

    Args:
        ctx: 运行上下文，用其单行 OCR。
        screen: 整屏 RGB。
        center: 格心，结果格须用已校准的格心。
    """
    x1 = int(center.x) - QUANTITY_HALF_WIDTH
    x2 = int(center.x) + QUANTITY_HALF_WIDTH
    y1 = int(center.y) + QUANTITY_TOP_OFFSET
    y2 = int(center.y) + QUANTITY_BOTTOM_OFFSET
    if x1 < 0 or y1 < 0 or x2 > screen.shape[1] or y2 > screen.shape[0]:
        return None
    try:
        text = ctx.ocr.run_ocr_single_line(screen[y1:y2, x1:x2])
    except Exception:
        log.error('读物品数量失败', exc_info=True)
        return None
    return parse_quantity(str(text))
