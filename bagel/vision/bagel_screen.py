from __future__ import annotations

import re
from typing import TYPE_CHECKING

import cv2
import numpy as np
from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.matcher.ocr import ocr_utils
from one_dragon.base.screen import screen_utils
from one_dragon.utils import cv2_utils, str_utils

from ..bagel_defs import RECOMMENDED_VALUE

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from zzz_od.context.zzz_context import ZContext

# 详情操作文字只出现在备战右侧，不读左侧仓储候选列表。
_PREPARE_RIGHT_RECT: Rect = Rect(1000, 0, 1920, 1080)


def read_area(ctx: ZContext, screen: MatLike, screen_name: str, area_name: str) -> str:
    """读取已建档区域的完整文字，不把空识别结果替换为零。"""
    area = ctx.screen_loader.get_area(screen_name, area_name)
    if area is None:
        raise ValueError(f'缺少贝果画面区域：{screen_name}/{area_name}')
    matches = ctx.ocr_service.get_ocr_result_list(
        screen, rect=area.pc_rect, crop_first=True, color_range=area.color_range,
    )
    return ''.join(match.data for match in sorted(matches, key=lambda match: (match.y, match.x)))


def mask_coin_left(crop: MatLike) -> MatLike | None:
    """定位唯一金币并遮挡其左侧区域，保持金额完整和 OCR 输入尺寸。"""
    if crop is None or crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([10, 100, 120]), np.array([40, 255, 255]))
    _, _, stats, _ = cv2.connectedComponentsWithStats(yellow)
    candidates: list[tuple[int, int]] = []
    for x, _y, width, height, area in stats[1:]:
        if area >= 100 and 18 <= width <= 45 and 18 <= height <= 45:
            candidates.append((int(x), int(x + width + 1)))
    if len(candidates) != 1:
        return None
    left, right = candidates[0]
    # 金币必须位于全部白色金额左侧；定位不明时不能遮掉高位后误读成零。
    if np.any(np.all(crop[:, :max(0, left - 2)] >= 180, axis=2)):
        return None
    masked = crop.copy()
    masked[:, :right] = 0
    return masked


def read_coin_value(ctx: ZContext, screen: MatLike, screen_name: str, area_name: str) -> str:
    """先排除金币再读完整金额；定位失败保留为未知，不返回零。

    金币紧贴数字，OCR 会把图标边缘读成多余的字（如「雪258000」）；
    游戏字体粗细设置会让标题和数字左右挪动几像素，固定裁边兼容不了两种设置。
    """
    area = ctx.screen_loader.get_area(screen_name, area_name)
    if area is None:
        raise ValueError(f'缺少贝果画面区域：{screen_name}/{area_name}')
    crop = cv2_utils.crop_image_only(screen, area.pc_rect)
    masked = mask_coin_left(crop)
    if masked is None:
        return ''
    matches = ctx.ocr_service.get_ocr_result_list(masked, color_range=area.color_range)
    return ''.join(match.data for match in sorted(matches, key=lambda match: (match.y, match.x)))


def warehouse_sale_state(ctx: ZContext, screen: MatLike) -> str | None:
    """识别仓库出售底栏和出售相关弹窗，供启动清空拒绝继续操作。"""
    for name in ('取消出售', '快速选择标题', '出售确认标题'):
        if screen_utils.find_area(ctx, screen, '贝果-仓库', name) == screen_utils.FindAreaResultEnum.TRUE:
            return name
    if all(screen_utils.find_area(ctx, screen, '贝果-仓库', name) == screen_utils.FindAreaResultEnum.TRUE
           for name in ('出售获得标题', '出售获得货币')):
        return '出售获得货币'
    return None


def read_containers(ctx: ZContext, screen: MatLike) -> dict[str, str]:
    """读取备战背包和安全箱的容量文字。"""
    return {area: read_area(ctx, screen, '贝果-备战', area) for area in ('背包数量', '安全箱数量')}


def prepare_detail_state(ctx: ZContext, screen: MatLike) -> bool:
    """备战右侧出现详情操作文字即视为有遮挡，否则视为无遮挡。"""
    matches = ctx.ocr_service.get_ocr_result_list(screen, rect=_PREPARE_RIGHT_RECT, crop_first=True)
    text = ''.join(re.sub(r'\s', '', match.data) for match in matches)
    return any(label in text for label in ('放入背包', '放入安全箱', '放入仓储枢纽'))


def empty_capacity(text: str) -> bool:
    """仅接受已识别的零占用和正容量，不接受空字符串或锁定容量。"""
    return re.fullmatch(r'0/[1-9][0-9]*', text.replace(' ', '')) is not None


def parse_capacity_pair(text: str) -> tuple[int, int] | None:
    """从“全部(235/280)”一类文字读取占用与容量；识别不清返回 None。"""
    match = re.search(r'(\d+)\s*/\s*(\d+)', text.replace(' ', ''))
    if match is None:
        return None
    used = int(match.group(1))
    total = int(match.group(2))
    if total <= 0 or used > total:
        return None
    return used, total


def parse_filter_count(text: str) -> int | None:
    """读取快速选择「符合以上条件的道具数量」。"""
    compact = text.replace(' ', '')
    match = re.search(r'数量[:：]?(\d+)', compact)
    if match is not None:
        return int(match.group(1))
    match = re.search(r'(\d+)$', compact)
    if match is not None and '符合' in compact:
        return int(match.group(1))
    return None


def zero_containers(values: dict[str, str]) -> bool:
    """背包和安全箱都明确为零占用才算空。"""
    return all(empty_capacity(values.get(name, '')) for name in ('背包数量', '安全箱数量'))


def complete_containers(values: dict[str, str]) -> bool:
    """两项容量都合法才允许判断；识别缺失或非法容量不能算非零。"""
    for name in ('背包数量', '安全箱数量'):
        text = values.get(name, '').replace(' ', '')
        if re.fullmatch(r'[0-9]+/[1-9][0-9]*', text) is None:
            return False
        if parse_capacity_pair(text) is None:
            return False
    return True


def read_entry_warning_text(ctx: ZContext, screen: MatLike) -> str:
    """按行读取入场提示，避免同一行识别框的高度偏差颠倒文字顺序。"""
    area = ctx.screen_loader.get_area('贝果-入场确认', '提示')
    if area is None:
        raise ValueError('缺少贝果画面区域：贝果-入场确认/提示')
    matches = ctx.ocr_service.get_ocr_result_list(
        screen, rect=area.pc_rect, crop_first=True, color_range=area.color_range,
    )
    # 先按行的中心高度排列，再复用合行规则从左到右拼接；不丢弃任何提示文字。
    ordered = sorted(matches, key=lambda match: (match.center.y, match.x))
    ocr_map = ctx.ocr_service.convert_list_to_map(ordered)
    return ocr_utils.merge_ocr_result_to_single_line(ocr_map, join_space=False)


#: 已知入场提示的识别片段；一条提示的全部片段都命中才算识别成功。
_ENTRY_WARNING_FRAGMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('代理人队伍未满编', ('队伍未满编',)),
    ('零装备价值', ('当前装备价值为0', '未达到推荐价值')),
    ('未装备武备', ('未装备武备',)),
    ('未穿戴队伍装备', ('未穿戴队伍装备',)),
)


def _has_fragment(compact: str, fragment: str) -> bool:
    """片段的字按顺序出现在提示中即命中；五个字及以上的片段允许 OCR 错漏一个字。"""
    allowed_misses = 1 if len(fragment) >= 5 else 0
    return str_utils.longest_common_subsequence_length(fragment, compact) >= len(fragment) - allowed_misses


def entry_warning(text: str) -> str | None:
    """区分已知入场弹窗；其他确认框不能自动确认。

    按关键片段识别，容忍 OCR 漏读、错读个别字或多读其他文字。
    零装备价值提示中的推荐价值与高危明显不同时（如其他难度的 72100），按未知提示处理。
    """
    compact = re.sub(r'[\s，,。？?：:（）()、]', '', text)
    for name, fragments in _ENTRY_WARNING_FRAGMENTS:
        if not all(_has_fragment(compact, fragment) for fragment in fragments):
            continue
        if name == '零装备价值':
            # 只判断四位及以上的数字串，避免把装备价值 0 或断开的残缺数字当成推荐价值。
            value = max(re.findall(r'\d+', compact), key=len, default='')
            if (len(value) >= 4 and str_utils.longest_common_subsequence_length(value, RECOMMENDED_VALUE)
                    < len(RECOMMENDED_VALUE) - 1):
                return None
        return name
    return None
