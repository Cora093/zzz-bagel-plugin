from __future__ import annotations

import time
from typing import TYPE_CHECKING

from one_dragon.base.geometry.point import Point
from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import str_utils
from one_dragon.utils.log_utils import log

from .open_compendium import BagelOpenCompendium
from .operation import BagelOperation

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

SHOP_BACK_POS: Point = Point(84, 50)
"""大富豪商店左上角返回按钮坐标。"""
SHOP_PERMANENT_TAB_POS: Point = Point(196, 302)
"""大富豪商店左侧「常驻商店」选项坐标。"""
SHOP_COMBAT_TAB_POS: Point = Point(1220, 160)
"""大富豪商店上方「战备直售」选项坐标。"""
PLUS_BUTTON_POS: Point = Point(1271, 677)
"""兑换确认弹窗右侧「+」增加数量按钮坐标。"""
CONFIRM_BUTTON_POS: Point = Point(1122, 756)
"""兑换确认弹窗右下角「确认」按钮坐标。"""
CLOSE_DIALOG_POS: Point = Point(1310, 314)
"""兑换确认弹窗右上角关闭「X」按钮坐标。"""
HUB_SHOP_BUTTON_POS: Point = Point(1760, 480)
"""研究站主界面中间偏右「商店」按钮坐标。"""


class BagelBuyKeycard(BagelOperation):
    """自动前往研究站商店购买指定房间的门禁卡。

    具体流程：
    1. 确保停在贝果研究站大厅（若在仓库/备战/选图则退回，若在大世界则通过快捷手册进入）；
    2. 点击右侧「商店」卡片进入大富豪商店；
    3. 左侧选择「常驻商店」，上方选择「战备直售」；
    4. 下滑商品列表，OCR 识别匹配目标房间名称的房卡；
    5. 点击房卡呼出「兑换确认」弹窗；
    6. 点击「+」调整购买数量至指定数量（上限 30 张）；
    7. 点击「确认」完成购买，处理可能的「获得」提示；
    8. 点击左上角返回，回到研究站大厅。
    """

    def __init__(self, ctx: ZContext, room_name: str, count: int = 1) -> None:
        """
        Args:
            ctx: 绝区零上下文。
            room_name: 房间显示名称（与商店内房卡名称对应）。
            count: 购买数量，上限 30 张。
        """
        super().__init__(ctx, op_name='贝果-购买门禁卡', timeout_seconds=120)
        self.room_name: str = room_name
        self.count: int = max(1, min(count, 30))
        self._scroll_count: int = 0
        self._dialog_checks: int = 0
        self._finish_checks: int = 0
        self._open_shop_clicks: int = 0

    def handle_init(self) -> None:
        """每次运行重置计数与状态。"""
        super().handle_init()
        self._scroll_count = 0
        self._dialog_checks = 0
        self._finish_checks = 0
        self._open_shop_clicks = 0

    def _at_research_station(self) -> bool:
        """核对是否停在研究站大厅（非选图页，具备标题和前往空洞）。"""
        return all(
            self.round_by_find_area(self.last_screenshot, '贝果-研究站', name).is_success
            for name in ('标题', '前往空洞')
        )

    def _in_shop(self) -> bool:
        """核对是否已在大富豪商店。"""
        if self.round_by_find_area(self.last_screenshot, '贝果-商店', '标题').is_success:
            return True
        ocr_res = self.ctx.ocr.run_ocr(self.last_screenshot)
        return any('大富豪商店' in k or '常驻商店' in k for k in ocr_res)

    @operation_node(name='准备前往商店', is_start_node=True, timeout_seconds=30)
    def prepare_navigation(self) -> OperationRoundResult:
        """从当前画面返回或进入贝果研究站大厅。"""
        if self._in_shop():
            return self.round_success('已在商店')
        if self._at_research_station():
            return self.round_success('已在研究站')

        # 若在选图、备战、仓库等子页面，点击左上返回
        if (
            self.round_by_find_area(self.last_screenshot, '贝果-选图', '前往备战').is_success
            or self.round_by_find_area(self.last_screenshot, '贝果-备战', '预设组合').is_success
            or self.round_by_find_area(self.last_screenshot, '贝果-仓库', '放入仓库').is_success
        ):
            self.round_by_click_area('菜单', '返回')
            return self.round_wait('从子页面返回研究站', wait=0.8)

        # 若在大世界，通过快捷手册进入
        if self.round_by_find_area(self.last_screenshot, '大世界-普通', '快捷手册').is_success:
            log.info('贝果购买房卡：在大世界画面，打开快捷手册进入研究站')
            res = BagelOpenCompendium(self.ctx).execute()
            self.screenshot()
            if not res.success:
                return self.round_fail('大世界进入贝果研究站失败')
            return self.round_wait('等待研究站加载', wait=1.0)

        # 未知弹窗兜底：点击返回或 ESC
        self.round_by_click_area('菜单', '返回')
        return self.round_wait('尝试返回研究站', wait=0.6)

    @node_from(from_name='准备前往商店', status='已在研究站')
    @operation_node(name='打开商店', timeout_seconds=15)
    def open_shop(self) -> OperationRoundResult:
        """在研究站大厅点击中间右侧的商店。"""
        if self._in_shop():
            return self.round_success('已打开商店')

        # 尝试通过区域查找并点击
        clicked = self.round_by_find_and_click_area(
            self.last_screenshot, '贝果-研究站', '商店', success_wait=1.0, retry_wait=0.5,
        ).is_success
        if not clicked:
            # 坐标兜底点击
            self.ctx.controller.click(HUB_SHOP_BUTTON_POS)
            log.info('点击研究站商店按钮坐标 %s', HUB_SHOP_BUTTON_POS)

        self._open_shop_clicks += 1
        return self.round_wait('等待打开商店', wait=1.0)

    @node_from(from_name='打开商店', status='已打开商店')
    @node_from(from_name='准备前往商店', status='已在商店')
    @operation_node(name='切换常驻战备', timeout_seconds=15)
    def switch_shop_tabs(self) -> OperationRoundResult:
        """确保左侧选择常驻商店，上方选择战备直售。"""
        # 点击左侧常驻商店
        self.ctx.controller.click(SHOP_PERMANENT_TAB_POS)
        time.sleep(0.3)
        # 点击上方战备直售
        self.ctx.controller.click(SHOP_COMBAT_TAB_POS)
        time.sleep(0.4)
        return self.round_success('已切换常驻战备')

    def _match_card_rect(self, screen) -> Rect | None:
        """在当前画面中匹配对应房间的房卡，返回点击区域。"""
        target = self.room_name.replace(' ', '').replace('·', '')
        results = self.ctx.ocr.run_ocr(screen)
        best_rect: Rect | None = None
        best_score: float = 0.0

        for text, mrls in results.items():
            clean_text = text.replace(' ', '').replace('·', '')
            if not clean_text:
                continue
            common_len = str_utils.longest_common_subsequence_length(clean_text, target)
            score = common_len / max(len(target), len(clean_text))

            if score >= 0.7 and score > best_score:
                for item in mrls:
                    rect: Rect | None = None
                    if hasattr(item, 'rect') and item.rect is not None:
                        rect = item.rect
                    elif isinstance(item, tuple) and len(item) >= 5:
                        rect = Rect(item[1], item[2], item[1] + item[3], item[2] + item[4])
                    if rect is not None and 400 <= rect.center.x <= 1850 and 240 <= rect.center.y <= 950:
                        best_score = score
                        best_rect = rect

        return best_rect

    @node_from(from_name='切换常驻战备')
    @operation_node(name='查找并选择房卡', timeout_seconds=30)
    def find_and_select_card(self) -> OperationRoundResult:
        """在商品列表中查找对应房卡；未显示则下滑列表。"""
        rect = self._match_card_rect(self.last_screenshot)
        if rect is not None:
            click_point = Point(rect.center.x, rect.center.y + 40)
            log.info('在商店中找到房卡「%s」，文字区域 %s，点击位置 %s', self.room_name, rect, click_point)
            self.ctx.controller.click(click_point)
            return self.round_success('已点击房卡', wait=0.8)

        if self._scroll_count < 4:
            self._scroll_count += 1
            log.info('当前未见房卡「%s」，向下滑动商品列表 500 像素 (第 %s 次)', self.room_name, self._scroll_count)
            # 以 300 像素/s 的速度向上拖拽 500 像素使列表下滑
            self.ctx.controller.drag_to(
                start=Point(1100, 750),
                end=Point(1100, 250),
                duration=500.0 / 300.0,
            )
            return self.round_wait('向下滑动商品列表', wait=0.5)

        return self.round_fail(f'在商店常驻战备中未找到房卡「{self.room_name}」')

    @node_from(from_name='查找并选择房卡', status='已点击房卡')
    @operation_node(name='兑换确认弹窗', timeout_seconds=20)
    def confirm_exchange(self) -> OperationRoundResult:
        """在兑换确认弹窗中设置数量并点击确认。"""
        ocr_res = self.ctx.ocr.run_ocr(self.last_screenshot)
        dialog_open = (
            any('兑换确认' in k for k in ocr_res)
            or self.round_by_find_area(self.last_screenshot, '贝果-商店', '兑换确认').is_success
        )
        if not dialog_open:
            self._dialog_checks += 1
            if self._dialog_checks > 5:
                return self.round_fail('未能呼出房卡兑换确认弹窗')
            return self.round_wait('等待兑换确认弹窗', wait=0.5)

        log.info('已呼出「%s」兑换确认弹窗，准备购买 %s 张', self.room_name, self.count)
        # 默认数量为 1，若需要购买更多张，点击「+」按钮 (self.count - 1) 次
        clicks_needed = min(self.count, 30) - 1
        if clicks_needed > 0:
            log.info('连续点击「+」按钮 %s 次以调整购买数量', clicks_needed)
            for _ in range(clicks_needed):
                self.ctx.controller.click(PLUS_BUTTON_POS)
                time.sleep(0.08)
            time.sleep(0.3)

        # 点击确认购买
        self.ctx.controller.click(CONFIRM_BUTTON_POS)
        log.info('已点击兑换确认按钮')
        return self.round_success('已确认兑换', wait=1.0)

    @node_from(from_name='兑换确认弹窗', status='已确认兑换')
    @operation_node(name='处理购买结果并返回', timeout_seconds=25)
    def finish_and_return(self) -> OperationRoundResult:
        """处理购买后的提示弹窗并返回研究站大厅。"""
        ocr_res = self.ctx.ocr.run_ocr(self.last_screenshot)

        # 处理可能出现的「获得」提示弹窗
        if any('获得' in k or '点击空白处关闭' in k for k in ocr_res):
            log.info('关闭房卡获得提示弹窗')
            self.ctx.controller.click(Point(960, 950))
            return self.round_wait('等待获得弹窗关闭', wait=0.6)

        # 检查兑换确认弹窗是否仍然残留（未成功兑换，如硬币不足）
        if any('兑换确认' in k for k in ocr_res):
            self._finish_checks += 1
            if self._finish_checks > 3:
                log.error('兑换确认弹窗未能关闭，可能齿轮硬币不足或达到上限')
                self.ctx.controller.click(CLOSE_DIALOG_POS)
                return self.round_fail('房卡兑换未完成，可能齿轮硬币不足')
            return self.round_wait('等待兑换确认弹窗关闭', wait=0.6)

        # 已在商店，点击返回返回大厅
        if self._in_shop():
            self.ctx.controller.click(SHOP_BACK_POS)
            return self.round_wait('点击返回研究站大厅', wait=1.0)

        # 检查是否已回到研究站
        if self._at_research_station():
            log.info('已成功购买「%s」房卡 %s 张并返回研究站', self.room_name, self.count)
            return self.round_success(f'已成功购买 {self.room_name} 房卡 {self.count} 张')

        # 兜底返回
        self.round_by_click_area('菜单', '返回')
        return self.round_wait('等待返回研究站', wait=0.8)
