from __future__ import annotations

from typing import TYPE_CHECKING

from one_dragon.base.geometry.point import Point
from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils import str_utils
from one_dragon.utils.i18_utils import gt
from zzz_od.operation.compendium.compendium_choose_category import (
    CompendiumChooseCategory,
)

from .operation import BagelOperation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from one_dragon.base.matcher.ocr.ocr_match_result import OcrMatchResult
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

COMPENDIUM_TABS: tuple[str, ...] = ('目标', '日常', '训练', '作战', '战术')
"""快捷手册顶部五个页签。"""

COMBAT_TAB_RECT: Rect = Rect(1265, 130, 1455, 225)
"""1080p 下「作战」页签所在槽位；「战术」在其右侧 1459 起。"""

COMBAT_CATEGORIES: tuple[str, ...] = ('周期征讨', '零号空洞', '式舆防卫战', '危局强袭战', '临界推演')
"""只在「作战」页签左栏出现的分类名，任一出现即视为已在作战页。

不同版本左栏首项不同（旧版为「零号空洞」，现版为「周期征讨」），所以列出多个。
「拟真鏖战试炼」常被 OCR 读成「拟真鹰战试炼」，不列入。
"""

OPEN_COMBAT_SECONDS: float = 25
"""打开作战页的等待上限，略短于节点超时，以便报告具体卡在哪一步。"""

OPEN_CLICK_LIMIT: int = 3
"""大世界手册按钮与「作战」页签各自的最多点击次数。"""


def count_compendium_tabs(results: Sequence[OcrMatchResult]) -> int:
    """统计无颜色过滤 OCR 读到的不同页签数量。"""
    names = {gt(name, 'game') for name in COMPENDIUM_TABS}
    return len({text for text in (str_utils.remove_whitespace(item.data) for item in results)
                if text in names})


def find_combat_tab(results: Sequence[OcrMatchResult]) -> Point | None:
    """返回「作战」页签文字中心；必须整串相等且落在作战槽位。

    主仓页签匹配用 LCS 0.5，「作战」与「战术」共有「战」字会互相命中，这里不用。
    """
    target = gt('作战', 'game')
    rect = COMBAT_TAB_RECT
    for item in results:
        center = item.center
        if (str_utils.remove_whitespace(item.data) == target
                and rect.x1 <= center.x <= rect.x2 and rect.y1 <= center.y <= rect.y2):
            return center
    return None


def is_combat_category_list(results: Sequence[OcrMatchResult]) -> bool:
    """左栏任一行包含作战页专属分类名时返回真。"""
    names = [gt(name, 'game') for name in COMBAT_CATEGORIES]
    return any(name in str_utils.remove_whitespace(item.data)
               for item in results for name in names)


class BagelOpenCompendium(BagelOperation):
    """从普通大世界打开快捷手册并进入贝果主界面。调用者负责准备入场前位置。

    手册页签的主仓画面定义按选中页签的颜色识别；部分环境选中页签更亮，过滤后读不出字
    （issue #30）。本操作只用无颜色过滤的 OCR 和页面内容判断位置，不调用画面路由。
    """

    def __init__(self, ctx: ZContext) -> None:
        """复用快捷手册列表区域，不执行副本传送确认。"""
        super().__init__(ctx, op_name='贝果-快捷手册入口', timeout_seconds=90)
        self.scroll_count: int = 0
        self.open_clicks: int = 0
        self.tab_clicks: int = 0

    def handle_init(self) -> None:
        """每次执行重新计算卡片查找与点击次数。"""
        super().handle_init()
        self.scroll_count = 0
        self.open_clicks = 0
        self.tab_clicks = 0

    def _ocr_area(self, area_name: str) -> list[OcrMatchResult]:
        """对快捷手册指定区域做无颜色过滤 OCR；区域缺失时返回空列表。"""
        area = self.ctx.screen_loader.get_area('快捷手册', area_name)
        if area is None:
            return []
        return self.ctx.ocr_service.get_ocr_result_list(image=self.last_screenshot, rect=area.rect)

    def _in_compendium(self) -> bool:
        """顶部读到至少两个页签名即视为手册已打开，与主仓返回大世界的判断口径一致。"""
        return count_compendium_tabs(self._ocr_area('TAB列表')) >= 2

    def _at_combat_tab(self) -> bool:
        """左栏出现作战页专属分类即视为已在作战页，不依赖选中页签颜色。"""
        return is_combat_category_list(self._ocr_area('分类列表'))

    @operation_node(name='打开快捷手册作战页', is_start_node=True, timeout_seconds=30)
    def open_combat(self) -> OperationRoundResult:
        """点击大世界手册按钮，再点「作战」页签，以左栏分类确认到达。

        过渡状态一律等待，由时间上限结束，不消耗框架的重试次数。
        """
        if self._at_combat_tab():
            return self.round_success('已在快捷手册作战页')
        if self._in_compendium():
            point = find_combat_tab(self._ocr_area('TAB列表'))
            if point is None:
                status = '未找到作战页签'
            elif self.tab_clicks >= OPEN_CLICK_LIMIT:
                status = '点击作战页签后未见作战页分类'
            elif self.ctx.controller.click(point):
                self.tab_clicks += 1
                return self.round_wait('已点击作战页签', wait=1)
            else:
                status = '点击作战页签失败'
        elif self._has('快捷手册', '大世界-普通'):
            if self.open_clicks >= OPEN_CLICK_LIMIT:
                status = '点击快捷手册按钮后手册未打开'
            else:
                result = self.round_by_click_area('大世界-普通', '快捷手册')
                if result.is_success:
                    self.open_clicks += 1
                    # 手册展开有动画，间隔过短会在动画中再次点到大世界按钮。
                    return self.round_wait('已点击快捷手册按钮', wait=2)
                status = result.status or '点击快捷手册按钮失败'
        else:
            status = '未识别到快捷手册或大世界手册按钮'
        if self.operation_usage_time >= OPEN_COMBAT_SECONDS:
            return self.round_fail(f'{status}，停止并保留现场')
        return self.round_wait(status, wait=0.5)

    @node_from(from_name='打开快捷手册作战页')
    @operation_node(name='选择周期征讨', screenshot_before_round=False)
    def choose_category(self) -> OperationRoundResult:
        """复用快捷手册分类选择，下一节点重新读取卡片内容。"""
        operation = CompendiumChooseCategory(self.ctx, '周期征讨')
        operation.timeout_seconds = min(15, max(0, self.timeout_seconds - self.operation_usage_time))
        return self.round_by_op_result(operation.execute())

    @node_from(from_name='选择周期征讨')
    @operation_node(name='打开迷宫诡域', timeout_seconds=30)
    def open_bagel(self) -> OperationRoundResult:
        """标题与前往按钮必须属于同一卡片；点击后只等完整贝果入口。"""
        if self._at_bagel_hub():
            return self.round_success('已到贝果入口')
        if self.node_clicked:
            # 点卡片「前往」后可能先停在活动页（标题为「迷宫诡域：古怪试炼」并带赛季内容）。
            # 该页同样有「前往空洞」，点它进入贝果选图页；选图页没有主界面标题，
            # 所以这里只等「前往备战」出现，不能等主界面标题。
            return self.round_by_find_and_click_area(
                self.last_screenshot, '贝果-研究站', '前往空洞',
                until_find_all=[('贝果-选图', '前往备战')],
                success_wait=1, retry_wait=1,
            )
        # 未确认停在作战页时只等待，避免页面切换中途读不到卡片就消耗滚动次数。
        if not self._at_combat_tab():
            return self.round_wait('等待快捷手册作战页', wait=0.5)
        titles_area = self.ctx.screen_loader.get_area('快捷手册', '副本列表')
        buttons_area = self.ctx.screen_loader.get_area('快捷手册', '前往列表')
        if titles_area is None or buttons_area is None:
            return self.round_fail('缺少快捷手册副本或前往列表区域')
        titles = self.ctx.ocr_service.get_ocr_result_list(
            image=self.last_screenshot, rect=titles_area.rect,
        )
        # 卡片标题在活动期间带副标题（如「迷宫诡域：古怪试炼」），只用前缀匹配，
        # 避免整串相等时找不到卡片而一直滚动。
        bagel_title = gt('迷宫诡域', 'game')
        target = next((item for item in titles
                       if str_utils.remove_whitespace(item.data).startswith(bagel_title)), None)
        if target is None:
            if self.scroll_count >= 5:
                return self.round_fail('快捷手册未找到迷宫诡域，停止并保留现场')
            self.scroll_count += 1
            # 迷宫诡域在周期征讨列表顶部，向下拖动可恢复上次留下的滚动位置。
            start = titles_area.center
            self.ctx.controller.drag_to(start=start, end=start + Point(0, 300))
            return self.round_wait('向上查找迷宫诡域卡片', wait=0.5)
        buttons = self.ctx.ocr_service.get_ocr_result_list(
            image=self.last_screenshot, rect=buttons_area.rect,
        )
        # 1080p 卡片中按钮在标题下约 100 像素；下一张卡片的按钮不在此范围。
        candidates = [item for item in buttons
                      if str_utils.remove_whitespace(item.data) == gt('前往', 'game')
                      and 0 < item.center.y - target.center.y <= 150]
        if len(candidates) != 1:
            return self.round_fail('无法确认迷宫诡域卡片的前往按钮，停止并保留现场')
        if not self.ctx.controller.click(candidates[0].center):
            return self.round_retry('点击迷宫诡域前往失败', wait=0.5)
        self.node_clicked = True
        return self.round_wait('已点击迷宫诡域前往，等待贝果主界面', wait=0.5)
