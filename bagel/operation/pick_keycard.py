from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils.log_utils import log

from ..screenshot import save_screenshot
from ..vision.keycard_pick import (
    FILTER_ALL_POS,
    FILTER_KEYCARD_POS,
    GRID_RECT,
    GRID_SCROLL_POS,
    LOCK_CONFIRM_RECT,
    PAGE_CENTERS,
    TOOLTIP_SAFE_BUTTON_RECT,
    TOOLTIP_TITLE_RECT,
    TOOLTIP_TYPE_RECT,
    GridCell,
    grid_cells,
    grid_changed,
    grid_rows,
    grid_signature,
    row_strips,
    scroll_overlap,
    title_matches,
)
from ..vision.screen import read_area
from ..vision.slots import safe_occupied_indices
from ..vision.template import load_template
from .operation import BagelOperation

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from one_dragon.base.geometry.rectangle import Rect
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

CANDIDATE_SCORE_MIN: float = 0.85
"""有图标模板时，相似度不低于它的格子才点开核对。实机同款卡 0.98 以上，
同图案不同字母的 D 卡约 0.90，同字母不同底色的卡约 0.86，其他门禁卡更低。"""
COUNT_SCORE_MIN: float = 0.95
"""按保留张数清点时，图标相似度不低于它才算一张本房间的卡；清点不点开浮窗。
取在实机同款卡（0.98 以上）与最接近的别款卡（约 0.90）之间。"""
CANDIDATE_LIMIT: int = 8
"""每一屏最多点开核对的格子数；卡按种类扎堆，排在前面的足够覆盖。"""
SCROLL_NOTCHES: int = 5
"""每次向下滚动的滚轮格数。"""
SCROLL_TOP_NOTCHES: int = 20
"""切页后回到顶部时，每次向上滚动的滚轮格数。"""
FILTER_CHECKS: int = 4
"""切换类型筛选后最多点几次，直到仓库标题显示对应类型。"""
PLACE_CHECKS: int = 4
"""点「放入安全箱」后最多看几帧确认安全箱第 1 格有了物品。"""


class BagelPickKeycard(BagelOperation):
    """在仓库挑 1 张指定房间的门禁卡放进安全箱。

    ``reserve`` 大于 0 时先清点：翻完全部页，按图标相似度数本房间的卡，不点开；
    不多于 ``reserve`` 张就以 ``STATUS_RESERVE`` 结束，不挑。

    挑卡时逐页、逐屏按图标相似度点开候选格，OCR 浮窗标题与房间名一致、类型是门禁卡，
    才点「放入安全箱」。所有页都翻完仍没有就以 ``STATUS_NO_CARD`` 结束。
    每切到一页先滚回顶部，避免从上次停下的位置开始找。

    ``from_prepare`` 为真时从备战页进出仓库；否则就在当前仓库（结算仓库）挑，挑完不离开。
    """

    STATUS_NO_CARD: str = '仓库没有该房间的门禁卡'
    STATUS_PICKED: str = '已挑卡放入安全箱'
    STATUS_RESERVE: str = '仓库该房间的门禁卡已到保留张数'
    STOP_STATUSES: tuple[str, ...] = (STATUS_NO_CARD, STATUS_RESERVE)
    """仓库没有可用的卡：调用方应正常结束任务。"""

    def __init__(
        self, ctx: ZContext, room_name: str, template_id: str | None, *, from_prepare: bool,
        reserve: int = 0,
    ) -> None:
        """
        Args:
            room_name: 房间显示名称，与门禁卡浮窗标题一致。
            template_id: 该房间门禁卡的图标模板；None 时按格子顺序逐个核对。
            from_prepare: 是否从备战页进入仓库并在挑完后返回备战。
            reserve: 保留张数；大于 0 时先清点，必须有图标模板。
        """
        super().__init__(ctx, op_name='贝果-挑选门禁卡', timeout_seconds=300 if reserve == 0 else 480)
        self.room_name: str = room_name
        self.template_id: str | None = template_id
        self.from_prepare: bool = from_prepare
        self.reserve: int = reserve
        self.counting: bool = reserve > 0
        """当前是否在清点阶段；清点完转入挑卡。"""
        self.counted: int | None = None
        """清点到的张数；没清点为 None。"""
        self.near_scores: list[float] = []
        """清点时相似度在候选线与计数线之间的格子，写进日志供标定。"""
        self.prev_strips: list[MatLike] = []
        self.count_crops: list[MatLike] = []
        """清点时每屏的网格截图；按保留张数结束时拼成一张存下，供核对数得对不对。"""
        self.filter_clicks: int = 0
        """已点「门禁卡」筛选的次数；标题显示门禁卡后不再点。"""
        self.filter_ready: bool = False
        self.lock_hit: bool = False
        """刚点到锁定分页并关掉了解锁提示。"""
        self.result: OperationRoundResult | None = None
        """挑卡已有结论，正在把筛选切回「全部」；切回后返回它。"""
        self.restore_clicks: int = 0
        self.to_top: bool = False
        self.top_signature: MatLike | None = None
        self.page: int = 0
        self.need_page_click: bool = True
        self.page_signature: MatLike | None = None
        self.scroll_signature: MatLike | None = None
        self.tried: set[tuple[int, int]] = set()
        self.pending: GridCell | None = None
        self.placing_checks: int | None = None
        self.opened: int = 0
        self.viewing_backpack: bool = False

    def handle_init(self) -> None:
        """重复执行时从第 1 页重新找。"""
        super().handle_init()
        self.counting = self.reserve > 0
        self.counted = None
        self.near_scores = []
        self.prev_strips = []
        self.count_crops = []
        self.filter_clicks = 0
        self.filter_ready = False
        self.lock_hit = False
        self.result = None
        self.restore_clicks = 0
        self.to_top = False
        self.top_signature = None
        self.page = 0
        self.need_page_click = True
        self.page_signature = None
        self.scroll_signature = None
        self.tried = set()
        self.pending = None
        self.placing_checks = None
        self.opened = 0
        self.viewing_backpack = False

    def _at_warehouse(self) -> bool:
        return self.round_by_find_area(self.last_screenshot, '贝果-仓库', '放入仓库').is_success

    def _at_prepare(self) -> bool:
        return self.round_by_find_area(self.last_screenshot, '贝果-备战', '预设组合').is_success

    def _ocr(self, rect: Rect) -> str:
        matches = self.ctx.ocr_service.get_ocr_result_list(self.last_screenshot, rect=rect, crop_first=True)
        return ''.join(match.data for match in sorted(matches, key=lambda match: (match.y, match.x)))

    @operation_node(name='打开挑卡仓库', is_start_node=True, timeout_seconds=20)
    def open_warehouse(self) -> OperationRoundResult:
        """备战页先切到背包视图，再点「前往仓库」；已在仓库时直接开始。"""
        if self._at_warehouse():
            return self.round_success()
        if not self.from_prepare:
            return self.round_retry('等待仓库画面', wait=0.5)
        if not self._at_prepare():
            return self.round_retry('等待备战仓库打开', wait=0.5)
        if self.node_clicked:
            return self.round_wait('已点击前往仓库，等待切换', wait=0.5)
        if self.round_by_find_area(self.last_screenshot, '贝果-备战', '前往仓库').is_success:
            result = self.round_by_find_and_click_area(
                self.last_screenshot, '贝果-备战', '前往仓库', success_wait=0.5, retry_wait=0.5,
            )
            if result.is_success:
                self.node_clicked = True
                return self.round_wait('等待备战仓库', wait=0.5)
            return result
        if not self.viewing_backpack:
            self.viewing_backpack = True
            result = self.round_by_click_area('贝果-备战', '背包数量')
            if result.is_success:
                return self.round_wait('切换到备战背包视图', wait=0.5)
            return result
        return self.round_retry('未识别前往仓库', wait=0.5)

    @node_from(from_name='打开挑卡仓库')
    @operation_node(name='核对挑卡安全箱', timeout_seconds=10)
    def check_safe(self) -> OperationRoundResult:
        """安全箱必须为空：挑卡只放这 1 张，放进去后靠第 1 格有物确认。"""
        occupied = safe_occupied_indices(self.last_screenshot)
        if occupied is None:
            return self.round_retry('仓库安全箱格子状态不明', wait=0.5)
        if occupied:
            return self.round_fail(f'安全箱已有物品（格 {occupied}），不挑卡')
        if self.counting and (self.template_id is None or load_template(self.template_id) is None):
            return self.round_fail('没有该房间门禁卡的图标模板，无法按保留张数清点')
        return self.round_success()

    @node_from(from_name='核对挑卡安全箱')
    @operation_node(name='查找门禁卡', timeout_seconds=420)
    def find_card(self) -> OperationRoundResult:
        """一轮只做一步：筛选、翻页、滚动、清点一屏、点开一格或核对一格浮窗，便于暂停与超时。"""
        if not self._at_warehouse():
            if self._ocr(LOCK_CONFIRM_RECT).find('确认') >= 0:
                self.ctx.controller.click(LOCK_CONFIRM_RECT.center, press_time=0.05)
                self.lock_hit = True
                return self.round_wait('分页未解锁，关闭解锁提示', wait=0.8)
            return self.round_retry('等待仓库画面', wait=0.5)
        if self.result is not None:
            return self._restore_filter()
        if not self.filter_ready:
            return self._apply_filter()
        if self.lock_hit:
            self.lock_hit = False
            self.page_signature = None
            return self._pages_done(f'第 {self.page + 1} 页未解锁')
        if self.placing_checks is not None:
            return self._check_placed()
        if self.pending is not None:
            return self._check_tooltip()
        if self.need_page_click:
            return self._click_page()
        if self.page_signature is not None:
            # 刚翻页：网格没变说明这一页锁定或没有更多页。
            changed = grid_changed(self.page_signature, grid_signature(self.last_screenshot))
            self.page_signature = None
            if not changed:
                return self._pages_done(f'第 {self.page + 1} 页翻不过去')
        if self.to_top:
            return self._scroll_top()
        if self.scroll_signature is not None:
            changed = grid_changed(self.scroll_signature, grid_signature(self.last_screenshot))
            self.scroll_signature = None
            if not changed:
                return self._next_page()
            self.tried.clear()
        if self.counting:
            return self._count_screen()
        cell = self._next_candidate()
        if cell is None:
            self.scroll_signature = grid_signature(self.last_screenshot)
            self.ctx.controller.scroll(SCROLL_NOTCHES, GRID_SCROLL_POS)
            return self.round_wait(f'第 {self.page + 1} 页向下滚动', wait=0.6)
        self.pending = cell
        self.tried.add(cell.center.tuple())
        self.opened += 1
        self.ctx.controller.click(cell.center, press_time=0.05)
        return self.round_wait('点开候选格查看浮窗', wait=0.5)

    def _next_candidate(self) -> GridCell | None:
        cells = grid_cells(self.last_screenshot, self.template_id)
        if self.template_id is not None and any(cell.score > 0 for cell in cells):
            cells = [cell for cell in cells if cell.score >= CANDIDATE_SCORE_MIN][:CANDIDATE_LIMIT]
        return next((cell for cell in cells if cell.center.tuple() not in self.tried), None)

    def _header(self) -> str:
        return read_area(self.ctx, self.last_screenshot, '贝果-仓库', '仓库数量')

    def _apply_filter(self) -> OperationRoundResult:
        """点网格上方的「门禁卡」类型筛选，标题显示门禁卡后开始找。"""
        header = self._header()
        if '门禁' in header:
            log.info('贝果挑卡：已切到门禁卡筛选（%s）', header)
            self.filter_ready = True
            return self.round_wait('已切到门禁卡筛选', wait=0.05)
        if self.filter_clicks >= FILTER_CHECKS:
            log.warning('贝果挑卡：点了 %s 次门禁卡筛选，标题仍是「%s」，按当前筛选继续找', self.filter_clicks, header)
            self.filter_ready = True
            return self.round_wait('门禁卡筛选未确认，继续', wait=0.05)
        self.filter_clicks += 1
        self.ctx.controller.click(FILTER_KEYCARD_POS, press_time=0.05)
        return self.round_wait('点击门禁卡筛选', wait=0.8)

    def _finish(self, result: OperationRoundResult) -> OperationRoundResult:
        """记下结论，先把类型筛选切回「全部」再返回；结算仓库的数量读数依赖「全部」。"""
        self.result = result
        return self.round_wait('挑卡结束，切回全部筛选', wait=0.05)

    def _restore_filter(self) -> OperationRoundResult:
        header = self._header()
        if '全部' in header or self.filter_clicks == 0 or self.restore_clicks >= FILTER_CHECKS:
            if '全部' not in header and self.filter_clicks > 0:
                log.warning('贝果挑卡：没能切回全部筛选，标题「%s」', header)
            result = self.result
            self.result = None
            return result
        self.restore_clicks += 1
        self.ctx.controller.click(FILTER_ALL_POS, press_time=0.05)
        return self.round_wait('点击全部筛选', wait=0.8)

    def _scroll_top(self) -> OperationRoundResult:
        """切页后向上滚到网格不再变化为止。"""
        signature = grid_signature(self.last_screenshot)
        if self.top_signature is not None and not grid_changed(self.top_signature, signature):
            self.to_top = False
            self.top_signature = None
            return self.round_wait('已在本页顶部', wait=0.05)
        self.top_signature = signature
        self.ctx.controller.scroll(-SCROLL_TOP_NOTCHES, GRID_SCROLL_POS)
        return self.round_wait(f'第 {self.page + 1} 页滚回顶部', wait=0.6)

    def _count_screen(self) -> OperationRoundResult:
        """数当前屏新露出的行里本房间的卡，然后向下滚。与上一屏重叠的行不重复数。"""
        rows = grid_rows(self.last_screenshot)
        strips = row_strips(self.last_screenshot, rows)
        skip = scroll_overlap(self.prev_strips, strips) if self.prev_strips else 0
        fresh = set(rows[skip:])
        found = 0
        cells = grid_cells(self.last_screenshot, self.template_id)
        for cell in cells:
            if cell.center.y not in fresh:
                continue
            if cell.score >= COUNT_SCORE_MIN:
                found += 1
            elif cell.score >= CANDIDATE_SCORE_MIN:
                self.near_scores.append(cell.score)
        self.counted = (self.counted or 0) + found
        self.count_crops.append(
            self.last_screenshot[GRID_RECT.y1:GRID_RECT.y2, GRID_RECT.x1:GRID_RECT.x2].copy())
        top = '、'.join(f'{cell.score:.3f}' for cell in cells[:6]) or '无'
        log.info('贝果挑卡清点：第 %s 页本屏 %s 行（与上屏重叠 %s 行）、%s 格有物，数到 %s 张，累计 %s 张；最高相似度 %s',
                 self.page + 1, len(rows), skip, len(cells), found, self.counted, top)
        self.prev_strips = strips
        self.scroll_signature = grid_signature(self.last_screenshot)
        self.ctx.controller.scroll(SCROLL_NOTCHES, GRID_SCROLL_POS)
        return self.round_wait(f'清点第 {self.page + 1} 页，向下滚动', wait=0.6)

    def _pages_done(self, reason: str) -> OperationRoundResult:
        """全部页看完：清点阶段据此决定挑不挑，挑卡阶段就是没找到。"""
        if not self.counting:
            return self._no_card(reason)
        counted = self.counted or 0
        near = '、'.join(f'{score:.3f}' for score in sorted(self.near_scores, reverse=True)[:8]) or '无'
        log.info('贝果挑卡：清点完毕（%s），仓库有「%s」的门禁卡 %s 张（图标相似度不低于 %.2f），保留 %s 张；'
                 '相似度在 %.2f–%.2f 之间未计入的格子：%s',
                 reason, self.room_name, counted, COUNT_SCORE_MIN, self.reserve,
                 CANDIDATE_SCORE_MIN, COUNT_SCORE_MIN, near)
        if counted <= self.reserve:
            if self.count_crops:
                path = save_screenshot(np.vstack(self.count_crops), 'BagelKeycardCount')
                log.info('贝果挑卡：清点的各屏网格已拼图保存：%s', path)
            return self._finish(self.round_fail(self.STATUS_RESERVE))
        self.counting = False
        self.page = 0
        self.need_page_click = True
        self.page_signature = None
        self.scroll_signature = None
        self.prev_strips = []
        return self.round_wait('清点完毕，开始挑卡', wait=0.05)

    def _click_page(self) -> OperationRoundResult:
        if self.page >= len(PAGE_CENTERS):
            return self._pages_done('已翻完全部分页')
        self.need_page_click = False
        self.page_signature = None if self.page == 0 else grid_signature(self.last_screenshot)
        self.to_top = True
        self.top_signature = None
        self.prev_strips = []
        self.tried.clear()
        self.ctx.controller.click(PAGE_CENTERS[self.page], press_time=0.05)
        return self.round_wait(f'切到仓库第 {self.page + 1} 页', wait=0.8)

    def _next_page(self) -> OperationRoundResult:
        self.page += 1
        self.need_page_click = True
        return self.round_wait('本页已到底，换下一页', wait=0.1)

    def _no_card(self, reason: str) -> OperationRoundResult:
        log.info('贝果挑卡：%s，共点开 %s 格，没有找到「%s」的门禁卡', reason, self.opened, self.room_name)
        if self.counted:
            return self.round_fail(f'清点到 {self.counted} 张却没挑到，停止并保留现场')
        return self._finish(self.round_fail(self.STATUS_NO_CARD))

    def _check_tooltip(self) -> OperationRoundResult:
        """核对浮窗标题和类型；对上就点「放入安全箱」。"""
        cell = self.pending
        self.pending = None
        title = self._ocr(TOOLTIP_TITLE_RECT)
        kind = self._ocr(TOOLTIP_TYPE_RECT)
        matched = title_matches(title, self.room_name) and '门禁' in kind
        log.info('贝果挑卡：第 %s 页格 %s 相似度 %.3f，浮窗「%s」「%s」%s',
                 self.page + 1, cell.center.tuple(), cell.score, title, kind, '，就是它' if matched else '')
        if not matched:
            return self.round_wait('不是目标卡，看下一格', wait=0.05)
        button = self._ocr(TOOLTIP_SAFE_BUTTON_RECT)
        if '安全箱' not in button:
            return self.round_fail(f'浮窗没有「放入安全箱」按钮（读到「{button}」），停止并保留现场')
        self.ctx.controller.click(TOOLTIP_SAFE_BUTTON_RECT.center, press_time=0.05)
        self.placing_checks = 0
        return self.round_wait('已点放入安全箱', wait=0.8)

    def _check_placed(self) -> OperationRoundResult:
        occupied = safe_occupied_indices(self.last_screenshot)
        if occupied:
            log.info('贝果挑卡：「%s」的门禁卡已放入安全箱，共点开 %s 格', self.room_name, self.opened)
            return self._finish(self.round_success(self.STATUS_PICKED))
        self.placing_checks = (self.placing_checks or 0) + 1
        if self.placing_checks >= PLACE_CHECKS:
            return self.round_fail('点了放入安全箱，安全箱仍为空，停止并保留现场')
        return self.round_wait('等待门禁卡进入安全箱', wait=0.5)

    @node_from(from_name='查找门禁卡')
    @operation_node(name='挑卡后返回备战', timeout_seconds=20)
    def return_prepare(self) -> OperationRoundResult:
        """从备战进来的走左上返回；结算仓库里挑卡则留在原地。"""
        if not self.from_prepare:
            return self.round_success(self.STATUS_PICKED)
        if self._at_prepare():
            return self.round_success(self.STATUS_PICKED)
        if not self._at_warehouse():
            return self.round_wait('等待返回备战', wait=0.5)
        # 浮窗可能吞掉第一次返回点击，间隔点按，由节点超时兜底。
        result = self.round_by_click_area('菜单', '返回')
        if result.is_success:
            return self.round_wait('已点击返回备战', wait=1)
        return result
