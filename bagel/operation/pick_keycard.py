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
    PAGE_ROWS_MAX,
    TAB_CLOSED,
    TAB_SELECTED,
    TOOLTIP_DISMISS_POS,
    TOOLTIP_SAFE_BUTTON_RECT,
    TOOLTIP_TITLE_RECT,
    TOOLTIP_TYPE_RECT,
    TOP_ROW_Y,
    VISIBLE_ROWS,
    GridCell,
    RowTracker,
    arrow_pixels,
    grid_cells,
    grid_rows,
    more_below,
    page_tab_states,
    rows_same,
    scroll_effect,
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

KEYCARD_SCORE_MIN: float = 0.95
"""有图标模板时，图标相似度不低于它才算本房间的卡：挑卡只点开这些格子核对浮窗，清点按它计数、不点开浮窗。
取在实机同款卡（0.98 以上）与最接近的别款卡（同图案不同字母的 D 卡约 0.90）之间。"""
CANDIDATE_LIMIT: int = 8
"""每一屏最多点开核对的格子数；卡按种类扎堆，排在前面的足够覆盖。"""
SCROLL_NOTCHES: int = 1
"""每次向下（上）滚动的滚轮格数。取 1 格：1 格在各次实机里都正好滚一整行（122 px）。
曾经 1–8 格都只滚一行，但有一次游戏重启后 5 格一条命令滚了至少 3 行，直接滚到本页底，
行位置错位而停止清点；原因未查明，所以取最小的 1 格。"""
TOP_COMMANDS: int = 5
"""点已选中的页签后要发的向上滚动命令数：列表可能停在中间，一页最多 8 行、最多向下滚 3 次，
每条命令滚一行，5 条足够回到顶部。列表滚离顶部后画面没有顶部提示，所以只能靠命令数。
切到另一页时列表本来就在顶部，只发 1 条（用来唤出 ▼ 提示）。"""
FILTER_CHECKS: int = 4
"""切换类型筛选后最多点几次，直到仓库标题显示对应类型。"""
PLACE_CHECKS: int = 4
"""点「放入安全箱」后最多看几帧确认安全箱第 1 格有了物品。"""
DISMISS_CHECKS: int = 3
"""滚动前最多点几次标题栏关浮窗；仍关不掉就停止保留现场。"""
PAGE_CHECKS: int = 3
"""点分页页签后最多点几次，直到该页签显示为选中；仍没选中就停止并保留现场。"""
SETTLE_WAIT: float = 0.3
"""滚动后等列表停稳时，两次截图的间隔（秒）。滚动动画实测 0.2–0.45 s。"""
SETTLE_MAX_CHECKS: int = 8
"""滚动后最多截几次等列表停稳；用完仍在动就按当前画面继续，并记警告。"""
STUCK_SCROLL_MAX: int = 3
WAKE_SCROLL_MAX: int = 3
"""每页清点前的向上滚动发完、停稳后 ▼ 提示仍不在，最多补发到这么多条；
仍不在就当本页滚不动（不超过一屏），只数这一屏。"""
TOP_MAX_EXTRA: int = 6
"""发完向上滚动命令后，最多再等几次截图让第 1 行落回顶部位置、画面停稳；仍不行就停止并保留现场。
列表在顶部时再向上滚会回弹，第 1 行要过一会儿才落回顶部。"""
TOP_ROW_TOLERANCE: int = 6
"""第 1 行行心与 ``TOP_ROW_Y`` 相差不超过它，算在顶部。"""


class BagelPickKeycard(BagelOperation):
    """在仓库挑 1 张指定房间的门禁卡放进安全箱。

    ``reserve`` 大于 0 时先清点：翻完全部页，按图标相似度数本房间的卡，不点开；
    不多于 ``reserve`` 张就以 ``STATUS_RESERVE`` 结束，不挑。

    挑卡时逐页、逐屏按图标相似度点开候选格，OCR 浮窗标题与房间名一致、类型是门禁卡，
    才点「放入安全箱」。所有页都翻完仍没有就以 ``STATUS_NO_CARD`` 结束。
    每切到一页先向上滚，回到顶部并刷新 ▼ 提示，再逐行向下滚，▼ 消失判本页到底。

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
        self.row_counts: dict[tuple[int, int], int] = {}
        """清点时每页每个绝对行号上数到的本房间的卡；一行被多次看到时取最大值，总数是它们的和。"""
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
        """正在做每页清点前的向上滚动（唤出 ▼ 提示，必要时回到顶部）。"""
        self.top_needed: int = 0
        """本页要发的向上滚动命令数。"""
        self.top_retries: int = 0
        """▼ 提示没出现而补发的向上滚动命令数。"""
        self.top_sent: int = 0
        """本页已发出的向上滚动命令数。"""
        self.top_extra: int = 0
        self.top_rows: list[int] | None = None
        """发完命令后上一次截图里各完整行的行心纵坐标。"""
        self.page: int = 0
        self.need_page_click: bool = True
        self.page_clicks: int | None = None
        """已点当前页签的次数；非 None 表示正在等页签显示为选中。"""
        self.tracker: RowTracker | None = None
        """本页的滚动记账；滚回顶部后在第一屏建立。"""
        self.scrolling: bool = False
        """刚发出向下滚动，正在等它的结果。"""
        self.scroll_before: MatLike | None = None
        """发出向下滚动前的整屏截图，用来判断命令生效没有。"""
        self.settle_rows: list[int] | None = None
        """等列表停稳时上一次截图里各完整行的行心纵坐标。"""
        self.settle_arrow: bool | None = None
        """等列表停稳时上一次截图里 ▼ 提示在不在。"""
        self.settle_checks: int = 0
        """这次滚动后已截了几次图等结果。"""
        self.stuck_scrolls: int = 0
        self.top_arrow: bool | None = None
        """向上滚动后上一次截图里 ▼ 提示在不在。"""
        """▼ 提示还在、滚动命令却没生效的连续次数。"""
        self.tried: set[tuple[int, int]] = set()
        self.pending: GridCell | None = None
        self.placing_checks: int | None = None
        self.opened: int = 0
        self.viewing_backpack: bool = False
        self.dismiss_clicks: int = 0
        """本次滚动前为关浮窗点了几次标题栏。"""

    def handle_init(self) -> None:
        """重复执行时从第 1 页重新找。"""
        super().handle_init()
        self.counting = self.reserve > 0
        self.counted = None
        self.row_counts = {}
        self.count_crops = []
        self.filter_clicks = 0
        self.filter_ready = False
        self.lock_hit = False
        self.result = None
        self.restore_clicks = 0
        self._reset_top()
        self.page = 0
        self.need_page_click = True
        self.page_clicks = None
        self.tracker = None
        self.scrolling = False
        self.scroll_before = None
        self.settle_rows = None
        self.settle_arrow = None
        self.settle_checks = 0
        self.stuck_scrolls = 0
        self.tried = set()
        self.pending = None
        self.placing_checks = None
        self.opened = 0
        self.viewing_backpack = False
        self.dismiss_clicks = 0

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
            self.page_clicks = None
            return self._pages_done(f'第 {self.page + 1} 页未解锁')
        if self.placing_checks is not None:
            return self._check_placed()
        if self.pending is not None:
            return self._check_tooltip()
        if self.need_page_click:
            return self._click_page()
        if self.page_clicks is not None:
            waiting = self._confirm_page()
            if waiting is not None:
                return waiting
        if self.to_top:
            return self._scroll_top()
        if self.scrolling:
            waiting = self._wait_scroll_settled()
            if waiting is not None:
                return waiting
        self._ensure_tracker()
        if self.counting:
            return self._count_screen()
        cell = self._next_candidate()
        if cell is None:
            if self._tooltip_open():
                return self._dismiss_tooltip()
            self.dismiss_clicks = 0
            return self._scroll_or_next_page()
        self.pending = cell
        self.tried.add(cell.center.tuple())
        self.opened += 1
        self.ctx.controller.click(cell.center, press_time=0.05)
        return self.round_wait('点开候选格查看浮窗', wait=0.5)

    def _next_candidate(self) -> GridCell | None:
        cells = grid_cells(self.last_screenshot, self.template_id)
        if self.template_id is not None and any(cell.score > 0 for cell in cells):
            cells = [cell for cell in cells if cell.score >= KEYCARD_SCORE_MIN][:CANDIDATE_LIMIT]
        return next((cell for cell in cells if cell.center.tuple() not in self.tried), None)

    def _tooltip_open(self) -> bool:
        return '门禁' in self._ocr(TOOLTIP_TYPE_RECT)

    def _dismiss_tooltip(self) -> OperationRoundResult:
        """滚动前关掉上一格的物品浮窗：浮窗开着时滚轮滚不动网格，会被误判为本页到底。"""
        if self.dismiss_clicks >= DISMISS_CHECKS:
            return self.round_fail(f'点了 {self.dismiss_clicks} 次仓库标题栏，物品浮窗仍没关，停止并保留现场')
        self.dismiss_clicks += 1
        self.ctx.controller.click(TOOLTIP_DISMISS_POS, press_time=0.05)
        return self.round_wait('滚动前关掉物品浮窗', wait=0.5)

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
        """每页清点前先向上滚，唤出 ▼ 提示；点的是已选中的页时同时回到顶部。

        切页后 ▼ 提示还是这一页上次离开时的状态，要滚动过才更新。切到另一页时列表在顶部，
        向上滚一条不会动，只用来刷新 ▼。命令发完后等第 1 行落回顶部位置、相邻两次截图的行
        和 ▼ 都不变，等不到就失败。停稳后 ▼ 不在：可能这条命令丢了，也可能本页不超过一屏，
        补发向上滚动，共 ``WAKE_SCROLL_MAX`` 条仍不在就按本页滚不动处理。
        """
        if self.top_sent < self.top_needed:
            self.top_sent += 1
            self.top_rows = None
            self.top_arrow = None
            self.ctx.controller.scroll(-SCROLL_NOTCHES, GRID_SCROLL_POS)
            return self.round_wait(f'第 {self.page + 1} 页向上滚动（唤出 ▼ 提示）', wait=0.6)
        screen = self.last_screenshot
        rows = grid_rows(screen)
        arrow = more_below(screen)
        previous, previous_arrow = self.top_rows, self.top_arrow
        self.top_rows, self.top_arrow = rows, arrow
        settled = (bool(rows) and abs(rows[0] - TOP_ROW_Y) <= TOP_ROW_TOLERANCE
                   and previous is not None and rows_same(previous, rows) and previous_arrow == arrow)
        if not settled:
            self.top_extra += 1
            if self.top_extra > TOP_MAX_EXTRA:
                first = rows[0] if rows else None
                return self._finish(self.round_fail(
                    f'第 {self.page + 1} 页向上滚动后第 1 行在 y={first}（顶部应为 {TOP_ROW_Y}）或画面一直在动，'
                    '停止并保留现场'))
            return self.round_wait('等第 1 行落回顶部、画面停稳', wait=0.3)
        if not arrow and self.top_retries < WAKE_SCROLL_MAX - 1:
            self.top_retries += 1
            self.top_needed += 1
            self.top_extra = 0
            return self.round_wait('▼ 提示没出现，补发向上滚动', wait=0.05)
        log.info('贝果挑卡：第 %s 页已发 %s 条向上滚动命令，第 1 行在 y=%s，▼ %s（像素 %s）',
                 self.page + 1, self.top_sent, rows[0], '在' if arrow else '不在，按本页滚不动处理',
                 arrow_pixels(screen))
        self._reset_top()
        self.tracker = None
        return self.round_wait('已在本页顶部', wait=0.05)

    def _reset_top(self) -> None:
        self.to_top = False
        self.top_needed = 0
        self.top_retries = 0
        self.top_sent = 0
        self.top_extra = 0
        self.top_rows = None
        self.top_arrow = None

    def _ensure_tracker(self) -> None:
        """滚回顶部后的第一屏建立滚动记账：第 1 个完整行记为第 0 行。"""
        if self.tracker is None:
            rows = grid_rows(self.last_screenshot)
            if rows:
                self.tracker = RowTracker(rows)

    def _count_screen(self) -> OperationRoundResult:
        """数当前屏各完整行里本房间的卡，按绝对行号记账，一行只算一次；然后向下滚。

        同一行在不同滚动位置被看到多次时取读数最大的一次：靠近列表上下边缘的卡片会被调暗，
        单次读数可能偏低。
        """
        rows = grid_rows(self.last_screenshot)
        tracker = self.tracker
        cells = grid_cells(self.last_screenshot, self.template_id)
        indexes: list[int] = []
        new_rows = 0
        if tracker is not None:
            for y in rows:
                index = tracker.row_index(y)
                indexes.append(index)
                found = sum(1 for cell in cells if cell.center.y == y and cell.score >= KEYCARD_SCORE_MIN)
                key = (self.page, index)
                before = self.row_counts.get(key)
                if before is None:
                    new_rows += 1
                    self.row_counts[key] = found
                elif found != before:
                    log.info('贝果挑卡清点：第 %s 页第 %s 行两次读数不同（%s、%s 张），取大',
                             self.page + 1, index, before, found)
                    self.row_counts[key] = max(before, found)
        self.counted = sum(self.row_counts.values())
        if new_rows:
            self.count_crops.append(
                self.last_screenshot[GRID_RECT.y1:GRID_RECT.y2, GRID_RECT.x1:GRID_RECT.x2].copy())
        span = f'{min(indexes)}–{max(indexes)}' if indexes else '无'
        top = '、'.join(f'{cell.score:.3f}' for cell in cells[:6]) or '无'
        log.info('贝果挑卡清点：第 %s 页已向下滚 %s 行，本屏 %s 行（行号 %s），新行 %s 行、%s 格有物，累计 %s 张；'
                 '▼ 像素 %s；最高相似度 %s',
                 self.page + 1, tracker.offset if tracker is not None else 0, len(rows), span, new_rows,
                 len(cells), self.counted, arrow_pixels(self.last_screenshot), top)
        return self._scroll_or_next_page()

    def _scroll_or_next_page(self) -> OperationRoundResult:
        """▼ 提示还在就向下滚一行；已经消失说明本页到底，换下一页。▼ 已在本页开头向上滚动时唤出。"""
        screen = self.last_screenshot
        if not more_below(screen):
            log.info('贝果挑卡：第 %s 页 ▼ 已消失，本页到底（已向下滚 %s 行，▼ 像素 %s）',
                     self.page + 1, self.tracker.offset if self.tracker is not None else 0, arrow_pixels(screen))
            return self._next_page()
        if self.tracker is not None and self.tracker.offset >= PAGE_ROWS_MAX - VISIBLE_ROWS:
            return self._finish(self.round_fail(
                f'第 {self.page + 1} 页已向下滚 {self.tracker.offset} 行，▼ 提示仍在；一页最多 {PAGE_ROWS_MAX} 行，'
                '记账或画面不对，停止并保留现场'))
        self._scroll_down()
        return self.round_wait(f'第 {self.page + 1} 页向下滚动一行', wait=0.6)

    def _scroll_down(self) -> None:
        """向下滚一行；下一轮由 ``_wait_scroll_settled`` 等结果。"""
        self.scrolling = True
        self.scroll_before = self.last_screenshot.copy()
        self.settle_rows = None
        self.settle_arrow = None
        self.settle_checks = 0
        self.ctx.controller.scroll(SCROLL_NOTCHES, GRID_SCROLL_POS)

    def _clear_scroll_wait(self) -> None:
        self.scrolling = False
        self.scroll_before = None
        self.settle_rows = None
        self.settle_arrow = None
        self.settle_checks = 0

    def _wait_scroll_settled(self) -> OperationRoundResult | None:
        """滚动后等列表停稳；仍要等或要重滚就返回结果，命令生效返回 None 继续。

        实机每条滚动命令正好滚一行，行的位置（相位）不变，所以不靠位置判断滚没滚：
        - 停稳：相邻两次截图（间隔 ``SETTLE_WAIT``）行的纵坐标和 ▼ 状态都不变。
        - 命令发出前 ▼ 在（本页开头已向上滚动唤出），列表还能滚：画面变了算滚了一行；画面没变而可见各行有不同的卡（``still``），
          是命令丢了，重发，连续 ``STUCK_SCROLL_MAX`` 次都不动就失败；可见各行全是同款卡时分不出
          （``unknown``），按滚了一行记账并写日志，这是已知的不确定处，见 docs/keycard.md。
        """
        screen = self.last_screenshot
        rows = grid_rows(screen)
        arrow = more_below(screen)
        previous_rows, previous_arrow = self.settle_rows, self.settle_arrow
        self.settle_rows, self.settle_arrow = rows, arrow
        self.settle_checks += 1
        stable = previous_rows is not None and previous_arrow == arrow and rows_same(previous_rows, rows)
        if not stable:
            if self.settle_checks < SETTLE_MAX_CHECKS:
                return self.round_wait('等列表滚动停稳', wait=SETTLE_WAIT)
            log.warning('贝果挑卡：滚动后截了 %s 次列表仍在动，按当前画面继续', self.settle_checks)
        checks = self.settle_checks
        before = self.scroll_before
        self._clear_scroll_wait()
        effect = scroll_effect(before, screen) if before is not None else 'unknown'
        if effect == 'still':
            self.stuck_scrolls += 1
            if self.stuck_scrolls >= STUCK_SCROLL_MAX:
                count = self.stuck_scrolls
                self.stuck_scrolls = 0
                return self._finish(self.round_fail(
                    f'列表还能向下滚（▼ 提示仍在），连续 {count} 次滚动命令画面都没变，停止并保留现场'))
            self._scroll_down()
            return self.round_wait('滚动命令没生效，重发', wait=0.6)
        self.stuck_scrolls = 0
        if self.tracker is not None:
            self.tracker.advance()
            if not self.tracker.aligned(rows):
                # 结束时会先切回「全部」筛选再截现场图，那张图看不到失败瞬间，这里当场存滚动前后两张。
                before_rows = grid_rows(before) if before is not None else None
                log.warning(
                    '贝果挑卡：滚动后行位置错位，滚动前行 %s、▼ %s，滚动后行 %s、▼ %s，停稳检查 %s 次',
                    before_rows, more_below(before) if before is not None else None,
                    rows, arrow, checks,
                )
                save_screenshot(before, 'BagelScrollBefore')
                save_screenshot(screen, 'BagelScrollAfter')
                return self._finish(self.round_fail(
                    f'第 {self.page + 1} 页滚动后行的位置与顶部不一致（{rows}），滚动不是整行，记账不可信，停止并保留现场'))
        if effect == 'unknown':
            log.info('贝果挑卡：第 %s 页第 %s 次滚动前后画面分不出动没动（整块同款卡），按滚了一行记账',
                     self.page + 1, self.tracker.offset if self.tracker is not None else 0)
        if checks > 3:
            log.info('贝果挑卡：滚动后第 %s 次截图列表才停稳', checks)
        self.tried.clear()
        return None

    def _pages_done(self, reason: str) -> OperationRoundResult:
        """全部页看完：清点阶段据此决定挑不挑，挑卡阶段就是没找到。"""
        if not self.counting:
            return self._no_card(reason)
        counted = self.counted or 0
        log.info('贝果挑卡：清点完毕（%s），仓库有「%s」的门禁卡 %s 张（图标相似度不低于 %.2f），保留 %s 张',
                 reason, self.room_name, counted, KEYCARD_SCORE_MIN, self.reserve)
        if counted <= self.reserve:
            if self.count_crops:
                path = save_screenshot(np.vstack(self.count_crops), 'BagelKeycardCount')
                log.info('贝果挑卡：清点的各屏网格已拼图保存：%s', path)
            return self._finish(self.round_fail(self.STATUS_RESERVE))
        self.counting = False
        self.page = 0
        self.need_page_click = True
        self.page_clicks = None
        self._clear_scroll_wait()
        self.tracker = None
        return self.round_wait('清点完毕，开始挑卡', wait=0.05)

    def _click_page(self) -> OperationRoundResult:
        """点下一页的页签。页签暗色（筛选后该页没有这类物品）或锁定，就算全部页看完。

        不用「点了以后网格变没变」判断翻页：相邻两页都是整块同款卡时网格看起来一样，
        会把能翻的页误判成翻不过去，漏掉后面所有页。

        点完先向上滚（``_scroll_top``）：切到另一页时列表在顶部，向上滚 1 条只为唤出 ▼ 提示；
        点的是已选中的页（刚打开仓库，或清点完回到同一页挑卡）时列表可能停在中间，向上滚 5 条回到顶部。
        """
        if self.page >= len(PAGE_CENTERS):
            return self._pages_done('已翻完全部分页')
        states = page_tab_states(self.last_screenshot)
        if states[self.page] == TAB_CLOSED:
            return self._pages_done(f'第 {self.page + 1} 页页签是暗的或已锁定（页签 {"、".join(states)}）')
        self.need_page_click = False
        self.page_clicks = 1
        self._reset_top()
        self.to_top = True
        self.top_needed = TOP_COMMANDS if states[self.page] == TAB_SELECTED else 1
        self.stuck_scrolls = 0
        self.tracker = None
        self._clear_scroll_wait()
        self.tried.clear()
        self.ctx.controller.click(PAGE_CENTERS[self.page], press_time=0.05)
        return self.round_wait(f'切到仓库第 {self.page + 1} 页', wait=0.8)

    def _confirm_page(self) -> OperationRoundResult | None:
        """等刚点的页签显示为选中；选中返回 None 继续，否则重点，点够次数仍没选中就停止。"""
        states = page_tab_states(self.last_screenshot)
        if states[self.page] == TAB_SELECTED:
            self.page_clicks = None
            return None
        clicks = self.page_clicks or 0
        if clicks >= PAGE_CHECKS:
            return self._finish(self.round_fail(
                f'点了 {clicks} 次第 {self.page + 1} 页页签仍没选中（页签 {"、".join(states)}），停止并保留现场'))
        self.page_clicks = clicks + 1
        self.ctx.controller.click(PAGE_CENTERS[self.page], press_time=0.05)
        return self.round_wait(f'第 {self.page + 1} 页页签没选中，重点', wait=0.8)

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
