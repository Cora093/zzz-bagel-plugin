"""正常退出前整理背包：把背包里比安全箱更值钱的物品换进安全箱。

退出时游戏只保留安全箱物资，背包里的东西带不出去。路上直接拾取的物品都进背包，
所以退出前按 B 打开局内背包页，按入箱同一套优先级逐件填空或对换。

背包页左栏（队伍、道具、背包）整体可以滚动，打开时的滚动位置不固定，背包格不能写死。
每次先用 OCR 找「背包(n/20)」标题，按标题定各排格心；件数多于可见格时先往下滚。
几何取自一张实机截图（1920×1080）：标题行心 y=373 时，背包首排格心 y=450，排距 120，
每排 5 格，列心 x=259/360/461/562/662；卡片上下各约 55 像素。左栏底部固定压着安全箱面板，
卡片下沿须在 y=795 以上才算完整可见。安全箱格心与搜查面板相同（y=898），沿用 ``inspect_safe_slots``。
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from one_dragon.base.controller.drag_timing import DragTiming
from one_dragon.base.geometry.point import Point
from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.utils.log_utils import log

from ..vision.item_vision import (
    ACTION_FILL,
    QUALITY_UNKNOWN,
    BagelSlotMark,
    choose_store_action,
    inspect_occupied,
)
from ..vision.slots import (
    SAFE_SLOT_CENTERS,
    inspect_safe_slots,
    slot_appearance_changed,
    slot_occupied,
)
from .operation import BagelOperation
from .store import record_safe_items

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

    from ..stats import BagelSafeLedger

BACKPACK_KEY: str = 'b'
"""局内打开背包页的按键。"""
_BACKPACK_XS: tuple[int, ...] = (259, 360, 461, 562, 662)
_BACKPACK_ROWS: int = 4
_FIRST_ROW_OFFSET: int = 77
"""标题行心到首排格心的纵向距离。"""
_ROW_STEP: int = 120
_CARD_HALF_HEIGHT: int = 55
_VISIBLE_TOP: int = 95
"""左栏可见区上沿（页签栏下方）。"""
_VISIBLE_BOTTOM: int = 795
"""左栏可见区下沿，再往下是固定的安全箱面板。"""
_SCROLL_POS: Point = Point(461, 600)
"""在左栏上滚动鼠标滚轮的位置。"""
_SCROLL_LIMIT: int = 6
_HEADER_PATTERN: re.Pattern[str] = re.compile(r'背包.*?(\d+)/(\d+)')

_STATUS_NEXT: str = '继续整理'
_REPRESS_SECONDS: float = 3.0
"""按 B 后这么久还没看到背包页，才允许再按一次；过早再按会把刚打开的背包又关掉。"""


class BagelOrganizeBackpack(BagelOperation):
    """打开局内背包页，把背包物品按优先级换进安全箱，再关掉背包页。

    只看定位时完整可见的几排；滚不到的排不参与整理。品质认不出的格子不拖，
    格心没对准时格子会被读成未知品质，不能拿它去换安全箱里的东西。
    任何一步失败都只影响整理本身：调用方照常退出，安全箱里已有的物资不受影响。
    """

    STATUS_DONE: str = '背包整理完成'
    FOREGROUND_DRAG_TIMING: DragTiming = DragTiming(press_wait=0.3, move_duration=0.3, release_wait=0.2)
    BACKGROUND_DRAG_TIMING: DragTiming = DragTiming(press_wait=0.02, move_duration=0.6, release_wait=0.2)
    DRAG_REPEAT_LIMIT: int = 1
    """拖拽后画面没确认时的重拖次数；仍不行就跳过这一格。"""
    ACTION_LIMIT: int = 20
    """本次最多发出的拖拽数；每次对换都让安全箱严格变好，正常远用不到。"""
    PANEL_WAIT_LIMIT: int = 10
    """背包页或安全箱格子认不清时的等待帧数上限（每帧约 0.3 秒）。"""

    def __init__(
        self, ctx: ZContext, blueprint_priority: bool = True,
        ledger: BagelSafeLedger | None = None,
    ) -> None:
        """
        Args:
            ctx: 运行上下文。
            blueprint_priority: 蓝图是否优先入箱，与开箱入箱同一开关。
            ledger: 本局安全箱物品账本；给定时整理后按画面覆盖，只用于统计日志。
        """
        super().__init__(ctx, op_name='贝果-整理背包', timeout_seconds=60)
        self.blueprint_priority: bool = blueprint_priority
        self.ledger: BagelSafeLedger | None = ledger
        self.moved: int = 0
        self.swapped: int = 0
        self._actions: int = 0
        self._skipped: set[int] = set()
        self._panel_wait: int = 0
        self._pending_before: MatLike | None = None
        self._pending_source: int | None = None
        self._pending_dest: int | None = None
        self._pending_kind: str | None = None
        self._repeated: int = 0
        self._pressed_at: float | None = None
        self._centers: tuple[Point, ...] = ()
        self._scrolls: int = 0
        self._last_header_y: int | None = None
        self._logged_unknown: set[int] = set()

    def handle_init(self) -> None:
        """每次执行清空计数与待核对拖拽。"""
        super().handle_init()
        self.moved = 0
        self.swapped = 0
        self._actions = 0
        self._skipped = set()
        self._panel_wait = 0
        self._pressed_at = None
        self._centers = ()
        self._scrolls = 0
        self._last_header_y = None
        self._logged_unknown = set()
        self._clear_pending()

    def _backpack_open(self) -> bool:
        """背包页签和安全箱标题同时可见，且不是开箱的搜查面板。"""
        return (
            self._has('背包页签')
            and self._has('搜查安全箱')
            and not self._has('搜查容器标题')
            and not self._has('电子保险箱标题')
        )

    @operation_node(name='打开背包', is_start_node=True, timeout_seconds=10)
    def open_backpack(self) -> OperationRoundResult:
        """局内画面按 B；开着的搜查面板先按 esc 关掉。"""
        if self.is_bagel_result():
            return self.round_fail(self.STATUS_DEFEATED)
        if self._backpack_open():
            return self.round_success('背包已打开')
        if self._has('搜查容器标题') or self._has('电子保险箱标题'):
            self.ctx.controller.btn_press('esc', press_time=0.1)
            return self.round_wait('先关闭搜查面板', wait=0.5)
        if self._pressed_at is not None and time.monotonic() - self._pressed_at < _REPRESS_SECONDS:
            return self.round_wait('等待背包页', wait=0.3)
        if self.round_by_find_area(self.last_screenshot, '战斗画面', '按键-普通攻击').is_success:
            self.ctx.controller.btn_press(BACKPACK_KEY, press_time=0.1)
            self._pressed_at = time.monotonic()
            return self.round_wait('已按 B 打开背包', wait=0.8)
        return self.round_retry('未识别局内画面，无法打开背包', wait=0.5)

    def _find_header(self) -> tuple[int, int] | None:
        """在左栏找「背包(n/20)」标题，返回（标题行心纵坐标, 件数）；找不到为 None。"""
        texts = self.ctx.ocr_service.get_ocr_result_list(
            image=self.last_screenshot, rect=self.ctx.screen_loader.get_area('贝果-局内', '背包标题').rect,
        )
        for item in texts:
            match = _HEADER_PATTERN.search(str(item.data).replace(' ', ''))
            if match is not None:
                return int(item.center.y), int(match.group(1))
        return None

    @node_from(from_name='打开背包')
    @operation_node(name='定位背包格', timeout_seconds=10)
    def locate_grid(self) -> OperationRoundResult:
        """按标题定格心；件数多于可见格时往下滚，标题不动了或滚到上限就按当前画面定。"""
        if not self._backpack_open():
            return self._wait_panel('背包页不在画面内')
        header = self._find_header()
        if header is None:
            if self._scrolls < _SCROLL_LIMIT:
                # 标题可能被滚出上沿，往回滚一格再找。
                self._scrolls += 1
                self._last_header_y = None
                self.ctx.controller.scroll(-1, _SCROLL_POS)
                return self.round_wait('找不到背包标题，往上滚一格', wait=0.5)
            return self.round_fail('左栏里找不到背包标题')
        header_y, count = header
        needed_rows = min(_BACKPACK_ROWS, max(1, -(-count // len(_BACKPACK_XS))))
        rows = self._visible_rows(header_y)
        if (len(rows) < needed_rows and self._scrolls < _SCROLL_LIMIT
                and header_y != self._last_header_y):
            self._scrolls += 1
            self._last_header_y = header_y
            self.ctx.controller.scroll(1, _SCROLL_POS)
            return self.round_wait('往下滚动背包', wait=0.5)
        self._centers = tuple(
            Point(x, header_y + _FIRST_ROW_OFFSET + _ROW_STEP * row) for row in rows for x in _BACKPACK_XS
        )
        log.info('贝果整理背包：背包 %s 件，标题 y=%s，可见第 %s 排（滚动 %s 次）',
                 count, header_y, '、'.join(str(row + 1) for row in rows) or '无', self._scrolls)
        if len(rows) < needed_rows:
            log.warning('贝果整理背包：背包有 %s 排物品，只看得到 %s 排，其余不参与整理', needed_rows, len(rows))
        if not rows:
            return self.round_fail('背包格都不在可见区')
        return self.round_success('已定位背包格')

    @staticmethod
    def _visible_rows(header_y: int) -> list[int]:
        """标题在 header_y 时，卡片上下沿都落在左栏可见区内的排号（从 0 开始）。"""
        rows: list[int] = []
        for row in range(_BACKPACK_ROWS):
            center = header_y + _FIRST_ROW_OFFSET + _ROW_STEP * row
            if center - _CARD_HALF_HEIGHT >= _VISIBLE_TOP and center + _CARD_HALF_HEIGHT <= _VISIBLE_BOTTOM:
                rows.append(row)
        return rows

    @node_from(from_name='定位背包格')
    @node_from(from_name='核对整理变化', status=_STATUS_NEXT)
    @operation_node(name='逐件换入安全箱')
    def store_next(self) -> OperationRoundResult:
        """每次只拖一件：有空格就填，满了只和箱内最差的一格严格更优才对换。"""
        if not self._backpack_open():
            return self._wait_panel('背包页不在画面内')
        safe = inspect_safe_slots(self.last_screenshot)
        if safe is None:
            return self._wait_panel('安全箱格子状态不明')
        self._panel_wait = 0
        backpack = self._backpack_marks()
        safes = inspect_occupied(self.last_screenshot, SAFE_SLOT_CENTERS, safe.occupied)
        choice = choose_store_action(backpack, safes, safe.empty, self.blueprint_priority)
        if choice is None:
            return self.round_success(self.STATUS_DONE)
        if self._actions >= self.ACTION_LIMIT:
            log.warning('贝果整理背包：已拖拽 %s 次，停止整理', self._actions)
            return self.round_success(self.STATUS_DONE)
        target = next((mark for mark in safes if mark.index == choice.dest_index), None)
        log.info(
            '贝果整理背包：%s %s 背包格 %s -> 安全箱格 %s（%s）；目标=%s，蓝图优先=%s',
            choice.mark.quality, choice.mark.item_type, choice.source_index + 1, choice.dest_index + 1,
            '填空' if choice.kind == ACTION_FILL else '对换',
            f'{target.quality} {target.item_type}' if target is not None else '空格',
            self.blueprint_priority,
        )
        self._pending_before = self.last_screenshot.copy()
        self._pending_source = choice.source_index
        self._pending_dest = choice.dest_index
        self._pending_kind = choice.kind
        self._repeated = 0
        if self.ledger is not None:
            self.ledger.start_transfer(choice.dest_index, (choice.mark.item_type, choice.mark.quality))
        self._drag(choice.source_index, choice.dest_index)
        return self.round_success('已拖拽一件', wait=0.7)

    @node_from(from_name='逐件换入安全箱', status='已拖拽一件')
    @operation_node(name='核对整理变化')
    def confirm_transfer(self) -> OperationRoundResult:
        """填空看安全箱目标格由空变有；对换看目标格外观变了。

        不要求背包源格变空：背包取走一件后其余物品可能前移补位，这一点还没实机确认。
        """
        before = self._pending_before
        source = self._pending_source
        dest = self._pending_dest
        kind = self._pending_kind
        if before is None or source is None or dest is None or kind is None:
            return self.round_fail('缺少整理前后对照数据')
        if not self._backpack_open():
            return self._wait_panel('拖拽后背包页不在画面内')
        destination = SAFE_SLOT_CENTERS[dest]
        after = self.last_screenshot
        if kind == ACTION_FILL:
            done = not slot_occupied(before, destination) and slot_occupied(after, destination)
        else:
            done = slot_occupied(after, destination) and slot_appearance_changed(before, after, destination)
        if done:
            if self.ledger is not None:
                self.ledger.confirm_transfer()
            if kind == ACTION_FILL:
                self.moved += 1
            else:
                self.swapped += 1
            self._clear_pending()
            return self.round_success(_STATUS_NEXT, wait=0.2)
        if self._repeated < self.DRAG_REPEAT_LIMIT:
            self._repeated += 1
            self._pending_before = after.copy()
            self._drag(source, dest)
            return self.round_wait(f'整理画面未确认，第 {self._repeated} 次重拖', wait=0.7)
        log.warning('贝果整理背包：背包格 %s 拖拽后安全箱没有变化，跳过这一格', source + 1)
        if self.ledger is not None:
            self.ledger.cancel_transfer()
        self._skipped.add(source)
        self._clear_pending()
        return self.round_success(_STATUS_NEXT, wait=0.2)

    @node_from(from_name='逐件换入安全箱', status=STATUS_DONE)
    @node_from(from_name='打开背包', success=False)
    @node_from(from_name='定位背包格', success=False)
    @node_from(from_name='逐件换入安全箱', success=False)
    @node_from(from_name='核对整理变化', success=False)
    @operation_node(name='关闭背包', timeout_seconds=10)
    def close_backpack(self) -> OperationRoundResult:
        """按 esc 关掉背包页，回到局内画面；esc 若直接打开了暂停菜单也算关好，退出流程认得它。"""
        if self.is_bagel_result():
            return self.round_fail(self.STATUS_DEFEATED)
        if (self.round_by_find_area(self.last_screenshot, '战斗画面', '按键-普通攻击').is_success
                or self._has('按钮-退出战斗', '战斗-菜单')):
            log.info('贝果整理背包：填入安全箱 %s 件，对换 %s 件，跳过 %s 格',
                     self.moved, self.swapped, len(self._skipped))
            return self.round_success(self.STATUS_DONE, data={'moved': self.moved, 'swapped': self.swapped})
        if self._backpack_open():
            record_safe_items(self.ctx, self.last_screenshot, self.ledger)
            self.ctx.controller.btn_press('esc', press_time=0.1)
            return self.round_wait('关闭背包页', wait=0.8)
        return self.round_retry('等待回到局内画面', wait=0.5)

    def _backpack_marks(self) -> list[BagelSlotMark]:
        """已定位的背包格里有物品、品质认得出、且没被跳过的格子。"""
        marks: list[BagelSlotMark] = []
        for mark in inspect_occupied(self.last_screenshot, self._centers):
            if mark.index in self._skipped:
                continue
            if mark.quality == QUALITY_UNKNOWN:
                if mark.index not in self._logged_unknown:
                    self._logged_unknown.add(mark.index)
                    log.info('贝果整理背包：背包格 %s 品质认不出（类型=%s），不拖', mark.index + 1, mark.item_type)
                continue
            marks.append(mark)
        return marks

    def _wait_panel(self, reason: str) -> OperationRoundResult:
        """背包页或安全箱认不清时有界等待，超限按失败交给关闭节点。"""
        self._panel_wait += 1
        if self._panel_wait < self.PANEL_WAIT_LIMIT:
            return self.round_wait(reason, wait=0.3)
        if self.ledger is not None and self._pending_dest is not None:
            self.ledger.cancel_transfer()
        self._clear_pending()
        return self.round_fail(reason)

    def _drag(self, source: int, dest: int) -> None:
        """先按住再移动，与开箱入箱同一套拖拽时序。"""
        timing = self.BACKGROUND_DRAG_TIMING if self.ctx.controller.background_mode else self.FOREGROUND_DRAG_TIMING
        self._actions += 1
        self.ctx.controller.drag_to(
            end=SAFE_SLOT_CENTERS[dest], start=self._centers[source], timing=timing,
        )

    def _clear_pending(self) -> None:
        """丢掉这一次拖拽的前后对照。"""
        self._pending_before = None
        self._pending_source = None
        self._pending_dest = None
        self._pending_kind = None
        self._repeated = 0
