"""容器靠近与开箱共享的预算和画面判断。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..vision.prompt_list import PromptFrame, PromptSelection
from .operation import BagelOperation, scroll_interaction_row

if TYPE_CHECKING:
    from collections.abc import Callable

    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

INTERACT_KEY_THRESHOLD: float = 0.85
"""交互 F 键模板的匹配阈值，沿用画面定义里原来的设置。"""


class ContainerRecovery:
    """一次容器动作的预算，子操作初始化不能重置。"""

    STATUS_REAPPROACH: str = '容器提示消失，需要重新靠近'
    TIME_LIMIT: float = 30
    INTERACT_LIMIT: int = 3

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """使用执行器的有效耗时计时，独立操作默认使用单调时钟。"""
        self.clock: Callable[[], float] = clock
        self.started_at: float | None = None
        self.interactions: int = 0
        self.last_interact_at: float | None = None
        self.last_input_frame: float | None = None

    def start(self) -> None:
        """首次进入箱前确认时开始计时。"""
        if self.started_at is None:
            self.started_at = self.clock()

    def error(self) -> str | None:
        """时间耗尽后禁止继续输入，仍允许识别已经打开的面板。"""
        if self.started_at is not None and self.clock() - self.started_at >= self.TIME_LIMIT:
            return '容器靠近与开箱恢复超过30秒'
        return None


def container_state(operation: BagelOperation, target: str) -> str:
    """先检查面板，避免把搜查中的 F 当成开箱提示。"""
    def has(area: str, screen: str = '贝果-局内') -> bool:
        """只读取当前截图。"""
        return operation._has(area, screen)

    box = has('搜查容器标题')
    safe = has('电子保险箱标题')
    unlocking = has('大保险解锁提示')
    slots = has('搜查安全箱')
    searching = has('搜查进行中') or has('搜查完成')
    if target == 'safe' and unlocking and not box:
        return 'ready'
    if (box if target == 'box' else safe) and not (safe if target == 'box' else box) and slots and searching:
        return 'ready'
    if box or safe or unlocking or slots or searching:
        return 'panel'
    if not has('按键-普通攻击', '战斗画面'):
        return 'unknown'
    return 'field'


class BagelContainerOperation(BagelOperation):
    """共用开箱输入与观察规则，移动由流程执行器协调。"""

    def __init__(
        self, ctx: ZContext, target: str, op_name: str,
        recovery: ContainerRecovery | None = None,
        prepare_open: Callable[[BagelContainerOperation], OperationRoundResult | None] | None = None,
    ) -> None:
        """独立执行自建预算；流程执行复用传入的预算。"""
        super().__init__(ctx, op_name=op_name)
        self.target: str = target
        self.prepare_open: Callable[[BagelContainerOperation], OperationRoundResult | None] | None = prepare_open
        self.selection: PromptSelection = PromptSelection()
        self._owns_recovery: bool = recovery is None
        self.recovery: ContainerRecovery = recovery or ContainerRecovery(lambda: self.operation_usage_time)
        self._missing_since: float | None = None

    def handle_init(self) -> None:
        """仅独立执行重置预算，流程重建操作保留次数和时限。"""
        super().handle_init()
        if self._owns_recovery:
            self.recovery = ContainerRecovery(lambda: self.operation_usage_time)
        self._missing_since = None
        self.selection.reset(float('-inf'))

    def _select_standalone(self) -> OperationRoundResult | None:
        """独立运行时使用相同的选行规则，此时没有路线转向线程。"""
        frame = PromptFrame(self.ctx, self.last_screenshot, self.last_screenshot_time)
        kind = 'electronic_safe' if self.target == 'safe' else 'box'
        step = frame.step_toward(kind)
        if step is None:
            self.selection.reset(frame.stamp)
            now = self.recovery.clock()
            if self._missing_since is None:
                self._missing_since = now
            if now - self._missing_since >= 0.5:
                return self.round_recoverable_fail(ContainerRecovery.STATUS_REAPPROACH)
            return self.round_wait('等待目标交互提示恢复', wait=0.1)
        self._missing_since = None
        if not self.selection.observe(frame):
            return self.round_wait('等待选中行稳定', wait=0.1)
        if step:
            scroll_interaction_row(step > 0)
            self.selection.reset(max(time.time(), frame.stamp))
            return self.round_wait('已滚动一格，复核高亮条目', wait=0.2)
        self.selection.reset(frame.stamp)
        return None

    def interact_container(
        self, sent_status: str | None = None,
        observation_wait: float = 0.25, wait_after_interact: float = 0.25,
    ) -> OperationRoundResult:
        """识别面板、限次补按；保险箱连续解锁可缩短观察等待。"""
        if self.is_bagel_result():
            return self.round_fail(self.STATUS_DEFEATED)
        state = container_state(self, self.target)
        if state == 'ready':
            return self.round_success('容器已打开')
        self.ctx.controller.stop_moving_forward()
        self.recovery.start()
        reason = self.recovery.error()
        if reason:
            return self.round_recoverable_fail(reason)
        if state in ('panel', 'unknown'):
            self._missing_since = None
            return self.round_wait('等待容器面板或局内画面确认', wait=observation_wait)
        if self.recovery.last_input_frame is not None and self.last_screenshot_time <= self.recovery.last_input_frame:
            return self.round_wait('等待交互后的新截图', wait=min(0.15, observation_wait))
        now = self.recovery.clock()
        if self.recovery.last_interact_at is not None and now - self.recovery.last_interact_at < 2:
            return self.round_wait('等待开箱交互结果', wait=observation_wait)
        if self.recovery.interactions >= self.recovery.INTERACT_LIMIT:
            return self.round_recoverable_fail('容器开箱交互已达3次上限')
        waiting = self._select_standalone() if self.prepare_open is None else self.prepare_open(self)
        if waiting is not None:
            return waiting
        # 识别耗时也计入预算，在实际输入前再次检查。
        reason = self.recovery.error()
        if reason:
            return self.round_recoverable_fail(reason)
        # 输入前占用次数；控制器在发送后抛错也不能额外获得一次 F。
        self.recovery.interactions += 1
        self.recovery.last_interact_at = self.recovery.clock()
        self.recovery.last_input_frame = max(time.time(), self.last_screenshot_time)
        self.ctx.controller.interact(press=True, press_time=0.2, release=True)
        self.recovery.last_interact_at = self.recovery.clock()
        self.recovery.last_input_frame = max(time.time(), self.last_screenshot_time)
        return self.round_success(sent_status, wait=wait_after_interact)
