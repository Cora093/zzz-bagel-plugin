from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import cv2
import numpy as np
from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node

from ..defs import INTERACT_TAP_SECONDS, SAFE_UNLOCK_HITS
from .container import BagelContainerOperation

if TYPE_CHECKING:
    from collections.abc import Callable

    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

    from .container import ContainerRecovery


class BagelUnlockSafe(BagelContainerOperation):
    """限次打开电子保险箱，光圈解锁后等待搜查面板。"""

    STATUS_UNLOCKED: str = '电子保险箱搜索完成'
    INTERRUPT_CONFIRM_SECONDS: float = 0.5
    """解锁与搜查特征均消失后，持续确认局内画面的最短时间。"""

    def __init__(
        self, ctx: ZContext, recovery: ContainerRecovery | None = None,
        prepare_open: Callable[[BagelContainerOperation], OperationRoundResult | None] | None = None,
    ) -> None:
        """一次完成开箱、光圈解锁和等待搜查面板。"""
        super().__init__(ctx, target='safe', op_name='贝果-解锁电子保险箱', recovery=recovery, prepare_open=prepare_open)
        self.timeout_seconds = 45
        self.hits_done: int = 0
        self.ring_armed: bool = True
        """当前光圈尚未点按；首圈与界面同时出现，开始即为真。"""
        self.search_hud_since: float | None = None
        """等待搜查时首次确认回到局内的截图时间；用于排除面板过渡。"""

    def handle_init(self) -> None:
        """每次运行重置解锁进度。"""
        super().handle_init()
        self.hits_done = 0
        self.ring_armed = True
        self.search_hud_since = None

    def _unlock_ready(self) -> OperationRoundResult:
        """确认解锁界面的截图立即参与光圈判断，不丢掉首圈。"""
        self._observe_ring()
        return self.round_success('已在解锁界面')

    def _open_container(self) -> OperationRoundResult:
        """按 F 后立即观察开箱结果，不丢掉首圈。"""
        return self.interact_container('已按F进入解锁', observation_wait=0.02, wait_after_interact=0)

    def _press_timing(self) -> None:
        """精确点按；F 与 Space 等价，沿用交互键。"""
        self.ctx.controller.interact(press=True, press_time=INTERACT_TAP_SECONDS, release=True)
        self.hits_done += 1

    def _search_ready(self) -> bool:
        """电子保险箱面板进入搜查中或搜索结果且安全箱可见。"""
        return (
            all(self.round_by_find_area(
                self.last_screenshot, '贝果-局内', area,
            ).is_success for area in ('电子保险箱标题', '搜查安全箱'))
            and any(self.round_by_find_area(
                self.last_screenshot, '贝果-局内', area,
            ).is_success for area in ('搜查进行中', '搜查完成'))
        )

    @operation_node(name='进入大保险解锁', is_start_node=True)
    def enter_unlock(self) -> OperationRoundResult:
        """已有搜查或解锁面板则跳过，否则按共用预算打开保险箱。"""
        if self.round_by_find_area(self.last_screenshot, '贝果-局内', '大保险解锁提示').is_success:
            return self._unlock_ready()
        if self.is_bagel_result():
            return self.round_fail(self.STATUS_DEFEATED)
        if self._search_ready():
            return self.round_success(self.STATUS_UNLOCKED)
        result = self._open_container()
        if result.is_success and result.status == '容器已打开':
            return self._unlock_ready()
        return result

    @node_from(from_name='进入大保险解锁', status='已按F进入解锁')
    @operation_node(name='等待解锁界面')
    def wait_unlock_ui(self) -> OperationRoundResult:
        """确认解锁界面；进入前允许限次补按或申请重新靠近。"""
        if self.round_by_find_area(self.last_screenshot, '贝果-局内', '大保险解锁提示').is_success:
            return self._unlock_ready()
        if self.is_bagel_result():
            return self.round_fail(self.STATUS_DEFEATED)
        if self._search_ready():
            return self.round_success(self.STATUS_UNLOCKED)
        result = self._open_container()
        if result.is_success:
            if result.status == '容器已打开':
                return self._unlock_ready()
            return self.round_wait('电子保险箱未打开，补按一次交互')
        return result

    @node_from(from_name='进入大保险解锁', status='已在解锁界面')
    @node_from(from_name='等待解锁界面', status='已在解锁界面')
    @operation_node(name='时序点按解锁', timeout_seconds=20)
    def timing_hits(self) -> OperationRoundResult:
        """每轮光圈在命中半径内点按一次，避免固定间隔逐轮漂移。

        解锁完成取决于进度条满，点按数只是参考，所以提示还在就继续点按。
        提示消失后进度条可能已满、面板正在切入，一律交给等待搜查面板的节点复核。
        """
        if not self.round_by_find_area(
            self.last_screenshot, '贝果-局内', '大保险解锁提示',
        ).is_success:
            if self.is_bagel_result():
                return self.round_fail(self.STATUS_DEFEATED)
            return self.round_success(self.STATUS_UNLOCKED)
        return self.round_wait(self._observe_ring(), wait=0.02)

    def _observe_ring(self) -> str:
        """处理当前截图中的光圈，入口确认和持续点按共用同一判断。

        光圈只外扩、点过的圈停在外缘，所以命中半径以内的圈必是新一轮；不要求先看到小圈。
        """
        ring = unlock_ring(self.last_screenshot)
        if ring is None:
            return '等待下一轮光圈'
        color, radius = ring
        hit_radius = 180 if color == '红' else 160
        if radius < hit_radius:
            self.ring_armed = True
        if self.ring_armed and hit_radius <= radius <= 220:
            self._press_timing()
            self.ring_armed = False
            return f'{color}圈已发送点按 {self.hits_done}（参考 {SAFE_UNLOCK_HITS}）'
        return '等待光圈进入命中范围'

    @node_from(from_name='进入大保险解锁', status=STATUS_UNLOCKED)
    @node_from(from_name='等待解锁界面', status=STATUS_UNLOCKED)
    @node_from(from_name='时序点按解锁', status=STATUS_UNLOCKED)
    @operation_node(name='等待电子保险箱搜查面板', timeout_seconds=20)
    def wait_search(self) -> OperationRoundResult:
        """解锁成功后进入电子保险箱搜查；不补按 F，避免全部拾取。"""
        if self.is_bagel_result():
            return self.round_fail(self.STATUS_DEFEATED)
        if self._search_ready():
            return self.round_success(self.STATUS_UNLOCKED)
        if self.round_by_find_area(self.last_screenshot, '贝果-局内', '电子保险箱标题').is_success:
            self.search_hud_since = None
            return self.round_wait('等待电子保险箱搜查状态', wait=0.5)
        if self.round_by_find_area(self.last_screenshot, '贝果-局内', '大保险解锁提示').is_success:
            self.search_hud_since = None
            return self.round_wait('解锁中，等待进入搜查', wait=0.5)
        if self.round_by_find_area(self.last_screenshot, '战斗画面', '按键-普通攻击').is_success:
            if self.search_hud_since is None:
                self.search_hud_since = self.last_screenshot_time
            elif self.last_screenshot_time - self.search_hud_since >= self.INTERRUPT_CONFIRM_SECONDS:
                return self.round_recoverable_fail('解锁被打断，已返回局内')
            return self.round_wait('疑似解锁中断，复核局内画面', wait=0.5)
        self.search_hud_since = None
        return self.round_wait('等待电子保险箱面板', wait=0.5)


@lru_cache(maxsize=1)
def _ring_sample_maps() -> tuple[np.ndarray, np.ndarray]:
    """沿 1080p 解锁圆盘的同心圆采样，复用坐标减少逐帧计算。"""
    angles = np.arange(360) * np.pi / 180
    radii = np.arange(15, 236)[:, None]
    return (
        (960 + radii * np.cos(angles)).astype(np.float32),
        (543 + radii * np.sin(angles)).astype(np.float32),
    )


def unlock_ring(screen: np.ndarray) -> tuple[str, float] | None:
    """返回扩张光圈颜色和外缘半径；零散背景颜色不足以组成圆周。"""
    map_x, map_y = _ring_sample_maps()
    sampled = cv2.remap(screen, map_x, map_y, cv2.INTER_LINEAR)
    hue, saturation, value = cv2.split(cv2.cvtColor(sampled, cv2.COLOR_RGB2HSV))
    bright = (saturation >= 95) & (value >= 140)
    candidates: list[tuple[float, str, float]] = []
    for name, color in (
        ('红', (hue <= 10) | (hue >= 165)),
        ('黄', (hue >= 10) & (hue <= 40)),
    ):
        coverage = (bright & color).mean(axis=1)
        valid = np.flatnonzero(coverage > 0.4)
        if len(valid):
            candidates.append((float(coverage.max()), name, float(valid[-1] + 15)))
    if not candidates:
        return None
    _, name, radius = max(candidates)
    return name, radius
