"""路线编辑器的定位线程：只截图定位，不移动、不打点、不做任何输入，也不进运行上下文。

- ``BagelLocateThread``：节点行的「定位」，切到游戏定位一次就交回编辑器。
- ``BagelTrackThread``：地图工具栏的「定位」开关，切到游戏后持续定位，直到手动停止。
"""

from __future__ import annotations

from threading import Event
from typing import TYPE_CHECKING

import numpy as np
from one_dragon.utils import cv2_utils
from one_dragon.utils.log_utils import log
from PySide6.QtCore import QThread, Signal

from ..route.route_recorder import (
    FOREGROUND_CHECK_INTERVAL,
    GAME_FOREGROUND_TIMEOUT,
    MINIMAP_AREA,
    is_foreground,
    wait_foreground,
)
from ..vision.map_locator import MultiMapLocator, load_map_locators

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from one_dragon.base.geometry.rectangle import Rect
    from PySide6.QtWidgets import QWidget
    from zzz_od.context.zzz_context import ZContext


TRACK_INTERVAL: float = 0.2
"""持续定位的采样间隔（秒）。只用来显示坐标，每秒 5 次足够。"""


def _game_target(ctx: ZContext) -> tuple[int, Rect] | str:
    """检查游戏窗口与小地图区域。

    Args:
        ctx: 运行上下文。

    Returns:
        (游戏窗口句柄, 小地图区域)；不满足时返回原因文字。
    """
    controller = ctx.controller
    if controller is None or not controller.init_game_win():
        return '游戏窗口未就绪，请先启动游戏'
    hwnd = controller.game_win.get_hwnd()
    if hwnd is None or not controller.is_game_window_ready:
        return '未找到游戏窗口，请先启动游戏'
    area = ctx.screen_loader.get_area(*MINIMAP_AREA)
    if area is None:
        return f'缺少画面区域：{MINIMAP_AREA[0]}/{MINIMAP_AREA[1]}'
    return hwnd, area.pc_rect


class BagelLocateThread(QThread):
    """在后台线程里做一次定位，避免切窗口时卡住界面。"""

    status_changed = Signal(str)
    """当前阶段文案。"""

    done = Signal(bool, str, str, object)
    """(是否成功, 结论文案, 分层名, 层内坐标)；失败时分层名为空、坐标为 None。"""

    def __init__(self, ctx: ZContext, parent: QWidget | None = None) -> None:
        """
        Args:
            ctx: 运行上下文。
            parent: Qt 父对象。
        """
        super().__init__(parent)
        self.ctx: ZContext = ctx

    def run(self) -> None:
        """切到游戏定位一次并立刻返回结论。"""
        try:
            self._locate_once()
        except Exception as error:
            log.error('贝果单次定位异常', exc_info=True)
            self.done.emit(False, f'定位异常：{error}', '', None)

    def _locate_once(self) -> None:
        """截一帧小地图，定位一次，把层名与层内坐标交回去。"""
        target = _game_target(self.ctx)
        if isinstance(target, str):
            self.done.emit(False, target, '', None)
            return
        hwnd, rect = target
        controller = self.ctx.controller

        self.status_changed.emit('正在切到游戏定位')
        controller.active_window()
        if not wait_foreground(hwnd, GAME_FOREGROUND_TIMEOUT):
            self.done.emit(False, '游戏窗口未切到前台，未定位', '', None)
            return

        screenshot = controller.get_screenshot()
        if screenshot is None:
            self.done.emit(False, '截图失败，未定位', '', None)
            return
        crop = self._crop(screenshot, rect)
        if crop is None:
            self.done.emit(False, '小地图裁剪为空，未定位', '', None)
            return

        locators = load_map_locators()
        result = MultiMapLocator(locators).locate(crop)
        if result is None:
            self.done.emit(False, '这一帧没定位到，可能不在贝果局内', '', None)
            return
        locator, score, center = result
        # 路线里的点存的是全局坐标（打点时写的就是 center），所以这里也回全局坐标。
        global_xy = (float(center[0]), float(center[1]))
        message = (
            f'{locator.name} · 坐标 ({global_xy[0]:.0f}, {global_xy[1]:.0f})'
            f' · 匹配 {score:.2f}'
        )
        log.info('贝果单次定位：%s', message)
        self.done.emit(True, message, locator.name, global_xy)

    @staticmethod
    def _crop(screenshot: MatLike, rect: Rect) -> MatLike | None:
        """裁出小地图并统一成三通道，与录制、导航用同一套裁剪。

        Args:
            screenshot: 整屏截图。
            rect: 小地图区域。

        Returns:
            小地图裁剪；裁剪失败时返回 None。
        """
        crop = cv2_utils.crop_image_only(screenshot, rect)
        if crop is None or crop.size == 0:
            return None
        if crop.ndim == 2:
            crop = np.stack([crop] * 3, axis=-1)
        return crop


class BagelTrackThread(QThread):
    """持续定位：切到游戏后按固定间隔定位，直到调用 ``stop``。

    游戏不在前台时暂停采样（画面可能被遮挡），回到游戏继续；暂停期间坐标停在最后一次结果。
    """

    status_changed = Signal(str)
    """当前阶段文案。"""

    position_changed = Signal(str, object)
    """(分层名, 全局坐标)；只在定位成功时发出。"""

    done = Signal(bool, str)
    """(是否正常结束, 结论文案)。手动停止算正常结束。"""

    def __init__(self, ctx: ZContext, parent: QWidget | None = None) -> None:
        """
        Args:
            ctx: 运行上下文。
            parent: Qt 父对象。
        """
        super().__init__(parent)
        self.ctx: ZContext = ctx
        self._stop: Event = Event()

    def stop(self) -> None:
        """请求停止；当前这一帧处理完就退出。"""
        self._stop.set()

    def run(self) -> None:
        """持续定位直到停止。"""
        try:
            self._track()
        except Exception as error:
            log.error('贝果持续定位异常', exc_info=True)
            self.done.emit(False, f'定位异常：{error}')

    def _track(self) -> None:
        """切到游戏，循环截小地图定位，把结果逐次交回去。"""
        target = _game_target(self.ctx)
        if isinstance(target, str):
            self.done.emit(False, target)
            return
        hwnd, rect = target
        controller = self.ctx.controller

        locator = MultiMapLocator(load_map_locators())
        self.status_changed.emit('正在切到游戏定位')
        controller.active_window()
        if not wait_foreground(hwnd, GAME_FOREGROUND_TIMEOUT, self._stop):
            if self._stop.is_set():
                self.done.emit(True, '定位已停止')
            else:
                self.done.emit(False, '游戏窗口未切到前台，未开始定位')
            return
        self.status_changed.emit('持续定位中；切回编辑器后暂停，点「停止定位」结束')

        while not self._stop.is_set():
            if not is_foreground(hwnd):
                self._stop.wait(FOREGROUND_CHECK_INTERVAL)
                continue
            screenshot = controller.get_screenshot()
            crop = BagelLocateThread._crop(screenshot, rect) if screenshot is not None else None
            result = locator.locate(crop) if crop is not None else None
            if result is not None:
                hit, _score, center = result
                self.position_changed.emit(hit.name, (float(center[0]), float(center[1])))
            self._stop.wait(TRACK_INTERVAL)
        self.done.emit(True, '定位已停止')
