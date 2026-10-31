"""调用主程序通用返回或传送前，先退出贝果界面页。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from one_dragon.base.operation.operation_node import operation_node

from .bagel_operation import BagelOperation

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

#: 点击返回的次数上限；备战 → 选图 → 主界面 → 大世界最多需要 3 次。
_MAX_BACK_CLICKS: int = 6
#: 连续几帧没有贝果界面才算已退出，避免把切页过渡帧当成已离开。
_CLEAR_FRAMES: int = 2


class BagelLeavePages(BagelOperation):
    """逐层点击返回，直到画面不再是贝果界面页。

    主程序「返回大世界」用 0.5 的最长公共子串匹配左上角「区域」，
    会把主界面标题「迷宫诡域：古怪试炼」的「域」当成返回入口并反复点击，
    停在贝果界面时无法退出。所以调用主程序传送或通用返回前，先由插件退出。
    """

    def __init__(self, ctx: ZContext) -> None:
        """不在贝果界面时连续确认两帧后直接完成。"""
        super().__init__(ctx, op_name='贝果-退出贝果界面', timeout_seconds=30)
        self.back_clicks: int = 0
        self.clear_frames: int = 0

    def handle_init(self) -> None:
        """重复执行时重新计数。"""
        super().handle_init()
        self.back_clicks = 0
        self.clear_frames = 0

    @operation_node(name='退出贝果界面', is_start_node=True)
    def leave_pages(self) -> OperationRoundResult:
        """在贝果界面就点返回；连续两帧不在贝果界面才完成。"""
        if not self._at_bagel_page():
            self.clear_frames += 1
            if self.clear_frames < _CLEAR_FRAMES:
                return self.round_wait('确认已退出贝果界面', wait=0.5)
            return self.round_success()
        self.clear_frames = 0
        if self.back_clicks >= _MAX_BACK_CLICKS:
            return self.round_fail(f'点击返回 {_MAX_BACK_CLICKS} 次后仍在贝果界面，停止并保留现场')
        result = self.round_by_click_area('菜单', '返回')
        if not result.is_success:
            return result
        self.back_clicks += 1
        return self.round_wait('已点击返回，等待退出贝果界面', wait=1)
