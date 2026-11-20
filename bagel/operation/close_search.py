"""独立关闭贝果搜查面板，不执行收集或交互。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from one_dragon.base.operation.operation_node import operation_node

from .operation import BagelOperation

if TYPE_CHECKING:
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext


CLOSE_WAIT_SECONDS: float = 1.0
"""点击「搜查返回」后等面板消失的上限（秒）；期间只轮询不重点，超时才再点一次。

实机录屏里面板在点击后一帧内就关闭，原先固定等 0.8 秒只是为了不重复点击。
"""

CLOSE_POLL_SECONDS: float = 0.1
"""等面板关闭、等回到局内时的轮询间隔（秒）。"""


class BagelCloseSearch(BagelOperation):
    """按调用方要求关闭面板，并确认回到局内；不判断物品是否已收集完整。"""

    def __init__(self, ctx: ZContext) -> None:
        """只处理已打开的搜查面板。"""
        super().__init__(ctx, op_name='贝果-关闭搜查面板', timeout_seconds=30)
        self._clicked_at: float | None = None
        """上次点击「搜查返回」的时刻（monotonic）；None 表示还没点过。"""

    def handle_init(self) -> None:
        """每次执行重新计点击时刻。"""
        super().handle_init()
        self._clicked_at = None

    @operation_node(name='关闭搜查面板', is_start_node=True)
    def close_panel(self) -> OperationRoundResult:
        """确认面板后点击返回，必须看到局内画面才完成关闭。"""
        if self.is_bagel_result():
            return self.round_fail(self.STATUS_DEFEATED)
        has_title = self._has('搜查容器标题') or self._has('电子保险箱标题')
        if not has_title:
            if self.round_by_find_area(
                self.last_screenshot, '战斗画面', '按键-普通攻击'
            ).is_success:
                return self.round_success('已关闭搜查面板')
            return self.round_wait('等待回到局内', wait=CLOSE_POLL_SECONDS)
        now = time.monotonic()
        if self._clicked_at is not None and now - self._clicked_at < CLOSE_WAIT_SECONDS:
            return self.round_wait('等待搜查面板关闭', wait=CLOSE_POLL_SECONDS)
        if not (self._has('全部拾取_单行') or self._has('全部拾取_双行')):
            return self.round_wait('等待搜查面板可关闭', wait=0.3)
        self.round_by_click_area('贝果-局内', '搜查返回')
        self._clicked_at = now
        return self.round_wait('等待搜查面板关闭', wait=CLOSE_POLL_SECONDS)
