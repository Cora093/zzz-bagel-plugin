"""电子保险箱光圈解锁：完成条件是提示消失，点按数不是出口。"""

from unittest.mock import Mock, call

import numpy as np
import pytest
from one_dragon.base.operation.operation_round_result import OperationRoundResultEnum

from bagel.defs import INTERACT_TAP_SECONDS
from bagel.operation import unlock_safe
from bagel.operation.unlock_safe import BagelUnlockSafe


@pytest.fixture
def op(monkeypatch: pytest.MonkeyPatch) -> BagelUnlockSafe:
    ctx = Mock()
    result = BagelUnlockSafe(ctx)
    result.last_screenshot = np.zeros((1, 1, 3), np.uint8)
    result.areas = set()
    monkeypatch.setattr(result, 'is_bagel_result', lambda: False)
    monkeypatch.setattr(
        result, 'round_by_find_area',
        lambda _screen, _screen_name, area: Mock(is_success=area in result.areas),
    )
    return result


def show_ring(monkeypatch: pytest.MonkeyPatch, ring: tuple[str, float] | None) -> None:
    monkeypatch.setattr(unlock_safe, 'unlock_ring', lambda _screen: ring)


def test_prompt_gone_hands_over_to_search_wait_even_before_search_state(op: BagelUnlockSafe) -> None:
    """进度条满后面板切入：提示已消失、标题可见但搜查状态还没出现，不判失败。"""
    op.hits_done = 3
    op.areas = {'电子保险箱标题', '搜查安全箱'}
    result = op.timing_hits()
    assert result.is_success
    assert result.status == BagelUnlockSafe.STATUS_UNLOCKED


def test_prompt_gone_without_panel_is_left_to_search_wait(op: BagelUnlockSafe) -> None:
    """过渡帧什么都认不出时也不在此处判失败，由等待节点复核后再判。"""
    result = op.timing_hits()
    assert result.is_success
    assert result.status == BagelUnlockSafe.STATUS_UNLOCKED
    assert not result.is_fail


def test_search_wait_still_fails_after_confirmed_return_to_hud(op: BagelUnlockSafe) -> None:
    """回到局内 HUD 持续 0.5 秒才判解锁被打断。"""
    op.areas = {'按键-普通攻击'}
    op.last_screenshot_time = 100.0
    assert op.wait_search().result == OperationRoundResultEnum.WAIT
    op.last_screenshot_time = 100.5
    result = op.wait_search()
    assert result.is_fail
    assert result.status == '解锁被打断，已返回局内'


def test_keeps_pressing_after_required_hits_while_prompt_remains(
    op: BagelUnlockSafe, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """点满 4 次进度条仍未满：提示还在就继续点按，不提前结束。"""
    op.areas = {'大保险解锁提示'}
    op.hits_done = 4
    show_ring(monkeypatch, ('黄', 200))
    op.ring_armed = True
    result = op.timing_hits()
    assert result.result == OperationRoundResultEnum.WAIT
    assert op.hits_done == 5
    assert op.ctx.controller.interact.call_args_list == [
        call(press=True, press_time=INTERACT_TAP_SECONDS, release=True),
    ]
