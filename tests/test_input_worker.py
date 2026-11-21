"""前进线程的短按节拍：用真实线程和记录按键时刻的假控制器检查停顿下限。"""

import time
from unittest.mock import Mock

import pytest

from bagel.route import input_worker
from bagel.route.input_worker import STEP_REST_SECONDS, BagelInputWorker


class _Keys:
    """只记录 W 的按下和松开时刻。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []

    def move_w(self, press: bool = False, release: bool = False) -> None:
        self.events.append(('down' if press else 'up', time.time()))

    def rests(self) -> list[float]:
        """每次松开到下一次按下的间隔。"""
        return [b[1] - a[1] for a, b in zip(self.events, self.events[1:], strict=False)
                if a[0] == 'up' and b[0] == 'down']


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> tuple[BagelInputWorker, _Keys]:
    keys = _Keys()
    ctx = Mock()
    ctx.controller.move_w = keys.move_w
    monkeypatch.setattr(input_worker._TurnWorker, '_loop', lambda self: None)
    result = BagelInputWorker(ctx)
    yield result, keys
    result.shutdown()


def _wait(predicate, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, '等待超时'
        time.sleep(0.002)


def test_back_to_back_steps_keep_rest(worker: tuple[BagelInputWorker, _Keys]) -> None:
    """上一步一松键就请求下一步，按下也要等停顿满。"""
    w, keys = worker
    for _ in range(4):
        _wait(lambda: not w.forward_step_active())
        assert w.step_forward(0.03)
    _wait(lambda: not w.forward_step_active())
    rests = keys.rests()
    assert len(rests) == 3
    assert min(rests) >= STEP_REST_SECONDS


def test_hold_then_step_releases_and_rests(worker: tuple[BagelInputWorker, _Keys]) -> None:
    """持续前进中改为短按：先松键，停顿满了才按下这一步。"""
    w, keys = worker
    w.hold_forward()
    _wait(lambda: w.forward_held_seconds() is not None)
    assert w.step_forward(0.03)
    _wait(lambda: not w.forward_step_active())
    assert [e[0] for e in keys.events] == ['down', 'up', 'down', 'up']
    assert keys.rests()[0] >= STEP_REST_SECONDS
