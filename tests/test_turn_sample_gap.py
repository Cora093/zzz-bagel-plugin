"""按模拟时间线核对转向激活、采样和空闲时的无样本间隔。"""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest

from bagel.route import input_worker


def test_sample_gap_starts_at_activation_then_last_valid_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    monkeypatch.setattr(input_worker, 'time', Mock(perf_counter=lambda: now[0]))
    monkeypatch.setattr(input_worker.threading.Thread, 'start', lambda self: None)
    captures = iter([(10.10, None), (10.16, 0.0), (10.20, 0.0), (20.12, 0.0)])

    def capture() -> float | None:
        now[0], angle = next(captures)
        return angle

    stats = Mock()
    worker = input_worker._TurnWorker(Mock(), capture, stats=stats)
    worker._wake = Mock()

    def schedule() -> Iterator[None]:
        worker.submit(0.0)
        now[0] = 10.08
        yield
        now[0] = 10.12
        worker.submit(0.0)
        yield
        now[0] = 10.18
        yield
        worker.submit(None)
        yield
        now[0] = 20.0
        worker.submit(0.0)
        now[0] = 20.03
        worker.submit(0.0)
        now[0] = 20.10
        yield
        worker._stopping = True
        yield

    steps = schedule()
    worker._wake.wait.side_effect = lambda _: next(steps)
    worker._loop()
    gaps = [seconds for (name, seconds), _ in stats.record.call_args_list if name == '无样本间隔']
    assert gaps == pytest.approx([0.10, 0.16, 0.04, 0.12])
