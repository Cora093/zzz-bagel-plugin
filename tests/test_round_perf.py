"""决策日志合并：首行照打，连续相同只累计，换键或满 1 秒再汇总。"""

import pytest

from bagel.route import round_perf
from bagel.route.round_perf import RepeatMerger, RoundPerf


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [0.0]
    monkeypatch.setattr(round_perf, 'perf_counter', lambda: now[0])
    return now


def submit(merger: RepeatMerger, clock: list[float], key: str | None, text: str) -> list[str]:
    perf = RoundPerf()
    perf.add('截图', 0.01)
    clock[0] += 0.02
    return merger.submit(key, text, perf)


def test_repeats_are_summarised_when_key_changes(clock: list[float]) -> None:
    merger = RepeatMerger()
    assert submit(merger, clock, 'a', '冲刺 · 距 30px') == ['冲刺 · 距 30px · 用时 20ms（截图 10 其他 10）']
    assert submit(merger, clock, 'a', '冲刺 · 距 29px') == []
    assert submit(merger, clock, 'a', '冲刺 · 距 28px') == []
    summary, line = submit(merger, clock, None, '已到点')
    assert summary.startswith('↑ 同上 2 轮 ') and '最后：冲刺 · 距 28px' in summary
    assert '用时中位 20ms 最大 20ms（各段平均 截图 10）' in summary
    assert line.startswith('已到点 · ')


def test_stuck_loop_still_prints_once_per_second(clock: list[float]) -> None:
    merger = RepeatMerger()
    printed = [line for _ in range(150) for line in submit(merger, clock, 'a', '短步')] + merger.flush()
    assert sum(line.startswith('短步') for line in printed) == 3
    assert sum(line.startswith('↑ 同上') for line in printed) == 3
