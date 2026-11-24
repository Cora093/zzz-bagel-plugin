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


def test_trusted_initial_ratio_lifts_warmup_cap_and_chunks_large_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无记忆首局受 45 度热身限幅；跨局可信起步值放开上限，<=120 度单段直发，>120 度拆两段连发。"""
    monkeypatch.setattr(input_worker, 'TURN_SAMPLE_INTERVAL', 0.002)
    monkeypatch.setattr(input_worker, 'TURN_CHUNK_INTERVAL', 0.0)
    monkeypatch.setattr(input_worker, 'TURN_POST_TURN_QUIET', 10.0)

    mem = input_worker.TurnRatioMemory()
    assert not mem.trusted
    mem.absorb(1.5, input_worker.TURN_HINT_MIN_SAMPLES, -19.0)
    assert mem.trusted and mem.hint == 1.5

    cold_sent: list[float] = []
    cold = input_worker._TurnWorker(cold_sent.append, lambda: 0.0, initial_ratio=1.0, trust_initial_ratio=False)
    try:
        cold.submit(150.0)
        _wait(lambda: len(cold_sent) >= 1)
        assert cold_sent == [pytest.approx(45.0)]
    finally:
        cold.stop()

    mid_sent: list[float] = []
    mid = input_worker._TurnWorker(
        mid_sent.append, lambda: 0.0, initial_ratio=mem.hint, trust_initial_ratio=mem.trusted,
    )
    try:
        mid.submit(110.0)
        _wait(lambda: len(mid_sent) >= 1)
        assert mid_sent == [pytest.approx(110.0 / 1.5)]
    finally:
        mid.stop()

    warm_sent: list[float] = []
    warm = input_worker._TurnWorker(
        warm_sent.append, lambda: 0.0, initial_ratio=mem.hint, trust_initial_ratio=mem.trusted,
    )
    try:
        warm.submit(150.0)
        _wait(lambda: len(warm_sent) >= 2)
        assert len(warm_sent) == 2
        assert sum(warm_sent) == pytest.approx(100.0)
    finally:
        warm.stop()


def test_idle_turn_worker_finishes_calibration_and_wakes_with_fresh_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取消目标后采完上一笔，空闲时停采，新目标重新取三帧。"""
    monkeypatch.setattr(input_worker, 'TURN_SAMPLE_INTERVAL', 0.004)
    monkeypatch.setattr(input_worker, 'TURN_POST_TURN_QUIET', 0.02)
    sampler = Mock(return_value=0.0)
    sent = Mock()
    worker = input_worker._TurnWorker(sent, sampler)
    try:
        time.sleep(0.04)
        sampler.assert_not_called()
        worker.submit(30.0)
        _wait(lambda: sent.called)
        worker.submit(None)
        sampler.return_value = 30.0
        _wait(lambda: worker.turn_samples == 1 and not worker._history)
        before = sampler.call_count
        for _ in range(4):
            worker.submit(None)
            time.sleep(0.01)
        assert sampler.call_count == before
        sent.side_effect = lambda _: worker.submit(None)
        worker.submit(60.0)
        _wait(lambda: sent.call_count == 2)
        assert sampler.call_count - before >= input_worker.TURN_STABLE_FRAMES
    finally:
        worker.stop()
    assert not worker._thread.is_alive()


def test_cancelled_turn_stops_sampling_when_minimap_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """下发后面板挡住小地图，撤销目标后只等到校准期限。"""
    monkeypatch.setattr(input_worker, 'TURN_SAMPLE_INTERVAL', 0.004)
    monkeypatch.setattr(input_worker, 'TURN_POST_TURN_QUIET', 0.0)
    monkeypatch.setattr(input_worker, 'TURN_SETTLE_GRACE', 0.04)
    monkeypatch.setattr(input_worker, 'TURN_STILL_TIMEOUT', 0.06)
    sampler = Mock(return_value=0.0)

    def open_panel(_: float) -> None:
        sampler.return_value = None
        worker.submit(None)

    worker = input_worker._TurnWorker(open_panel, sampler)
    try:
        worker.submit(30.0)
        _wait(lambda: sampler.return_value is None)
        time.sleep(0.15)
        count = sampler.call_count
        time.sleep(0.06)
        assert sampler.call_count == count
        assert worker._pending is None and worker.turn_samples == 0
    finally:
        worker.stop()


def test_expired_turn_target_sleeps_even_without_heading(monkeypatch: pytest.MonkeyPatch) -> None:
    """朝向一直读不到时也应回收过期目标，并可从休眠直接停止。"""
    monkeypatch.setattr(input_worker, 'TURN_SAMPLE_INTERVAL', 0.004)
    monkeypatch.setattr(input_worker, 'TURN_TARGET_FRESH_SECONDS', 0.04)
    sampler = Mock(return_value=None)
    worker = input_worker._TurnWorker(Mock(), sampler)
    try:
        worker.submit(45.0)
        _wait(lambda: worker._target is None)
        count = sampler.call_count
        time.sleep(0.04)
        assert count > 0 and sampler.call_count == count
    finally:
        worker.stop()
    assert not worker._thread.is_alive()


def test_chunked_sample_isolated_and_prior_plus_outlier_guard_protect_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """拆段样本不写单段峰值箱；跨局先验权重防早局噪声劫持；偶发离群值忽略、连续 3 次放行。"""
    monkeypatch.setattr(input_worker._TurnWorker, '_loop', lambda self: None)
    tw = input_worker._TurnWorker(
        lambda _: None, lambda: 0.0, initial_ratio=1.5, trust_initial_ratio=True,
    )
    try:
        # 拆 2 段样本（倍率 1.7）：只消费放行，不污染单段比例，也不计入 turn_samples
        tw._pending = 80.0
        tw._pending_chunks = 2
        tw._anchor = 0.0
        assert tw._learn(0.0, 80.0, 136.0) is True
        assert tw.turn_ratio == pytest.approx(1.5)
        assert tw.turn_samples == 0

        # 跨局先验权重 3 票：2 个略偏高样本（1.90，在 1.5*1.4=2.1 内）仍不撼动 1.5
        for _ in range(2):
            tw._pending = 40.0
            tw._pending_chunks = 1
            tw._anchor = 0.0
            assert tw._learn(0.0, 40.0, 76.0) is True
        assert tw.turn_ratio == pytest.approx(1.5)
        assert tw.turn_samples == 2

        # 偶发离群样本（3.0 > 1.5*1.4=2.1）连续 2 次均被忽略，正常样本到达后清零离群计数
        for _ in range(2):
            tw._pending = 20.0
            tw._pending_chunks = 1
            tw._anchor = 0.0
            assert tw._learn(0.0, 20.0, 60.0) is True
        assert tw.turn_samples == 2
        tw._pending = 40.0
        tw._pending_chunks = 1
        tw._anchor = 0.0
        assert tw._learn(0.0, 40.0, 60.0) is True
        assert tw._outlier_streak == 0
        assert tw.turn_samples == 3
        assert tw._peak_count == input_worker.TURN_RATIO_PRIOR_WEIGHT + 1

        # 真实灵敏度变化到 2.5：前 2 次带外样本拦截，第 3 次起连续放行，累计超过先验（4 票）后切到 2.5
        for _ in range(6):
            tw._pending = 20.0
            tw._pending_chunks = 1
            tw._anchor = 0.0
            assert tw._learn(0.0, 20.0, 50.0) is True
        assert tw.turn_ratio == pytest.approx(1.5)
        tw._pending = 20.0
        tw._pending_chunks = 1
        tw._anchor = 0.0
        assert tw._learn(0.0, 20.0, 50.0) is True
        assert tw.turn_ratio == pytest.approx(2.5)
    finally:
        tw.stop()


def test_small_turn_releases_without_still_timeout_and_big_target_preempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """<8 度微调需有 >=0.5 度变化或满 0.4 秒宽限才放行；>=8 度转向宽限后遇 >=45 度新目标抢先放行。"""
    monkeypatch.setattr(input_worker._TurnWorker, '_loop', lambda self: None)
    now = [100.0]
    monkeypatch.setattr(input_worker.time, 'monotonic', lambda: now[0])
    tw = input_worker._TurnWorker(lambda _: None, lambda: 0.0)
    try:
        # <8 度小微调：宽限内画面未动仍等待，避免转稳前提前补转；满 0.4 秒宽限或已有变化即放行
        tw._pending = 3.0
        tw._pending_at = 100.0
        tw._anchor = 10.0
        now[0] = 100.2
        assert tw._learn(10.0, 3.0, 10.0) is False
        assert tw._learn(10.0, 3.0, 11.0) is True
        assert tw._pending is None

        tw._pending = 3.0
        tw._pending_at = 100.0
        tw._anchor = 10.0
        now[0] = 100.45
        assert tw._learn(10.0, 3.0, 10.0) is True
        assert tw._pending is None

        # >=8 度转向：宽限内仍等待；宽限过后无大目标进静止计时，遇 >=45 度新目标立刻放行
        tw._pending = 12.0
        tw._pending_at = 100.0
        tw._anchor = 10.0
        tw._target = 15.0
        now[0] = 100.2
        assert tw._learn(10.0, 12.0, 10.0) is False
        now[0] = 100.5
        assert tw._learn(10.0, 12.0, 10.0) is False
        assert tw._still_since == 100.5
        tw._target = 100.0
        assert tw._learn(10.0, 12.0, 10.0) is True
        assert tw._pending is None
    finally:
        tw.stop()
