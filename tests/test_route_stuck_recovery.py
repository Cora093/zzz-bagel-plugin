"""贝果路线执行：卡死恢复阶梯与退让停止安全的行为测试。

用 ``__new__`` 造一个只挂卡死恢复所需字段的执行器，假 input_worker 记录
``side_step`` / ``back_step`` / ``retreat_step``，手动推进时钟，不真的 sleep。
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from one_dragon.base.operation.operation_round_result import OperationRoundResultEnum

REPO_DIR = Path(__file__).resolve().parents[1]
"""插件仓库根：放进来后插件以 ``bagel`` 包名加载。"""

POSITION = (0.0, 0.0)
"""测试里角色原地不动，位置取路径线起点。"""


@pytest.fixture(scope='module', autouse=True)
def _plugin_on_sys_path() -> Iterator[None]:
    """插件以仓库根为模块根，导入前先把它放进 sys.path。"""
    entry = str(REPO_DIR)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    yield


def _load_walker_module() -> ModuleType:
    """等 fixture 把仓库根放进 sys.path 后再导入插件模块。"""
    return importlib.import_module('bagel.route.route_walker')


def _load_input_module() -> ModuleType:
    return importlib.import_module('bagel.route.input_worker')


def _load_operation_module() -> ModuleType:
    return importlib.import_module('bagel.operation.operation')


class _Clock:
    """可手动推进的假时钟；同时假装墙钟，卡位计时不真的等。"""

    def __init__(self) -> None:
        self.now: float = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)


class _FakeInputWorker:
    """只记录路线执行器会调用的按键方法。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.forward_released_at: float | None = None

    def side_step(self, direction: int, duration: float = 0.2) -> None:
        self.calls.append(('side_step', direction))

    def back_step(self, duration: float = 0.2) -> None:
        self.calls.append(('back_step', 0))

    def retreat_step(self, direction: int, back_duration: float, side_duration: float) -> None:
        self.calls.append(('retreat_step', direction))

    def stop_forward(self) -> None:
        self.calls.append(('stop_forward', 0))

    def stop_backward(self) -> None:
        self.calls.append(('stop_backward', 0))

    def submit_turn(self, target: float | None, **_kwargs: object) -> None:
        self.calls.append(('submit_turn', 0))

    def take_actions(self) -> list[tuple[str, int]]:
        """取出从上次调用以来产生的按键动作。"""
        actions = [call for call in self.calls if call[0].endswith('_step')]
        self.calls.clear()
        return actions


def _make_walker(
    module: ModuleType, clock: _Clock, input_worker: _FakeInputWorker,
) -> object:
    """造一个只挂卡死恢复所需字段的执行器；游标停在第 2 个途径点。"""
    walker = module.BagelRouteWalker.__new__(module.BagelRouteWalker)
    walker.routes = [module.BagelRoute(
        route_id='test', name='test', spawn_id='test',
        points=[
            module.BagelRoutePoint(0.0, 0.0, 'L1', module.BEACON),
            module.BagelRoutePoint(100.0, 0.0, 'L1', module.BEACON),
        ],
    )]
    walker.route_index = 0
    walker.cursor = 1
    walker.point_started_at = None
    walker.stuck_since = None
    walker.stuck_steps = 0
    walker.retreats = 0
    walker.backtracks = 0
    walker.lane_offsets = {}
    walker.point_backtracks = {}
    walker.recover_stage = ''
    walker.recover_point = 0
    walker.recover_until = 0.0
    walker.recover_aim_frames = 0
    walker.steps = 0
    walker.last_input_frame = None
    walker.last_screenshot_time = 0.0
    walker.sprinting = False
    walker.current_kind = None
    walker.f_pressed = False
    walker.mechanical_hold_until = None
    walker.mechanical_retried = False
    walker.target_wait_since = None
    walker.electronic_small_steps = False
    walker.small_step_since = None
    walker.elevator_phase = None
    walker.elevator_since = None
    walker.input_worker = input_worker
    walker._path_side = lambda position, point, angle: 1
    walker._after_round_wait = lambda **_kwargs: None
    return walker


def _trigger_stuck(walker: object, clock: _Clock) -> object:
    """登记卡位起点，推进过 ``STUCK_SECONDS`` 后再判一次，返回本轮结果。"""
    point = walker._current_point()
    walker._check_stuck(POSITION, point, angle=0.0)
    clock.now += 3.5
    return walker._check_stuck(POSITION, point, angle=0.0)


def test_stuck_ladder_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """完整阶梯：退让 2 次（第 2 次换边）→ 侧移 5 次（第 2 次起退半步）→ 换道 3 次 → 失败。"""
    module = _load_walker_module()
    clock = _Clock()
    input_worker = _FakeInputWorker()
    walker = _make_walker(module, clock, input_worker)
    monkeypatch.setattr(module, 'time', clock)

    retreat_cycle = [
        ('retreat_step', 1), ('retreat_step', -1),
        ('side_step', 1),
        ('back_step', 0), ('back_step', 0), ('back_step', 0), ('back_step', 0),
    ]
    for cycle in range(module.BACKTRACK_LIMIT):
        expected = list(retreat_cycle)
        for _ in range(len(retreat_cycle)):
            assert _trigger_stuck(walker, clock).result == OperationRoundResultEnum.WAIT
        assert input_worker.take_actions() == expected
        backtrack = _trigger_stuck(walker, clock)
        assert backtrack.result == OperationRoundResultEnum.WAIT
        assert walker.cursor == 0
        assert walker.recover_stage == 'settle'
        assert walker.recover_point == 1
        assert walker.backtracks == cycle + 1
        assert walker.point_backtracks[1] == cycle + 1
        assert walker.lane_offsets[1] == module.BACKTRACK_LANE_OFFSETS[cycle]
        assert walker.retreats == 0  # 进入上一点时单点状态已清
        assert walker.stuck_steps == 0
        walker._enter_point(0, 1)  # 模拟走回上一点后重新进入卡点
        walker.recover_stage = ''

    for _ in range(len(retreat_cycle)):
        assert _trigger_stuck(walker, clock).result == OperationRoundResultEnum.WAIT
    assert input_worker.take_actions() == list(retreat_cycle)
    result = _trigger_stuck(walker, clock)
    assert result.result == OperationRoundResultEnum.FAIL
    assert isinstance(result.data, _load_operation_module().BagelRecoverableFailure)
    assert walker.cursor == 1


def test_recover_settle_walk_back_aim(monkeypatch: pytest.MonkeyPatch) -> None:
    """回退恢复三段：松键停稳 → 走回上一点 → 等视角转正好后重新进入卡点。"""
    module = _load_walker_module()
    clock = _Clock()
    walker = _make_walker(module, clock, _FakeInputWorker())
    monkeypatch.setattr(module, 'time', clock)
    walker.locate = lambda: POSITION
    walker._check_timeout = lambda point: None
    walker._check_hud = lambda: None
    walker.player_angle = lambda: 0.0
    walker._enter_point(0, 0)
    walker.recover_stage = 'settle'
    walker.recover_point = 1
    walker.recover_until = clock.now + module.RECOVER_SETTLE_SECONDS

    assert walker._move_toward().result == OperationRoundResultEnum.WAIT
    assert walker.recover_stage == 'settle'
    clock.now += module.RECOVER_SETTLE_SECONDS
    assert walker._move_toward().result == OperationRoundResultEnum.WAIT
    assert walker.recover_stage == 'aim'  # 已走到上一点，等视角转正
    assert walker._move_toward().result == OperationRoundResultEnum.WAIT
    assert walker.recover_aim_frames == 1
    assert walker._move_toward().result == OperationRoundResultEnum.WAIT
    assert walker.cursor == 1
    assert walker.recover_stage == ''


def test_enter_point_keeps_lane_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    """换道偏移按 cursor 跨点保留，进入新点只清退让/侧移等单点状态。"""
    module = _load_walker_module()
    walker = _make_walker(module, _Clock(), _FakeInputWorker())
    walker.lane_offsets[1] = 10.0
    walker.point_backtracks[1] = 2
    walker.retreats = module.RETREAT_LIMIT
    walker.stuck_steps = 4

    walker._enter_point(0, 1)

    assert walker.lane_offsets == {1: 10.0}
    assert walker.point_backtracks == {1: 2}
    assert walker.retreats == 0
    assert walker.stuck_steps == 0


def test_retreat_side() -> None:
    """退让侧别：算得出就跟随，算不出左右交替；第二次退让换边。"""
    module = _load_walker_module()
    assert module.retreat_side(1, 1) == 1
    assert module.retreat_side(1, 2) == -1
    assert module.retreat_side(-1, 1) == -1
    assert module.retreat_side(-1, 2) == 1
    assert module.retreat_side(None, 1) == 1
    assert module.retreat_side(None, 2) == -1


def test_lane_offset() -> None:
    """换道偏移按回退次数取，超出表长时沿用最后一个。"""
    module = _load_walker_module()
    assert [module.lane_offset(tries) for tries in range(5)] == [10.0, -16.0, 22.0, 22.0, 22.0]


def test_lane_shift_identity_offset_and_taper() -> None:
    """换道瞄准：无偏移时恒等，有偏移时垂直平移，进到 30px 内按距离衰减。"""
    module = _load_walker_module()
    walker = _make_walker(module, _Clock(), _FakeInputWorker())
    point = walker._current_point()

    assert walker._lane_shift((100.0, 0.0), point, (0.0, 0.0)) == (100.0, 0.0)

    walker.lane_offsets[1] = 10.0
    assert walker._lane_shift((100.0, 0.0), point, (0.0, 0.0)) == (100.0, 10.0)

    x, y = walker._lane_shift((100.0, 0.0), point, (80.0, 0.0))
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(10.0 * 20.0 / module.LANE_TAPER_DISTANCE)


def test_backtrack_index_skips_interaction_points() -> None:
    """回退只退到途径点，不回头重做箱子/保险箱。"""
    module = _load_walker_module()
    walker = _make_walker(module, _Clock(), _FakeInputWorker())
    walker.routes = [module.BagelRoute(
        route_id='test', name='test', spawn_id='test',
        points=[
            module.BagelRoutePoint(0.0, 0.0, 'L1', module.BEACON),
            module.BagelRoutePoint(50.0, 0.0, 'L1', 'box'),
            module.BagelRoutePoint(100.0, 0.0, 'L1', module.BEACON),
        ],
    )]
    walker.cursor = 2
    assert walker._backtrack_index() == 0

    walker.routes[0].points.pop(0)
    walker.cursor = 1
    assert walker._backtrack_index() is None


def test_stuck_timer_resets_when_forward_released(monkeypatch: pytest.MonkeyPatch) -> None:
    """前进键真的松开超过 1 秒（交互、等待）时，卡位计时清零、不触发恢复动作。"""
    module = _load_walker_module()
    clock = _Clock()
    input_worker = _FakeInputWorker()
    walker = _make_walker(module, clock, input_worker)
    monkeypatch.setattr(module, 'time', clock)
    point = walker._current_point()

    walker._check_stuck(POSITION, point, angle=0.0)
    input_worker.forward_released_at = clock.now - 2.0
    clock.now += 3.5

    assert walker._check_stuck(POSITION, point, angle=0.0) is None
    assert walker.stuck_since is None
    assert input_worker.take_actions() == []


class _FakeController:
    """记录每个方向键按下/松开，并维护当前实际状态。"""

    def __init__(self) -> None:
        self.state: dict[str, bool] = {'w': False, 's': False, 'a': False, 'd': False, 'f': False}
        self.history: list[tuple[str, bool]] = []

    def _set(self, key: str, press: bool) -> None:
        self.state[key] = press
        self.history.append((key, press))

    def move_w(self, press: bool = False, release: bool = False) -> None:
        self._set('w', press and not release)

    def move_s(self, press: bool = False, release: bool = False) -> None:
        self._set('s', press and not release)

    def move_a(self, press: bool = False, release: bool = False) -> None:
        self._set('a', press and not release)

    def move_d(self, press: bool = False, release: bool = False) -> None:
        self._set('d', press and not release)

    def interact(self, press: bool = False, release: bool = False) -> None:
        self._set('f', press and not release)


class _FakeCtx:
    """只带 controller 的假上下文。"""

    def __init__(self, controller: _FakeController) -> None:
        self.controller = controller


def _make_input_worker(input_module: ModuleType, controller: _FakeController) -> object:
    """造一个只挂退让路径所需字段的输入 worker。"""
    worker = input_module.BagelInputWorker.__new__(input_module.BagelInputWorker)
    worker.ctx = _FakeCtx(controller)
    worker._forward_lock = threading.Lock()
    worker._forward_want = False
    worker._forward_down = False
    worker._forward_step_duration = None
    worker._forward_step_until = None
    worker._forward_released_at = 0.0
    worker._backward_lock = threading.Lock()
    worker._backward_want = False
    worker._backward_down = False
    worker._stopping = False
    worker._action_epoch = 0
    return worker


def test_retreat_releases_side_key_when_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """退让途中（后退段）置停止标志：作废后不再按横移键，所有键最终松开。"""
    input_module = _load_input_module()
    controller = _FakeController()
    worker = _make_input_worker(input_module, controller)
    worker_holder = worker

    class _StopClock(_Clock):
        """第二段 sleep（后退 S）期间模拟暂停/停止。"""

        def __init__(self) -> None:
            super().__init__()
            self.sleep_calls = 0

        def sleep(self, seconds: float) -> None:
            self.sleep_calls += 1
            if self.sleep_calls == 2:
                worker_holder.release_all()
            super().sleep(seconds)

    clock = _StopClock()
    monkeypatch.setattr(input_module, 'time', clock)
    command = input_module._Command(
        input_module.RETREAT_STEP, duration=0.7, angle=1.0, side_duration=0.3,
    )

    worker._run_retreat(command)

    assert worker._action_epoch == 1
    assert all(not pressed for pressed in controller.state.values())
    assert ('s', True) in controller.history
    # 作废发生在后退段：handler 不能再进入横移段按 D。
    assert not any(key == 'd' and press for key, press in controller.history)
    assert not any(key == 'a' and press for key, press in controller.history)
    assert controller.history[-1] == ('s', False)


def test_release_all_releases_every_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """暂停路径的 release_all 必须真的松开全部 5 个键，而不是只改意图。"""
    input_module = _load_input_module()
    controller = _FakeController()
    worker = _make_input_worker(input_module, controller)
    for press in (
        controller.move_w, controller.move_s, controller.move_a,
        controller.move_d, controller.interact,
    ):
        press(press=True)
    assert all(controller.state.values())

    worker.release_all()

    assert all(not pressed for pressed in controller.state.values())


class _RaisingWController(_FakeController):
    """move_w 抛错的 controller：模拟 controller 已被清理。"""

    def move_w(self, press: bool = False, release: bool = False) -> None:
        raise RuntimeError('controller 已清理')


def test_release_all_continues_after_one_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """一个键的 release 抛错，其余键仍必须被松开。"""
    input_module = _load_input_module()
    controller = _RaisingWController()
    worker = _make_input_worker(input_module, controller)
    controller.state['w'] = True  # W 已按下，但 release 会抛错
    for press in (controller.move_s, controller.move_a, controller.move_d, controller.interact):
        press(press=True)

    worker.release_all()

    assert controller.state['w'] is True  # release 抛错，松不掉
    assert controller.state['s'] is False
    assert controller.state['a'] is False
    assert controller.state['d'] is False
    assert controller.state['f'] is False
