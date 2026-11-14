"""贝果输入线程：暂停作废排队中的即时动作，并逐键松开全部按键。

用 ``__new__`` 造一个只挂即时动作所需字段的输入线程，假 controller 记录每个键的
按下和松开，假时钟代替真 sleep。用例取自 #38 的输入线程部分。
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
"""插件仓库根：放进来后插件以 ``bagel`` 包名加载。"""


@pytest.fixture(scope='module', autouse=True)
def _plugin_on_sys_path() -> Iterator[None]:
    """插件以仓库根为模块根，导入前先把它放进 sys.path。"""
    entry = str(REPO_DIR)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    yield


def _load_input_module() -> ModuleType:
    """等 fixture 把仓库根放进 sys.path 后再导入插件模块。"""
    return importlib.import_module('bagel.route.input_worker')


class _Clock:
    """假时钟：sleep 只推进读数，不真的等。"""

    def __init__(self) -> None:
        self.now: float = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)


class _FakeController:
    """记录 W/S/A/D/F 每次按下与松开，并维护当前实际状态。"""

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


class _RaisingWController(_FakeController):
    """move_w 抛错的 controller：模拟 controller 已被清理。"""

    def move_w(self, press: bool = False, release: bool = False) -> None:
        raise RuntimeError('controller 已清理')


class _FakeCtx:
    """只带 controller 的假上下文。"""

    def __init__(self, controller: _FakeController) -> None:
        self.controller: _FakeController = controller


class _FakeInstantChannel:
    """记录入队指令与清空次数，不起线程。"""

    def __init__(self) -> None:
        self.commands: list[object] = []
        self.cleared: int = 0

    def put(self, command: object) -> None:
        self.commands.append(command)

    def clear(self) -> None:
        self.cleared += 1


def _make_input_worker(input_module: ModuleType, controller: _FakeController) -> object:
    """造一个只挂即时动作与松键所需字段的输入线程。"""
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
    worker._instant_channel = _FakeInstantChannel()
    return worker


def test_release_all_releases_every_key() -> None:
    """暂停时的 release_all 必须真的松开 5 个键，而不是只改意图。"""
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


def test_release_all_continues_after_one_key_error() -> None:
    """一个键松开时抛错，其余键仍必须松开。"""
    input_module = _load_input_module()
    controller = _RaisingWController()
    worker = _make_input_worker(input_module, controller)
    controller.state['w'] = True
    for press in (controller.move_s, controller.move_a, controller.move_d, controller.interact):
        press(press=True)

    worker.release_all()

    assert controller.state['w'] is True
    assert not any(controller.state[key] for key in ('s', 'a', 'd', 'f'))


def test_enqueue_stamps_epoch_and_release_clears_queue() -> None:
    """入队时记下当前版本号；暂停时版本号前进并清空还没执行的队列。"""
    input_module = _load_input_module()
    worker = _make_input_worker(input_module, _FakeController())
    channel = worker._instant_channel

    worker.side_step(1)
    worker._action_epoch = 3
    worker.hold_interact()
    assert [command.epoch for command in channel.commands] == [0, 3]

    worker.release_all()

    assert worker._action_epoch == 4
    assert channel.cleared == 1


def test_stale_commands_do_not_press_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """入队后暂停过（版本号变了），已出队的侧移、长按 F、点按 F 一个键都不按。"""
    input_module = _load_input_module()
    controller = _FakeController()
    worker = _make_input_worker(input_module, controller)
    monkeypatch.setattr(input_module, 'time', _Clock())
    worker._action_epoch = 1
    Command = input_module._Command

    worker._run_instant(Command(input_module.SIDE_STEP, duration=0.2, angle=1.0))
    worker._run_instant(Command(input_module.HOLD_INTERACT))
    worker._run_instant(Command(input_module.TAP_INTERACT, duration=0.2))

    assert controller.history == []


def test_stale_release_interact_still_releases() -> None:
    """松开 F 不受版本号限制：多松一次无害，漏松会让 F 一直按着。"""
    input_module = _load_input_module()
    controller = _FakeController()
    worker = _make_input_worker(input_module, controller)
    worker._action_epoch = 1

    worker._run_instant(input_module._Command(input_module.RELEASE_INTERACT))

    assert controller.history == [('f', False)]


def test_current_side_step_presses_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    """没暂停时侧移照常执行：按下 D 再松开。"""
    input_module = _load_input_module()
    controller = _FakeController()
    worker = _make_input_worker(input_module, controller)
    monkeypatch.setattr(input_module, 'time', _Clock())
    worker._action_epoch = 2

    worker._run_instant(input_module._Command(input_module.SIDE_STEP, duration=0.2, angle=1.0, epoch=2))

    assert controller.history == [('d', True), ('d', False)]


def test_stopping_worker_skips_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """停止后即使版本号对得上也不再按键。"""
    input_module = _load_input_module()
    controller = _FakeController()
    worker = _make_input_worker(input_module, controller)
    monkeypatch.setattr(input_module, 'time', _Clock())
    worker._stopping = True

    worker._run_instant(input_module._Command(input_module.SIDE_STEP, duration=0.2, angle=-1.0))

    assert controller.history == []
