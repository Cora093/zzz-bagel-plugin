"""贝果计划插件：松键兜底（退出钩子与操作收尾）的单元测试。

用假的 ctx/controller 记录按键释放，不初始化游戏上下文：
覆盖正常松键、controller 已被清理时不外抛、退出钩子幂等，
以及 ``BagelOperation`` 生命周期里 ``handle_init`` 注册、``after_operation_done`` 释放。
"""

from __future__ import annotations

import atexit
import importlib
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from one_dragon.base.operation.operation_base import OperationResult
from pynput import mouse

REPO_DIR = Path(__file__).resolve().parents[1]
"""插件仓库根：放进来后插件以 ``bagel`` 包名加载。"""

ALL_KEYS = {'w', 'a', 's', 'd', 'f'}
"""路线执行时会被按下的五个键：四个方向键加交互键。"""


@pytest.fixture(scope='module', autouse=True)
def _plugin_on_sys_path() -> Iterator[None]:
    """插件以仓库根为模块根，导入前先把它放进 sys.path。"""
    entry = str(REPO_DIR)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    yield


def _load_operation_module() -> ModuleType:
    """等 fixture 把仓库根放进 sys.path 后再导入插件模块。"""
    return importlib.import_module('bagel.operation.operation')


class _FakeKeyboard:
    """记录键盘 release；reset 只松开登记为按下的键。"""

    def __init__(self, held: Iterable[str] = ()) -> None:
        self.held: set[str] = set(held)
        self.released: list[str] = []

    def release(self, button: object) -> None:
        char = getattr(button, 'char', None)
        self.released.append(char if isinstance(char, str) else str(button))

    def reset(self) -> None:
        """真实 reset 只处理持续按下的集合，这里同样只吐登记过的键。"""
        self.released.extend(sorted(self.held))
        self.held.clear()


class _FakeMouse:
    """记录鼠标 release。"""

    def __init__(self) -> None:
        self.released: list[object] = []

    def release(self, button: object) -> None:
        self.released.append(button)


class _FakeKeyboardController:
    """键盘与鼠标的假实现，默认按住四个方向键。"""

    def __init__(self, held: Iterable[str] = ('w', 'a', 's', 'd')) -> None:
        self.keyboard = _FakeKeyboard(held)
        self.mouse = _FakeMouse()

    def reset(self) -> None:
        self.keyboard.reset()


class _FakeBtnController:
    """记录 reset 次数。"""

    def __init__(self) -> None:
        self.reset_count: int = 0

    def reset(self) -> None:
        self.reset_count += 1


class _FakeGameConfig:
    """只提供松键需要的动作键映射。"""

    def get_action_keys(self, _mode: str) -> dict[str, str]:
        return {'move_w': 'w', 'interact': 'f'}


class _FakeController:
    """release_flow_inputs 用到的最小 controller。"""

    def __init__(self) -> None:
        self.is_moving: bool = True
        self.background_mode: bool = False
        self.btn_controller = _FakeBtnController()
        self.keyboard_controller = _FakeKeyboardController()
        self.game_config = _FakeGameConfig()


class _FakeCtx:
    """只持有 controller 的假上下文。"""

    def __init__(self, controller: object) -> None:
        self.controller = controller


class _BrokenCtx:
    """模拟 controller 已被 cleanup 的上下文。"""

    @property
    def controller(self) -> object:
        raise RuntimeError('controller 已清理')


class _FakeEventBus:
    """基类收尾会解除事件订阅。"""

    def unlisten_all_event(self, _listener: object) -> None:
        pass


class _FakeRunContext:
    """只提供 event_bus 的运行上下文。"""

    def __init__(self) -> None:
        self.event_bus = _FakeEventBus()


class _FakeScreenLoader:
    """画面定义一直在：handle_init 的画面自愈检查直接通过，不重注入。"""

    def get_area(self, _screen_name: str, _area_name: str) -> object:
        return object()


class _IntegrationCtx(_FakeCtx):
    """after_operation_done 会解订阅，需要 run_context 与 unlisten_all_event；
    handle_init 会检查插件画面，需要 screen_loader。"""

    def __init__(self, controller: object) -> None:
        super().__init__(controller)
        self.run_context = _FakeRunContext()
        self.screen_loader = _FakeScreenLoader()

    def unlisten_all_event(self, _listener: object) -> None:
        pass


def test_release_inputs_safely_releases_five_keys() -> None:
    """正常路径松开 w/a/s/d/f 五个键并重置输入状态。"""
    operation = _load_operation_module()
    controller = _FakeController()

    operation.release_inputs_safely(_FakeCtx(controller))

    released = controller.keyboard_controller.keyboard.released
    assert set(released) == ALL_KEYS
    assert controller.keyboard_controller.mouse.released == [mouse.Button.left]
    assert controller.is_moving is False
    assert controller.btn_controller.reset_count == 1


def test_release_inputs_safely_swallows_controller_error() -> None:
    """controller 已被清理时只记日志，不向调用方抛异常。"""
    operation = _load_operation_module()

    operation.release_inputs_safely(_BrokenCtx())


def test_register_release_at_exit_only_registers_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """注册两次只保留一个退出回调，回调执行时确实松键。"""
    operation = _load_operation_module()
    monkeypatch.setattr(operation, '_EXIT_HOOK_REGISTERED', False)
    registered: list[Callable[[], None]] = []
    monkeypatch.setattr(atexit, 'register', registered.append)
    ctx = _FakeCtx(_FakeController())

    operation.register_release_at_exit(ctx)
    operation.register_release_at_exit(ctx)

    assert len(registered) == 1
    registered[0]()
    assert set(ctx.controller.keyboard_controller.keyboard.released) == ALL_KEYS


def test_exit_hook_swallows_broken_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    """退出回调里 controller 已清理也不抛出。"""
    operation = _load_operation_module()
    monkeypatch.setattr(operation, '_EXIT_HOOK_REGISTERED', False)
    registered: list[Callable[[], None]] = []
    monkeypatch.setattr(atexit, 'register', registered.append)

    operation.register_release_at_exit(_BrokenCtx())
    registered[0]()


def test_operation_lifecycle_registers_hook_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    """BagelOperation 生命周期：handle_init 注册钩子，after_operation_done 松键。

    不在构造里初始化完整操作对象；只挂收尾路径用到的属性。
    """
    operation = _load_operation_module()
    monkeypatch.setattr(operation, '_EXIT_HOOK_REGISTERED', False)
    registered: list[Callable[[], None]] = []
    monkeypatch.setattr(atexit, 'register', registered.append)
    controller = _FakeController()

    op = operation.BagelOperation.__new__(operation.BagelOperation)
    op.ctx = _IntegrationCtx(controller)
    op.op_name = 'test'
    op.op_callback = None
    op.last_screenshot = None

    op.handle_init()
    op.handle_init()
    assert len(registered) == 1

    op.after_operation_done(OperationResult(False, 'x'))
    released = controller.keyboard_controller.keyboard.released
    assert set(released) == ALL_KEYS
    assert len(released) == len(ALL_KEYS) + 1  # w 由 reset 与显式释放各一次
