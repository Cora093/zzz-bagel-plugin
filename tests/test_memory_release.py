"""长时间运行的内存回归（issue #101）：执行器、截图和监听在操作结束后立即释放。

用例关闭垃圾回收，只靠引用计数检查释放；循环引用会让弱引用一直存活。
"""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Iterator
from unittest.mock import Mock

import numpy as np
import pytest
from one_dragon.base.operation.context_event_bus import ContextEventBus
from one_dragon.base.operation.operation import Operation
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.base.operation.operation_round_result import OperationRoundResult

from bagel.operation import operation as bagel_operation
from bagel.operation.operation import BagelOperation, execute_foreign_operation
from bagel.route import route_walker
from bagel.route.route_walker import BagelRouteWalker


@pytest.fixture
def no_gc() -> Iterator[None]:
    """关掉自动垃圾回收，并在用例前清掉其它用例留下的循环。"""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def make_ctx() -> Mock:
    """窗口与输入用 Mock，运行事件总线用真实实现，避免 Mock 记录回调造成额外引用。"""
    ctx = Mock()
    ctx.controller.game_config.turn_dx = 1.0
    ctx.controller.game_win.get_hwnd.return_value = None
    ctx.controller.screenshot.side_effect = lambda: (time.time(), np.zeros((1080, 1920, 3), np.uint8))
    ctx.run_context.event_bus = ContextEventBus()
    ctx.unlisten_all_event = lambda obj: None  # Mock 会记住参数，让操作一直存活
    ctx.run_context.is_context_stop = False
    ctx.run_context.is_context_pause = False
    ctx.controller.game_config.get_action_keys.return_value = {'move_w': 'w', 'interact': 'f'}
    return ctx


def worker_threads(walker: BagelRouteWalker) -> list[threading.Thread]:
    """路线执行器输入线程的全部线程。"""
    worker = walker.input_worker
    return [
        worker._forward_thread, worker._backward_thread,
        worker._turn_worker._thread, worker._instant_channel._thread,
    ]


@pytest.fixture
def unsupported_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """每局都定位成功，但没有登记的出生点，走「出生点不支持」退出。"""
    store = Mock()
    store.return_value.load_spawns.return_value = []
    monkeypatch.setattr(route_walker, 'BagelRouteStore', store)
    monkeypatch.setattr(route_walker, 'load_map_locators', Mock(side_effect=AssertionError('应复用调用方的地图')))
    monkeypatch.setattr(BagelRouteWalker, 'locate', lambda self: (0.0, 0.0))
    monkeypatch.setattr(BagelOperation, '_after_round_wait', lambda self, wait=None, wait_round_time=None: None)
    monkeypatch.setattr(bagel_operation, 'ensure_screens', lambda loader: None)
    monkeypatch.setattr(bagel_operation, 'register_release_at_exit', lambda ctx: None)


@pytest.mark.usefixtures('unsupported_spawn', 'no_gc')
def test_walkers_release_without_gc_and_share_map_layers() -> None:
    ctx = make_ctx()
    layers = [Mock()]
    refs: list[weakref.ref[BagelRouteWalker]] = []
    screenshots: list[weakref.ref[np.ndarray]] = []
    for _ in range(3):
        walker = BagelRouteWalker(ctx, map_layers=layers)
        assert walker.locator.locators is layers
        result = walker.execute()
        assert result.status == BagelRouteWalker.STATUS_ANCHOR_MISS
        assert not any(thread.is_alive() for thread in worker_threads(walker))
        assert '_on_pause' not in vars(walker)
        assert '_on_resume' not in vars(walker)
        refs.append(weakref.ref(walker))
        screenshots.append(weakref.ref(walker.last_screenshot))
        del walker, result
    assert [ref() for ref in refs] == [None, None, None]
    assert [ref() for ref in screenshots] == [None, None, None]
    assert ctx.run_context.event_bus.callbacks == {
        key: [] for key in ctx.run_context.event_bus.callbacks
    }


@pytest.mark.usefixtures('unsupported_spawn')
def test_init_failure_stops_input_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bagel_operation, 'ensure_screens', Mock(side_effect=RuntimeError('画面定义损坏')))
    walker = BagelRouteWalker(make_ctx(), map_layers=[Mock()])
    assert all(thread.is_alive() for thread in worker_threads(walker))
    result = walker.execute()
    assert not result.success
    assert not any(thread.is_alive() for thread in worker_threads(walker))
    walker.input_worker.shutdown()  # 重复调用直接返回


class _Listener:
    """非操作对象的监听方；总线按回调的 ``__self__`` 解除监听，回调必须是绑定方法。"""

    def on_event(self, event: object) -> None:
        pass


class _Child(Operation):
    """模拟主仓子操作：成功结束，但不解除运行总线上的监听。"""

    @operation_node(name='完成', is_start_node=True, screenshot_before_round=False)
    def done(self) -> OperationRoundResult:
        return self.round_success()


class _Foreign(Operation):
    """模拟主仓操作：执行中调用子操作，并有一个不属于操作的监听。"""

    def __init__(self, ctx: Mock, outside: _Listener) -> None:
        super().__init__(ctx, op_name='主仓操作', need_check_game_win=False)
        self.outside: _Listener = outside

    @operation_node(name='执行', is_start_node=True, screenshot_before_round=False)
    def run(self) -> OperationRoundResult:
        self.ctx.run_context.event_bus.listen_event('其它', self.outside.on_event)
        return self.round_by_op_result(_Child(self.ctx, need_check_game_win=False).execute())


@pytest.mark.usefixtures('no_gc')
def test_foreign_operation_listeners_removed() -> None:
    ctx = make_ctx()
    bus = ctx.run_context.event_bus
    kept = _Listener().on_event
    bus.listen_event('其它', kept)
    outside = _Listener()
    operation = _Foreign(ctx, outside)
    ref = weakref.ref(operation)

    assert execute_foreign_operation(operation).success

    # 主仓收尾只在 ctx 上解除，这里补清运行总线：两个操作的监听都移除，其它监听保留。
    remaining = [callback for callbacks in bus.callbacks.values() for callback in callbacks]
    assert remaining == [kept, outside.on_event]
    del operation
    assert ref() is None
