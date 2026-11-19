"""逐帧驱动真实路线节点，检查鼠标和 F 的顺序、次数与停止条件。"""

from dataclasses import dataclass, field
from unittest.mock import Mock, call

import numpy as np
import pytest

from bagel.operation import container
from bagel.operation.container import ContainerRecovery
from bagel.operation.operation import BagelOperation
from bagel.operation.unlock_safe import BagelUnlockSafe
from bagel.route import route_walker
from bagel.route.input_worker import BagelInputWorker
from bagel.route.route_model import KEYCARD_DOOR, BagelRoute, BagelRoutePoint
from bagel.route.route_walker import BagelRouteWalker
from bagel.vision.prompt_list import PromptRow

from .prompt_frames import ObservedFrame

DOOR = PromptRow(555, 'door', '金属单开门')
READER = PromptRow(634, 'card_reader', '芝邦邦餐厅1/1')
KEYCARD = PromptRow(555, 'pickup', '六昏街录像店一楼')
SEARCHED = PromptRow(634, 'searched', '公文包')
MECHANICAL = PromptRow(555, 'mechanical_safe', '机械保险箱')
ELECTRONIC = PromptRow(634, 'electronic_safe', '电子保险箱')


@dataclass
class Scene:
    walker: BagelRouteWalker
    worker: Mock
    panels: set[str] = field(default_factory=set)
    now: float = 100.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def show(self, stamp: float, selected: PromptRow | None, *others: PromptRow) -> None:
        """时间由用例指定。同一时间戳代表同一张截图。"""
        self.now = stamp
        self.walker.last_screenshot_time = stamp
        self.walker.prompt_frame = ObservedFrame(stamp, selected, *others)

    def target(self, kind: str) -> None:
        point = BagelRoutePoint(0, 0, '测试层', kind)
        self.walker.routes = [BagelRoute('test', '测试路线', 'spawn', points=[point])]


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch) -> Scene:
    """替换窗口、输入线程、地图加载和等待。选行、节点和计数使用插件实现。"""
    ctx = Mock()
    ctx.controller.game_config.turn_dx = 1.0
    ctx.controller.game_win.get_hwnd.return_value = None
    worker = Mock(spec_set=BagelInputWorker)
    worker.instant_idle = True
    monkeypatch.setattr(route_walker, 'BagelInputWorker', Mock(return_value=worker))
    monkeypatch.setattr(route_walker, 'BagelRouteStore', Mock())
    monkeypatch.setattr(route_walker, 'load_map_locators', Mock(return_value=[Mock()]))
    monkeypatch.setattr(BagelOperation, '_after_round_wait', Mock())
    walker = BagelRouteWalker(ctx)
    result = Scene(walker, worker)
    walker.last_screenshot = np.zeros((1, 1, 3), np.uint8)
    monkeypatch.setattr(route_walker, 'time', result)
    monkeypatch.setattr(container, 'time', result)
    monkeypatch.setattr(walker, '_has', lambda area: area in result.panels)
    return result


@pytest.mark.parametrize(('item', 'neighbor'), [
    (KEYCARD, SEARCHED),
    (PromptRow(515, 'pickup', '纳米纤维×4'), ELECTRONIC),
], ids=['round-182', 'round-186'])
def test_moving_pickup_taps_once_without_stopping(scene: Scene, item: PromptRow, neighbor: PromptRow) -> None:
    walker = scene.walker
    scene.show(100, item, neighbor)
    walker._try_pickup(moving=True)
    walker._try_pickup(moving=True)
    assert scene.worker.method_calls == []
    scene.show(100.2, item, neighbor)
    walker._try_pickup(moving=True)
    scene.show(100.3, item, neighbor)
    walker._try_pickup(moving=True)
    assert scene.worker.method_calls == [call.tap_interact(0.2)]
    assert walker.pickup_presses == 1


def test_moving_past_loot_on_another_row_sends_no_input(scene: Scene) -> None:
    for stamp in (100, 100.2, 100.4):
        scene.show(stamp, SEARCHED, KEYCARD)
        scene.walker._try_pickup(moving=True)
    assert scene.worker.method_calls == []


def test_round_175_stop_at_reader_then_scroll_off_door_and_hold(scene: Scene) -> None:
    walker = scene.walker
    scene.target(KEYCARD_DOOR)
    scene.show(100, DOOR, READER)
    assert walker._prompt_stop(41) is None
    assert walker._prompt_stop(40).is_success
    assert walker.current_kind == 'card_reader'
    for stamp in (100.1, 100.2, 100.2):
        scene.show(stamp, DOOR, READER)
        walker.interact()
    scene.worker.scroll_prompt.assert_called_once_with(True)
    scene.worker.hold_interact.assert_not_called()
    for stamp in (100.5, 100.6, 101.0):
        scene.show(stamp, READER, DOOR)
        walker.interact()
        scene.worker.hold_interact.assert_not_called()
    scene.show(101.2, READER, DOOR)
    walker.interact()
    scene.worker.hold_interact.assert_called_once_with()
    scene.worker.tap_interact.assert_not_called()
    assert scene.worker.scroll_prompt.call_count == 1


def test_missing_keycard_stops_before_any_hold(scene: Scene) -> None:
    scene.walker.current_kind = 'card_reader'
    scene.show(100, PromptRow(555, 'card_reader', '芝邦邦餐厅0/1'))
    result = scene.walker.interact()
    assert result.is_fail and result.status == scene.walker.STATUS_KEYCARD_MISSING
    scene.worker.hold_interact.assert_not_called()
    scene.worker.tap_interact.assert_not_called()


def test_mechanical_timeout_reselects_before_retry_and_releases_on_panel(scene: Scene) -> None:
    walker = scene.walker
    walker.current_kind = 'mechanical_safe'
    scene.show(100, MECHANICAL)
    walker.interact()
    scene.worker.hold_interact.assert_not_called()
    scene.show(100.1, MECHANICAL)
    walker.interact()
    scene.worker.hold_interact.assert_called_once_with()
    scene.show(105.2, ELECTRONIC, MECHANICAL)
    walker.interact()
    scene.worker.release_interact.assert_called_once_with()
    for stamp in (105.5, 105.6):
        scene.show(stamp, ELECTRONIC, MECHANICAL)
        walker.interact()
    scene.worker.scroll_prompt.assert_called_once_with(False)
    scene.worker.hold_interact.assert_called_once_with()
    scene.show(106, MECHANICAL)
    walker.interact()
    scene.worker.hold_interact.assert_called_once_with()
    scene.show(106.1, MECHANICAL)
    walker.interact()
    assert scene.worker.hold_interact.call_count == 2
    scene.panels.add('搜查安全箱')
    scene.show(107, None)
    assert walker.interact().is_success
    assert scene.worker.release_interact.call_count == 2
    scene.worker.tap_interact.assert_not_called()


def test_cleanup_selects_off_row_loot_then_advances_without_walking(scene: Scene) -> None:
    walker = scene.walker
    for stamp in (100, 100.2):
        scene.show(stamp, SEARCHED, KEYCARD)
        walker.advance()
    scene.worker.scroll_prompt.assert_called_once_with(False)
    scene.show(100.5, KEYCARD, SEARCHED)
    walker.advance()
    scene.worker.tap_interact.assert_not_called()
    scene.show(100.6, KEYCARD, SEARCHED)
    walker.advance()
    scene.worker.tap_interact.assert_called_once_with(0.2)
    assert walker.cursor == 0
    scene.show(101.5, None)
    assert walker.advance().is_success
    assert walker.cursor == 1
    assert all(c[0] in {'stop_forward', 'stop_backward', 'tap_interact', 'scroll_prompt'}
               or c == call.submit_turn(None) for c in scene.worker.method_calls)


def test_repeated_pickup_failure_skips_item_and_eventually_treats_bag_as_full(scene: Scene) -> None:
    walker = scene.walker
    for item in ('诡异药粉', '纳米纤维', '弱酸液滴', '特化支撑件'):
        for _ in range(10):
            scene.show(scene.now + 0.5, PromptRow(555, 'pickup', item))
            walker._try_pickup(moving=True)
    assert walker.pickup_ignored == {'诡异药粉', '纳米纤维', '弱酸液滴'}
    assert walker.pickup_failures == 3
    assert scene.worker.method_calls == [call.tap_interact(0.2)] * 9


def test_electronic_safe_spends_budget_only_on_confirmed_f_and_rechecks_after_timeout(
    scene: Scene, monkeypatch: pytest.MonkeyPatch,
) -> None:
    walker = scene.walker
    budget = ContainerRecovery(scene.monotonic)
    op = BagelUnlockSafe(walker.ctx, recovery=budget, prepare_open=walker._prepare_safe_open)
    monkeypatch.setattr(op, 'is_bagel_result', lambda: False)
    monkeypatch.setattr(op, '_has', lambda area, *_: area == '按键-普通攻击')

    def open_frame(stamp: float, selected: PromptRow, other: PromptRow) -> None:
        scene.show(stamp, selected, other)
        op.last_screenshot = walker.last_screenshot
        op.last_screenshot_time = stamp
        op._open_container()

    for stamp in (100, 100.2):
        open_frame(stamp, MECHANICAL, ELECTRONIC)
    assert budget.interactions == 0
    scene.worker.scroll_prompt.assert_called_once_with(True)
    for stamp in (100.5, 100.6, 100.6, 101.0):
        open_frame(stamp, ELECTRONIC, MECHANICAL)
    assert budget.interactions == 1
    for stamp in (102.7, 102.8):
        open_frame(stamp, MECHANICAL, ELECTRONIC)
    assert budget.interactions == 1
    assert scene.worker.scroll_prompt.call_args_list == [call(True), call(True)]
    for stamp in (103.1, 103.2):
        open_frame(stamp, ELECTRONIC, MECHANICAL)
    assert budget.interactions == 2
    assert walker.ctx.controller.interact.call_args_list == [call(press=True, press_time=0.2, release=True)] * 2
