"""逐帧驱动真实路线节点，检查鼠标和 F 的顺序、次数与停止条件。"""

from dataclasses import dataclass, field
from unittest.mock import Mock, call

import numpy as np
import pytest
from one_dragon.base.geometry.point import Point
from one_dragon.base.operation.operation_base import OperationResult
from one_dragon.base.operation.operation_round_result import OperationRoundResultEnum

from bagel.defs import INTERACT_TAP_SECONDS
from bagel.operation import container, organize_backpack
from bagel.operation.container import ContainerRecovery
from bagel.operation.operation import BagelOperation
from bagel.operation.organize_backpack import BagelOrganizeBackpack
from bagel.operation.unlock_safe import BagelUnlockSafe
from bagel.route import route_walker
from bagel.route.input_worker import STEP_REST_SECONDS, BagelInputWorker
from bagel.route.route_model import BEACON, KEYCARD_DOOR, BagelRoute, BagelRoutePoint
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
    assert scene.worker.method_calls == [call.tap_interact(INTERACT_TAP_SECONDS)]
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
    scene.worker.tap_interact.assert_called_once_with(INTERACT_TAP_SECONDS)
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
    assert scene.worker.method_calls == [call.tap_interact(INTERACT_TAP_SECONDS)] * 9


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
    assert walker.ctx.controller.interact.call_args_list == [call(press=True, press_time=INTERACT_TAP_SECONDS, release=True)] * 2


NANO = PromptRow(555, 'pickup', '纳米纤维×4')
POWDER = PromptRow(555, 'pickup', '诡异药粉')


def test_moving_pickup_has_no_fixed_cooldown_except_same_item(scene: Scene) -> None:
    """换成别的拾取物两帧判稳就按；同名条目 0.4 秒内不重按。"""
    frames = [(100, POWDER), (100.1, POWDER), (100.15, None), (100.2, NANO), (100.25, NANO),
              (100.3, NANO), (100.4, NANO), (100.7, NANO)]
    taps = []
    for stamp, row in frames:
        scene.show(stamp, row)
        scene.walker._try_pickup(moving=True)
        taps.append(len(scene.worker.method_calls))
    assert taps == [0, 1, 1, 1, 2, 2, 2, 3]
    assert scene.walker.pickups == 1


def test_fast_round_skips_locate_until_time_limit(scene: Scene, monkeypatch: pytest.MonkeyPatch) -> None:
    """拾取物在选中行时只走快轮，不定位；满 0.6 秒交回完整一轮。"""
    walker = scene.walker
    scene.target('box')
    walker.last_position = (50.0, 0.0)
    monkeypatch.setattr(walker, 'locate', Mock(side_effect=RuntimeError('完整一轮')))
    monkeypatch.setattr(walker, '_check_hud', Mock(return_value=None))
    statuses = []
    for stamp, row in ((100, POWDER), (100.1, POWDER), (100.2, None), (100.3, NANO)):
        scene.show(stamp, row)
        statuses.append(walker._move_toward().status)
    assert statuses == ['快轮等拾取物判稳', '快轮已点按F拾取', '快轮等列表重排', '快轮等拾取物判稳']
    scene.show(100.7, NANO)
    with pytest.raises(RuntimeError, match='完整一轮'):
        walker._move_toward()


def test_fast_round_off_near_beacon_while_holding_forward(scene: Scene) -> None:
    """按住前进走向 20px 内的途径点时不走快轮，避免冲过到点半径。"""
    walker = scene.walker
    scene.target('beacon')
    scene.worker.forward_step_active.return_value = False
    scene.worker.forward_held_seconds.return_value = 1.0
    scene.show(100, POWDER)
    walker.last_position = (15.0, 0.0)
    assert walker._pickup_fast_round(walker._current_point()) is None
    walker.last_position = (30.0, 0.0)
    assert walker._pickup_fast_round(walker._current_point()) is not None


def test_hud_throttled_when_located_but_forced_on_locate_fail(
    scene: Scene, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """定位成功时暂停菜单检查按 0.5 秒节流；定位失败那一轮不看节流，必查。"""
    walker = scene.walker
    find = Mock(return_value=Mock(is_success=False))
    monkeypatch.setattr(walker, 'round_by_find_area', find)
    checks = []
    for stamp, force in ((100, False), (100.2, False), (100.45, False), (100.5, False),
                         (100.6, True), (100.7, False), (101.1, False)):
        scene.show(stamp, None)
        walker._check_hud(force=force)
        checks.append(find.call_count)
    assert checks == [1, 1, 1, 2, 3, 3, 4]


def test_pause_menu_found_on_locate_fail_round(scene: Scene, monkeypatch: pytest.MonkeyPatch) -> None:
    """节流期内定位失败，当轮就查到暂停菜单并停下。"""
    walker = scene.walker
    scene.target('beacon')
    walker.last_position = (5.0, 0.0)
    walker.hud_checked_at = 100.0
    paused = Mock(is_success=True)
    find = Mock(return_value=paused)
    monkeypatch.setattr(walker, 'round_by_find_area', find)
    monkeypatch.setattr(walker, 'is_bagel_result', Mock(return_value=False))
    monkeypatch.setattr(walker, 'locate', Mock(return_value=None))
    scene.show(100.1, None)
    assert walker._move_toward().status == '移动中打开了暂停菜单'
    find.assert_called_once()


def test_short_step_waits_for_frame_after_rest(scene: Scene) -> None:
    """碎步只在松键加停顿之后拍的截图上迈步，主线轮次再快也不提前。"""
    walker = scene.walker
    scene.worker.forward_step_active.return_value = False
    scene.worker.forward_released_at = 100.0
    statuses = []
    for stamp in (100.01, 100.05, 100.09, 100.1):
        scene.show(stamp, None)
        statuses.append(walker._short_step().status)
    assert statuses == ['前进停顿中'] * 4
    scene.worker.step_forward.assert_not_called()
    scene.show(100.0 + STEP_REST_SECONDS + 0.005, None)
    assert walker._short_step().status == '短步前进'
    scene.worker.step_forward.assert_called_once_with(route_walker.SMALL_STEP_PRESS)


def test_box_f_not_retapped_while_panel_opening(scene: Scene) -> None:
    """面板开着时再按 F 是全部拾取：点按后 0.4 秒内只轮询，提示还在也不重按。"""
    box = PromptRow(555, 'box', '公文包')
    walker = scene.walker
    for stamp in (100, 100.1, 100.2, 100.3, 100.4):
        scene.show(stamp, box)
        walker._press_f('box')
    scene.worker.tap_interact.assert_called_once_with(INTERACT_TAP_SECONDS)
    for stamp in (100.6, 100.7):
        scene.show(stamp, box)
        walker._press_f('box')
    assert scene.worker.tap_interact.call_count == 2


def test_pick_all_rest_skips_empty_and_finishes_early_when_cleared(
    scene: Scene, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """结果区已空跳过全部拾取；有物时按一次 F，清空后提前放行，未清空到上限放行。"""
    walker = scene.walker
    walker.last_screenshot = np.zeros((1080, 1920, 3), np.uint8)
    walker._last_result_centers = (Point(1353, 256),)
    scene.panels.update({'搜查容器标题', '搜查完成'})
    occupied = [False]
    monkeypatch.setattr(route_walker, 'slot_occupied', lambda *_: occupied[0])

    # 结果区已空：直接跳过
    assert walker.pick_all_rest().status == '结果区已空，跳过'
    assert walker.pickup_presses == 0
    scene.worker.tap_interact.assert_not_called()

    # 有物 → 点按 F，0.10s 清空即提前放行
    occupied[0] = True
    scene.now = 200.0
    assert walker.pick_all_rest().result == OperationRoundResultEnum.WAIT
    assert walker.pickup_presses == 1
    scene.worker.tap_interact.assert_called_once_with(INTERACT_TAP_SECONDS)
    occupied[0] = False
    scene.now = 200.10
    assert walker.pick_all_rest().status == '全部拾取已清空'

    # 背包满一直未清空 → 满 0.4s 上限放行
    occupied[0] = True
    scene.now = 300.0
    assert walker.pick_all_rest().result == OperationRoundResultEnum.WAIT
    scene.now = 300.20
    assert walker.pick_all_rest().result == OperationRoundResultEnum.WAIT
    scene.now = 300.41
    assert walker.pick_all_rest().status == '已点按全部拾取'


def test_organize_in_panel_last_container_and_followup_pickup(
    scene: Scene, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非末容器跳过；末容器就地整理；后续途径点再拾取重新触发兜底；DEFEAT 透传。"""
    walker = scene.walker
    walker.organize_backpack = True
    walker.pickup_presses = 1
    walker.routes = [BagelRoute('r', '测试', 's', points=[
        BagelRoutePoint(0, 0, '测试/L', 'box'),
        BagelRoutePoint(10, 0, '测试/L', 'box'),
        BagelRoutePoint(20, 0, '测试/L', BEACON),
    ])]
    outcomes: list[OperationResult] = []
    calls: list[dict[str, object]] = []

    def fake_op(ctx, **kwargs):
        calls.append(kwargs)
        op = Mock()
        op.last_screenshot = None
        op.execute.return_value = outcomes.pop(0)
        return op

    monkeypatch.setattr(route_walker, 'BagelOrganizeBackpack', fake_op)

    # 第 1 个容器后面还有容器：不整理
    walker.cursor = 0
    assert walker.organize_in_panel().status == '无需容器内整理'
    assert calls == []

    # 第 2 个容器后面只有途径点：就地整理成功
    walker.cursor = 1
    outcomes.append(OperationResult(True, BagelOrganizeBackpack.STATUS_DONE))
    assert walker.organize_in_panel().status == '容器内背包整理完成'
    assert calls[-1]['in_search_panel'] is True
    assert not walker.needs_backpack_organize

    # 收尾途径点路上又捡了东西：退出前重新需要兜底整理
    walker.pickup_presses += 1
    assert walker.needs_backpack_organize

    # 末尾现场容器：整理失败保留兜底；DEFEAT 透传失败
    walker.cursor = 3
    outcomes.append(OperationResult(False, '搜查面板已关闭'))
    assert walker.organize_in_panel().status == '容器内整理跳过'
    assert walker.needs_backpack_organize

    outcomes.append(OperationResult(False, BagelOperation.STATUS_DEFEATED))
    defeat = walker.organize_in_panel()
    assert defeat.is_fail and defeat.status == BagelOperation.STATUS_DEFEATED


def test_organize_backpack_in_search_panel_and_empty_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """容器内模式不按 B/esc，失败直接透传；刚全部拾取后 0 件先复核一帧再跳过。"""
    ctx = Mock()
    clock = [100.0]
    monkeypatch.setattr(organize_backpack.time, 'monotonic', lambda: clock[0])
    op = BagelOrganizeBackpack(ctx, in_search_panel=True, pickup_at=99.9)
    op.last_screenshot = np.zeros((1080, 1920, 3), np.uint8)
    monkeypatch.setattr(op, 'is_bagel_result', lambda: False)

    # 面板已关闭：直接失败、不按 B/esc，关闭节点透传失败
    monkeypatch.setattr(op, '_has', lambda *_: False)
    opened = op.open_backpack()
    assert opened.is_fail
    op._previous_round_result = opened
    assert op.close_backpack().is_fail
    ctx.controller.btn_press.assert_not_called()

    # 面板开着、刚按过全部拾取且读到 0 件：先复核一帧再完成，结束不按 esc
    monkeypatch.setattr(op, '_has', lambda area, *_: area in {'搜查安全箱', '搜查容器标题'})
    monkeypatch.setattr(op, '_find_header', lambda: (480, 0))
    assert op.open_backpack().is_success
    assert op.locate_grid().result == OperationRoundResultEnum.WAIT
    done = op.locate_grid()
    assert done.is_success and done.status == op.STATUS_DONE
    op._previous_round_result = done
    monkeypatch.setattr(organize_backpack, 'record_safe_items', Mock())
    assert op.close_backpack().is_success
    ctx.controller.btn_press.assert_not_called()


def test_container_title_single_ocr_matches_all_variants() -> None:
    """「搜查容器标题」五个同坐标关键词合并为单次 OCR 识别。"""
    ctx = Mock()
    areas = {
        '搜查容器标题': Mock(pc_rect=(1290, 140, 1745, 195), color_range=None, text='箱', lcs_percent=0.7),
        '搜查容器标题_包': Mock(text='包', lcs_percent=0.7),
        '搜查容器标题_柜': Mock(text='柜', lcs_percent=0.7),
        '搜查容器标题_搜索物资': Mock(text='搜索物资', lcs_percent=0.6),
        '搜查容器标题_医疗站': Mock(text='医疗站', lcs_percent=0.6),
    }
    ctx.screen_loader.get_area.side_effect = lambda _screen, name: areas.get(name)
    ctx.ocr_service.get_ocr_result_list.return_value = [Mock(data='搜索物资')]
    op = BagelOperation(ctx, op_name='test')
    op.last_screenshot = np.zeros((1080, 1920, 3), np.uint8)
    assert op._has('搜查容器标题') is True
    assert ctx.ocr_service.get_ocr_result_list.call_count == 1
