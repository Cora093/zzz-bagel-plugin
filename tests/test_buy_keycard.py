"""自动买卡按仓库空位定张数（#91）：放不下的卡会进背包，入场核对零携带时停住。"""

from unittest.mock import Mock

import pytest

from bagel.operation.buy_keycard import buy_count_for_capacity
from bagel.operation.pick_keycard import BagelPickKeycard


@pytest.mark.parametrize(('capacity', 'expected'), [
    ((270, 280), 10),
    ((279, 280), 1),
    ((280, 280), 0),
    (None, 1),
])
def test_buy_count_capped_by_free_slots(capacity: tuple[int, int] | None, expected: int) -> None:
    assert buy_count_for_capacity(10, capacity) == expected


def test_no_card_switches_preset_filter_back_to_all_and_reports_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """进仓库时已是门禁卡筛选：没卡后仍切回「全部」，结果带上全仓占用。"""
    op = BagelPickKeycard(Mock(), '房间', None, from_prepare=False)
    op.result = op.round_fail(BagelPickKeycard.STATUS_NO_CARD)
    monkeypatch.setattr(op, '_tooltip_open', lambda: False)

    monkeypatch.setattr(op, '_header', lambda: '门禁卡(0/280)')
    assert op._restore_filter().status == '点击全部筛选'

    monkeypatch.setattr(op, '_header', lambda: '全部(279/280)')
    result = op._restore_filter()
    assert result.status == BagelPickKeycard.STATUS_NO_CARD
    assert result.data == (279, 280)
