"""整理背包一次算出计划：每个安全箱格最多拖一次，件数读错也不来回换（#120）。"""

from collections import Counter

from one_dragon.base.geometry.point import Point

from bagel.vision.item_vision import (
    ACTION_FILL,
    ACTION_SWAP,
    ITEM_TYPE_MATERIAL,
    BagelSlotMark,
    plan_store_moves,
)

_BACKPACK_Y: int = 0
_SAFE_Y: int = 1


def slot(index: int, quality: str, in_safe: bool = False, item_type: str = ITEM_TYPE_MATERIAL) -> BagelSlotMark:
    """格心纵坐标区分背包和安全箱，件数表按（纵坐标, 下标）取。"""
    return BagelSlotMark(index, Point(0, _SAFE_Y if in_safe else _BACKPACK_Y), quality, item_type)


def plan(
    backpack: list[BagelSlotMark],
    safes: list[BagelSlotMark],
    empty: list[int],
    counts: dict[tuple[int, int], int],
) -> tuple[list[tuple[str, int, int]], Counter]:
    """返回（计划步骤, 每格读件数次数）。"""
    reads: Counter = Counter()

    def quantity(mark: BagelSlotMark) -> int | None:
        key = (mark.center.y, mark.index)
        reads[key] += 1
        return counts.get(key)

    steps = plan_store_moves(backpack, safes, empty, True, quantity)
    return [(step.kind, step.source_index, step.dest_index) for step in steps], reads


def test_misread_count_swaps_once() -> None:
    """#120 现场：背包那件 1 件读成 11，换进箱后读回 1；计划只换一次，每格只读一次。"""
    steps, reads = plan(
        [slot(1, 'A')],
        [slot(0, 'S', True), slot(4, 'A', True)],
        [],
        {(_BACKPACK_Y, 1): 11, (_SAFE_Y, 4): 2},
    )
    assert steps == [(ACTION_SWAP, 1, 4)]
    assert set(reads.values()) == {1}


def test_top_items_by_count_and_unknown_kept() -> None:
    """同档按件数取前几：最多的两件换掉最少的两格；读不出件数的箱内格不换出。"""
    steps, _ = plan(
        [slot(0, 'A'), slot(1, 'A'), slot(2, 'A')],
        [slot(0, 'A', True), slot(1, 'A', True), slot(2, 'A', True)],
        [],
        {(_BACKPACK_Y, 0): 3, (_BACKPACK_Y, 1): 7, (_BACKPACK_Y, 2): 5, (_SAFE_Y, 0): 4, (_SAFE_Y, 1): 1},
    )
    assert steps == [(ACTION_SWAP, 1, 1), (ACTION_SWAP, 2, 0)]


def test_swaps_first_then_fills_from_back() -> None:
    """先对换，再按背包格从后往前填空；最好的两件填空，第三件按优先级换掉箱内最差一格。"""
    steps, _ = plan(
        [slot(0, 'S'), slot(3, 'A'), slot(5, 'Z')],
        [slot(0, 'B', True)],
        [1, 2],
        {},
    )
    assert steps == [(ACTION_SWAP, 3, 0), (ACTION_FILL, 5, 1), (ACTION_FILL, 0, 2)]
