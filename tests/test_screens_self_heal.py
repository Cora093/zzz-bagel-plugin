"""贝果画面自愈：被其他插件的画面重载冲掉后要能补回来。

两个用例都只依赖 ScreenLoader 的公开行为：一个验证「缺失时补回」，一个验证
「已在时不重复注入」。不依赖具体框架版本是否保留插件画面（主仓 #3021 修好之后
重载本身就会保留，此时缺失只能通过不注入来构造）。
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from one_dragon.base.screen.screen_loader import ScreenContext
from one_dragon.utils import os_utils

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from bagel.screens import ensure_screens  # noqa: E402

SENTINEL = ('贝果-研究站', '前往空洞')
SCREEN_DIR = str(REPO_DIR / 'bagel' / 'screen_info')


@pytest.fixture
def empty_work_dir(tmp_path: Path) -> Iterator[Path]:
    """空工作目录：插件画面只能来自插件自己的 screen_info。"""
    original = os_utils.get_work_dir()
    os_utils.set_work_dir(tmp_path)
    try:
        yield tmp_path
    finally:
        os_utils.set_work_dir(original)


def test_ensure_screens_reinjects_when_missing(empty_work_dir: Path) -> None:
    """画面缺失时补回：注入前哨兵区域不存在，补回后 9 个画面都在。"""
    loader = ScreenContext()
    loader.reload()
    assert loader.get_area(*SENTINEL) is None

    assert ensure_screens(loader) is True
    assert loader.get_area(*SENTINEL) is not None
    assert sum(1 for name in loader.screen_info_map if name.startswith('贝果-')) == 9


def test_ensure_screens_is_noop_when_present(empty_work_dir: Path) -> None:
    """已在时不重复注入：返回 False，画面数不变。"""
    loader = ScreenContext()
    loader.reload()
    loader.load_extra_screen_dir(SCREEN_DIR, default_app_id='bagel')
    before = len(loader.screen_info_map)
    assert loader.get_area(*SENTINEL) is not None

    assert ensure_screens(loader) is False
    assert len(loader.screen_info_map) == before
