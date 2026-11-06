"""贝果画面自愈：被其他插件的画面重载冲掉后要能补回来。

背景：别的插件会调 ``screen_loader.reload(from_separated_files=True)``，旧框架只重读
工作目录下的分离文件，注入式画面（本插件 ``screen_info/``）会被清空。主仓修好之后
这种重载会自己保留插件画面，所以断言写成「重载之后哨兵区域必须还在」。
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


@pytest.fixture
def empty_work_dir(tmp_path: Path) -> Iterator[Path]:
    """空工作目录：插件画面只能来自插件自己的 screen_info。"""
    original = os_utils.get_work_dir()
    os_utils.set_work_dir(tmp_path)
    try:
        yield tmp_path
    finally:
        os_utils.set_work_dir(original)


def test_ensure_screens_reinjects_after_reload(empty_work_dir: Path) -> None:
    """重载之后画面必须还在；缺了就补回来，补过之后不再重复注入。"""
    loader = ScreenContext()
    loader.reload()
    loader.load_extra_screen_dir(str(REPO_DIR / 'bagel' / 'screen_info'), default_app_id='bagel')
    assert loader.get_area(*SENTINEL) is not None

    loader.reload(from_separated_files=True)
    survived = loader.get_area(*SENTINEL) is not None

    assert ensure_screens(loader) is (not survived)
    assert loader.get_area(*SENTINEL) is not None
    assert ensure_screens(loader) is False
