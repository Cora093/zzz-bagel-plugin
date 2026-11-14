from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from one_dragon.utils.log_utils import log

from .const import APP_ID
from .defs import PLUGIN_DIR

if TYPE_CHECKING:
    from one_dragon.base.screen.screen_loader import ScreenContext

SCREEN_DIR: Path = PLUGIN_DIR / 'screen_info'
"""插件画面定义目录，框架启动时从这里注入。"""

SENTINEL_SCREEN: tuple[str, str] = ('贝果-研究站', '前往空洞')
"""用来判断插件画面还在不在；这个区域每次入场都要用。"""


def ensure_screens(screen_loader: ScreenContext) -> bool:
    """插件画面被别的插件重载冲掉时重新注入；返回是否真的重注入了。

    其他插件调用 ``screen_loader.reload(from_separated_files=True)`` 时只会重读
    工作目录下的分离文件，注入式画面（本插件的 ``screen_info/``）会被清空，
    之后一律报「区域未配置」。
    """
    if screen_loader.get_area(*SENTINEL_SCREEN) is not None:
        return False
    screen_loader.load_extra_screen_dir(str(SCREEN_DIR), default_app_id=APP_ID)
    if screen_loader.get_area(*SENTINEL_SCREEN) is None:
        log.error('贝果画面定义缺失，重新注入后仍然没有：%s', SCREEN_DIR)
    else:
        log.warning('贝果画面定义被其他插件的画面重载冲掉，已重新注入')
    return True
