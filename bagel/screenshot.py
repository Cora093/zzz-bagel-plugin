"""贝果现场截图：写到插件数据目录，不占用主仓的调试图片目录。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from cv2.typing import MatLike
from one_dragon.utils.log_utils import log

from .defs import get_plugin_data_dir

if TYPE_CHECKING:
    from zzz_od.context.zzz_context import ZContext


def screenshot_dir(ctx: ZContext) -> Path:
    """现场截图目录；跟路线数据同在插件数据目录下。

    Args:
        ctx: 运行上下文，用来取插件数据目录。

    Returns:
        插件数据目录下的 ``screenshots``。
    """
    return get_plugin_data_dir(ctx) / 'screenshots'


def save_screenshot(ctx: ZContext, image: MatLike | None, prefix: str) -> Path | None:
    """把现场截图写到插件数据目录。

    Args:
        ctx: 运行上下文，用来取插件数据目录。
        image: RGB 截图；None 表示没有可保存的画面。
        prefix: 文件名前缀，通常取操作名。

    Returns:
        保存的文件路径；没有截图或写入失败时返回 None。
    """
    if image is None:
        return None
    directory = screenshot_dir(ctx)
    path = directory / f'{prefix}_{round(time.time() * 1000)}.png'
    try:
        directory.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except (OSError, cv2.error):
        log.warning('保存贝果现场截图失败', exc_info=True)
        return None
    if not written:
        log.warning('保存贝果现场截图失败：%s', path)
        return None
    return path
