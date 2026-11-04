"""贝果现场截图：写到插件数据目录，不占用主仓的调试图片目录。"""

from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
from cv2.typing import MatLike
from one_dragon.utils.log_utils import log

from .defs import get_plugin_data_dir

RETENTION_DAYS: int = 3
"""现场截图保留天数；与主程序日志的 3 份轮转备份大致同步，超期的图也对不上日志。"""

MAX_TOTAL_BYTES: int = 500 * 1024 * 1024
"""现场截图总大小上限；失败集中时保留期内也可能超限，超限从最旧的开始删。"""

_SCREENSHOT_NAME: re.Pattern[str] = re.compile(r'.+_(\d+)\.png')
"""程序写入的文件名，分组 1 是毫秒时间戳；不匹配的文件不属于程序，清理时不动。"""


def screenshot_dir() -> Path:
    """现场截图目录；跟路线数据同在插件数据目录下。

    Returns:
        工作目录下的 ``plugin_data/<APP_ID>/screenshots``。
    """
    return get_plugin_data_dir() / 'screenshots'


def save_screenshot(image: MatLike | None, prefix: str) -> Path | None:
    """把现场截图写到插件数据目录。

    Args:
        image: RGB 截图；None 表示没有可保存的画面。
        prefix: 文件名前缀，通常取操作名。

    Returns:
        保存的文件路径；没有截图或写入失败时返回 None。
    """
    if image is None:
        return None
    directory = screenshot_dir()
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


def prune_screenshots(now: float | None = None) -> None:
    """删除超过保留期的现场截图，总大小仍超限时继续从最旧的开始删；出错只记警告。

    Args:
        now: 当前时间戳（秒）；None 表示本机当前时间。
    """
    try:
        directory = screenshot_dir()
        if not directory.is_dir():
            return
        shots: list[tuple[int, int, Path]] = []
        for path in directory.iterdir():
            match = _SCREENSHOT_NAME.fullmatch(path.name)
            if match is not None and path.is_file():
                shots.append((int(match.group(1)), path.stat().st_size, path))
    except OSError:
        log.warning('读取贝果现场截图目录失败，本次不清理', exc_info=True)
        return

    shots.sort(key=lambda shot: shot[0])
    cutoff_ms = ((time.time() if now is None else now) - RETENTION_DAYS * 86400) * 1000
    total = sum(size for _, size, _ in shots)
    removed = 0
    for taken_ms, size, path in shots:
        if taken_ms >= cutoff_ms and total <= MAX_TOTAL_BYTES:
            break
        try:
            path.unlink()
        except OSError:
            # 图片可能正被看图软件占用；跳过这一张，下次任务启动再删。
            log.warning('删除贝果现场截图失败：%s', path, exc_info=True)
            continue
        total -= size
        removed += 1
    if removed > 0:
        log.info('贝果现场截图：删除 %s 张过期或超限的截图', removed)
