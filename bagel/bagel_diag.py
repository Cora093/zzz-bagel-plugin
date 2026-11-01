"""贝果结构化诊断日志：每轮一条，供离线核对走位质量。

调用方必须自己控频率：只在每轮（约 8Hz）或按键变化时打，不要在 33Hz 的采样
线程里无脑打，日志 I/O 会拖慢主循环。
"""

from __future__ import annotations

from one_dragon.utils.log_utils import log

DIAG_ENABLED: bool = True
"""结构化诊断开关：关掉就完全不产出「贝果诊断」行。"""


def diag(text: str) -> None:
    """打一条结构化诊断日志；时间戳由日志器给。"""
    if DIAG_ENABLED:
        log.info('贝果诊断 · %s', text)
