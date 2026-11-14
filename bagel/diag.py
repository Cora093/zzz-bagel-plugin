"""导航诊断输出：固定键值对格式，供离线解析（默认关闭）。

格式契约（只追加字段、不改名、不改顺序、带版本前缀）::

    贝果诊断 v1 · 点=3 距=12.4 差=-3.1 偏=2.2 位置=(830,351)

- 设 ``BAGEL_DIAG=1`` 才输出，不采样、每轮一条；未设时一行都不输出。
- 新增字段一律追加在末尾，解析端只认自己认识的键；改字段名或顺序才升版本。
"""

from __future__ import annotations

import os
from typing import Any

from one_dragon.utils.log_utils import log

DIAG_VERSION: str = 'v1'
"""格式版本：只追加字段保持不变，改字段名/顺序才升版本。"""

DIAG_ENV: str = 'BAGEL_DIAG'
"""开关环境变量；这是自测用的，不进用户配置页。"""

_OFF_VALUES: frozenset[str] = frozenset({'', '0', 'false', 'off', 'no'})


def diag_enabled() -> bool:
    """是否输出导航诊断；每次现读环境变量，便于测试切换。"""
    return os.environ.get(DIAG_ENV, '').strip().lower() not in _OFF_VALUES


def diag(fields: dict[str, Any]) -> None:
    """按契约打一条导航诊断；未启用时不做任何事。

    Args:
        fields: 有序键值对，值已格式化成字符串（保留两位小数的由调用方负责）。
    """
    if not diag_enabled():
        return
    body = ' '.join(f'{key}={value}' for key, value in fields.items())
    log.info('贝果诊断 %s · %s', DIAG_VERSION, body)
