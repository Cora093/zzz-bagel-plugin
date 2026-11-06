"""测试公共配置：插件以仓库根为导入根，临时目录固定到仓库内。

本机 ``%TEMP%`` 下的 pytest 目录可能因 ACL 拒绝访问而让 ``tmp_path`` 直接报错，
统一指到仓库内的 ``.pytest_tmp``（已 gitignore）后行为与作者环境一致。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
TEMP_DIR = REPO_DIR / '.pytest_tmp'


def pytest_configure(config) -> None:  # noqa: ANN001
    """设置导入路径与临时目录；`--basetemp` 显式传入时不覆盖。"""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(TEMP_DIR)
    if not getattr(config.option, 'basetemp', None):
        config.option.basetemp = str(TEMP_DIR / 'basetemp')
    entry = str(REPO_DIR)
    if entry not in sys.path:
        sys.path.insert(0, entry)
