from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from one_dragon.base.config.yaml_operator import invalidate_cache
from one_dragon.base.operation.application_run_record import AppRunRecord
from one_dragon.utils.log_utils import log

from .const import APP_ID


class BagelRunRecord(AppRunRecord):
    """保存应用运行状态，写入时原子替换文件。"""

    def __init__(self, instance_idx: int, game_refresh_hour_offset: int = 0) -> None:
        """加载记录；坏文件由基类回落为空记录，旧版残留的物资记录直接丢弃。"""
        super().__init__(APP_ID, instance_idx, game_refresh_hour_offset)
        if not isinstance(self.data, dict):
            log.warning('贝果运行记录不是键值格式，按空记录处理：%s', self.file_path)
            self.data = {}
        self.data.pop('collection', None)

    def save(self) -> None:
        """原子替换记录，写入失败时保留原文件并向调用者报错。"""
        if not self._ensure_write_path_ready():
            return
        write_path = self._get_write_path()
        if write_path is None:
            return
        target = Path(write_path)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=target.parent,
                prefix=f'{target.name}.',
                suffix='.tmp',
                delete=False,
            ) as file:
                temporary = Path(file.name)
                yaml.safe_dump(self.data, file, allow_unicode=True, sort_keys=False)
            temporary.replace(target)
            invalidate_cache(str(target))
        except OSError:
            log.error('保存贝果运行记录失败', exc_info=True)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
