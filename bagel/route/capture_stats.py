"""路线截图计数。按决策轮归段，每秒和路线结束时汇总。"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import psutil
import win32process
from one_dragon.utils.log_utils import log
from pywintypes import error as WinError


def movement_stage(node: str, status: str) -> str:
    """按本轮结束时的决策，把时间与该轮两路截图记在同一段。"""
    if '停车等转向' in status:
        return '停车转向'
    if status.startswith(('走向电梯', '朝电梯转向')):
        return '走向电梯'
    if status in ('冲刺前进', '按住前进'):
        return '直线冲刺'
    if node == '碎步移动' or status in (
        '短步前进', '慢步前进', '等待本次前进短按结束', '等待前进短按可用', '等待松键', '前进停顿中',
    ):
        return '碎步慢步'
    return '其他'


@dataclass
class _Bucket:
    seconds: float = 0.0
    counts: Counter[str] = field(default_factory=Counter)
    durations: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))

    def merge(self, other: _Bucket) -> None:
        self.seconds += other.seconds
        self.counts.update(other.counts)
        for name, values in other.durations.items():
            self.durations[name].update(values)

    def report(self) -> dict[str, object]:
        result: dict[str, object] = {'秒': round(self.seconds, 4), **self.counts}
        for name, values in self.durations.items():
            ranks = [math.ceil(values.total() * q) for q in (0.50, 0.95)]
            percentiles: list[int] = []
            count = 0
            for ms, times in sorted(values.items()):
                count += times
                while len(percentiles) < 2 and count >= ranks[len(percentiles)]:
                    percentiles.append(ms)
            result[name] = percentiles
            if name == '无样本间隔ms':
                result['最长无样本ms'] = max(values)
        return result


class CaptureStats:
    """两条线程只累计内存数据。暂停排除在计时之外，CPU 按整个窗口计。"""

    def __init__(self, get_hwnd: Callable[[], int | None]) -> None:
        self._get_hwnd: Callable[[], int | None] = get_hwnd
        self._lock: threading.Lock = threading.Lock()
        self._round: _Bucket = _Bucket()
        self._window: dict[str, _Bucket] = defaultdict(_Bucket)
        self._total: dict[str, _Bucket] = defaultdict(_Bucket)
        self._start: float | None = None
        self._round_start: float = 0.0
        self._game: psutil.Process | None = None
        self._cpu: tuple[float | None, float | None] = (None, None)
        self._main_sample_at: float | None = None

    def _cpu_times(self) -> tuple[float, float | None]:
        game_cpu = None
        try:
            if self._game is None:
                hwnd = self._get_hwnd()
                if hwnd:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    if pid:
                        self._game = psutil.Process(pid)
            if self._game is not None:
                times = self._game.cpu_times()
                game_cpu = times.user + times.system
        except (psutil.Error, WinError, OSError):
            self._game = None
        return time.process_time(), game_cpu

    def start(self) -> None:
        with self._lock:
            if self._start is None:
                self._start = self._round_start = time.monotonic()
                self._cpu = self._cpu_times() if log.isEnabledFor(logging.DEBUG) else (None, None)

    def record(self, name: str, seconds: float | None = None) -> None:
        with self._lock:
            if self._start is None:
                return
            self._round.counts[name] += 1
            if seconds is not None:
                self._round.durations[name + 'ms'][round(seconds * 1000)] += 1

    def main_sample(self, stamp: float) -> None:
        with self._lock:
            if self._start is None or self._main_sample_at == stamp:
                return
            if self._main_sample_at is not None:
                ms = round((stamp - self._main_sample_at) * 1000)
                self._round.durations['主样本间隔ms'][ms] += 1
            self._main_sample_at = stamp

    def finish_round(self, stage: str, *, close: bool = False) -> None:
        with self._lock:
            if self._start is None:
                return
            now = time.monotonic()
            self._round.seconds = now - self._round_start
            self._window[stage].merge(self._round)
            self._round = _Bucket()
            self._round_start = now
            elapsed = now - self._start
            if elapsed < 1 and not close:
                return
            debug_window = log.isEnabledFor(logging.DEBUG)
            previous_cpu = self._cpu
            current_cpu = self._cpu_times() if debug_window else (None, None)
            self._cpu = current_cpu
            window = self._window
            self._window = defaultdict(_Bucket)
            for name, bucket in window.items():
                self._total[name].merge(bucket)
            self._start = None if close else now
            if close:
                self._main_sample_at = None
        if debug_window:
            cores = [
                round((b - a) / elapsed, 3) if a is not None and b is not None and elapsed > 0 else None
                for a, b in zip(previous_cpu, current_cpu, strict=True)
            ]
            report = {'插件核': cores[0], '游戏核': cores[1], '段': self._stage_reports(window)}
            log.debug('贝果截图统计 窗口 %s', json.dumps(report, ensure_ascii=False))

    def finish_route(self) -> None:
        self.finish_round('其他', close=True)
        with self._lock:
            total = self._total
            self._total = defaultdict(_Bucket)
        if total and log.isEnabledFor(logging.INFO):
            report = {'段': self._stage_reports(total)}
            log.info('贝果截图统计 全程 %s', json.dumps(report, ensure_ascii=False))

    @staticmethod
    def _stage_reports(data: dict[str, _Bucket]) -> dict[str, dict[str, object]]:
        return {stage: bucket.report() for stage, bucket in data.items()}
