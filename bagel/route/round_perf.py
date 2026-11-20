"""walker 单轮分段计时与决策日志合并。只写日志，不参与决策。"""

from __future__ import annotations

import statistics
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

REPEAT_FLUSH_SECONDS: float = 1.0
"""同一条决策连续重复时，最多隔这么久（秒）打一次汇总并重打一条完整行。

卡住的表现正是同一条反复出现，合并不能把它藏掉：每秒至少留一条完整行和一条汇总。
"""


class RoundPerf:
    """记一轮内各段的累计耗时与次数。

    各段不能嵌套计时，否则重叠部分会算两次。没有计时的部分统一记为「其他」。
    用 ``perf_counter``，不受测试里替换的 ``time`` 模块影响。
    """

    def __init__(self) -> None:
        self.started: float = perf_counter()
        """本轮开始的时刻。"""
        self.parts: dict[str, float] = {}
        """段名到累计耗时（秒），按首次出现的顺序排列。"""
        self.counts: dict[str, int] = {}
        """段名到本轮出现次数。"""

    def start(self) -> None:
        """开始新一轮，清空上一轮的记录。"""
        self.started = perf_counter()
        self.parts.clear()
        self.counts.clear()

    def add(self, name: str, seconds: float) -> None:
        """累加一段耗时。

        Args:
            name: 段名，如「定位」「OCR」。
            seconds: 这一次的耗时（秒）。
        """
        self.parts[name] = self.parts.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """给 with 块计时，块内抛异常也照记。

        Args:
            name: 段名。
        """
        start = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - start)

    def total(self) -> float:
        """本轮到现在的总耗时（秒）。"""
        return perf_counter() - self.started

    def summary(self) -> str:
        """一行文字：总耗时和各段毫秒数，出现多次的段带次数。"""
        total = self.total()
        parts = [
            f'{name}×{self.counts[name]} {seconds * 1000:.0f}' if self.counts[name] > 1
            else f'{name} {seconds * 1000:.0f}'
            for name, seconds in self.parts.items()
        ]
        parts.append(f'其他 {(total - sum(self.parts.values())) * 1000:.0f}')
        return f'用时 {total * 1000:.0f}ms（{" ".join(parts)}）'


class RepeatMerger:
    """把连续相同的决策行合并成「首行 + 汇总」。

    同一个键第一次出现时照打完整行；之后连续出现的只累计，直到键变了或满
    ``REPEAT_FLUSH_SECONDS``，再打一条汇总：轮数、时长、最后一轮的文字、
    总用时的中位和最大值、各段平均耗时。满时长后当前这一轮照打完整行，重新开始累计。
    """

    def __init__(self, flush_seconds: float = REPEAT_FLUSH_SECONDS) -> None:
        self.flush_seconds: float = flush_seconds
        self.key: str | None = None
        """当前累计中的键；None 表示没有可合并的上一行。"""
        self.first_at: float = 0.0
        """本次累计窗口的起始时刻（perf_counter）。"""
        self.count: int = 0
        """已被合并、没打出来的轮数。"""
        self.totals: list[float] = []
        """被合并各轮的总用时（秒）。"""
        self.parts: dict[str, float] = {}
        """被合并各轮的分段耗时之和（秒）。"""
        self.last_text: str = ''
        """被合并的最后一轮的文字，不含用时。"""

    def submit(self, key: str | None, text: str, perf: RoundPerf) -> list[str]:
        """提交一轮的决策文字，返回这一轮要打出的行（可能为空）。

        Args:
            key: 合并用的键，相同键视为同一条决策；None 表示这一轮不合并。
            text: 决策文字，不含用时。
            perf: 本轮分段计时。
        """
        now = perf_counter()
        if key is not None and key == self.key and now - self.first_at < self.flush_seconds:
            self.count += 1
            self.totals.append(perf.total())
            for name, seconds in perf.parts.items():
                self.parts[name] = self.parts.get(name, 0.0) + seconds
            self.last_text = text
            return []
        lines = self.flush()
        self.key = key
        self.first_at = now
        lines.append(f'{text} · {perf.summary()}')
        return lines

    def flush(self) -> list[str]:
        """打出尚未汇总的合并轮次；没有时返回空列表。"""
        if self.count == 0:
            self.key = None
            return []
        rounds = self.count
        totals = self.totals
        parts = ' '.join(f'{name} {seconds / rounds * 1000:.0f}' for name, seconds in self.parts.items())
        line = (
            f'↑ 同上 {rounds} 轮 {perf_counter() - self.first_at:.2f}s · 最后：{self.last_text} · '
            f'用时中位 {statistics.median(totals) * 1000:.0f}ms 最大 {max(totals) * 1000:.0f}ms'
            f'（各段平均 {parts}）'
        )
        self.key = None
        self.count = 0
        self.totals = []
        self.parts = {}
        self.last_text = ''
        return [line]
