"""离线解析「贝果诊断」日志行，输出走偏等指标的分布。

用法::

    python tests/tools/nav_metrics.py <日志文件> [<日志文件> ...]

格式契约见 ``bagel/diag.py``：只认认识的键，遇到新键忽略，遇到不认识的版本整体跳过。
"""

from __future__ import annotations

import re
import sys
from statistics import median

LINE_PATTERN = re.compile(r'贝果诊断 (?P<version>v\d+) · (?P<body>.+)$')
SUPPORTED_VERSIONS = {'v1'}


def parse_line(line: str) -> dict[str, str] | None:
    """把一行诊断日志解析成键值对；不是诊断行或不支持的版本返回 None。"""
    match = LINE_PATTERN.search(line)
    if match is None or match.group('version') not in SUPPORTED_VERSIONS:
        return None
    fields: dict[str, str] = {}
    for token in match.group('body').split():
        key, sep, value = token.partition('=')
        if not sep or not key:
            continue
        fields[key] = value
    return fields


def quantile(values: list[float], ratio: float) -> float:
    """取分位数（最近秩法）；空样本返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[index]


def _float(fields: dict[str, str], key: str) -> float | None:
    raw = fields.get(key)
    if raw is None or raw == '?':
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    """汇总指标：样本数、偏的分位数、>8px 占比、涉及点位。"""
    offsets = [value for row in rows if (value := _float(row, '偏')) is not None]
    points = {row.get('点') for row in rows if row.get('点')}
    return {
        'frames': len(rows),
        'offset_samples': len(offsets),
        'p50': median(offsets) if offsets else 0.0,
        'p75': quantile(offsets, 0.75),
        'p90': quantile(offsets, 0.90),
        'max': max(offsets) if offsets else 0.0,
        'over_8_ratio': (sum(1 for value in offsets if value > 8.0) / len(offsets)) if offsets else 0.0,
        'points': sorted(points, key=lambda item: int(item) if item.isdigit() else 0),
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        rows: list[dict[str, str]] = []
        with open(path, encoding='utf-8', errors='replace') as file:
            for line in file:
                fields = parse_line(line)
                if fields is not None:
                    rows.append(fields)
        stats = summarize(rows)
        print(f'== {path}')
        print(f'  帧：{stats["frames"]}  偏样本：{stats["offset_samples"]} '
              f'P50 {stats["p50"]:.1f}  P75 {stats["p75"]:.1f}  P90 {stats["p90"]:.1f}  '
              f'max {stats["max"]:.1f}  >8px {stats["over_8_ratio"] * 100:.1f}%')
        print(f'  点位：{", ".join(stats["points"]) or "无"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
