"""导航诊断的格式契约：字段顺序、开关行为、离线解析器可解析。"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from bagel.diag import DIAG_ENV, DIAG_VERSION, diag, diag_enabled  # noqa: E402
from tests.tools.nav_metrics import parse_line, summarize  # noqa: E402

SAMPLE_LINE = '[22:04:25.123] [route_walker.py 1288] [INFO]: 贝果诊断 v1 · 点=3 距=12.4 差=-3.1 偏=2.2 位置=(830,351)'
"""固定 v1 样本：改格式前先确认老样本仍能解析。"""


@pytest.fixture
def captured(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """抓取 INFO 级别日志。"""
    caplog.set_level(logging.INFO)
    yield caplog


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """每个用例后恢复 BAGEL_DIAG。"""
    monkeypatch.delenv(DIAG_ENV, raising=False)
    yield


def test_disabled_by_default(captured: pytest.LogCaptureFixture) -> None:
    """未设开关时不输出任何诊断行。"""
    assert diag_enabled() is False
    diag({'点': 1, '距': '1.0'})
    assert [record for record in captured.records if '贝果诊断' in record.getMessage()] == []


def test_enabled_outputs_contracted_line(captured: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """设了开关就按契约顺序输出一行，且能被离线解析器解析。"""
    monkeypatch.setenv(DIAG_ENV, '1')
    diag({'点': 3, '距': '12.4', '差': '-3.1', '偏': '2.2', '位置': '(830,351)'})
    lines = [record.getMessage() for record in captured.records if '贝果诊断' in record.getMessage()]
    assert len(lines) == 1
    assert lines[0] == f'贝果诊断 {DIAG_VERSION} · 点=3 距=12.4 差=-3.1 偏=2.2 位置=(830,351)'
    parsed = parse_line(lines[0])
    assert parsed is not None
    assert parsed['点'] == '3' and parsed['偏'] == '2.2'


def test_fixed_v1_sample_still_parses() -> None:
    """固定样本必须一直可解析：字段改名或改序会让历史日志静默失效。"""
    parsed = parse_line(SAMPLE_LINE)
    assert parsed == {'点': '3', '距': '12.4', '差': '-3.1', '偏': '2.2', '位置': '(830,351)'}


def test_extra_fields_are_kept_and_ignored_by_summarize() -> None:
    """追加字段不影响老解析器：解析时保留，汇总时只取认识的键。"""
    parsed = parse_line(SAMPLE_LINE.replace('位置=(830,351)', '位置=(830,351) 墙=6.0'))
    assert parsed is not None
    assert parsed['偏'] == '2.2'
    assert '墙' in parsed


def test_unsupported_version_is_skipped() -> None:
    """未支持的版本整体跳过，不猜字段。"""
    assert parse_line(SAMPLE_LINE.replace('贝果诊断 v1', '贝果诊断 v2')) is None


def test_summarize_reports_offsets() -> None:
    """汇总只统计有数值的偏，>8px 按样本数算比例。"""
    rows = [parse_line(SAMPLE_LINE), parse_line(SAMPLE_LINE.replace('偏=2.2', '偏=?')),
            parse_line(SAMPLE_LINE.replace('偏=2.2', '偏=9.0'))]
    stats = summarize([row for row in rows if row is not None])
    assert stats['frames'] == 3
    assert stats['offset_samples'] == 2
    assert stats['over_8_ratio'] == 0.5
