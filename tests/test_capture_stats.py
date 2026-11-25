"""核对截图频率、分位数和暂停时长的统计口径。"""

import json
import logging
from unittest.mock import Mock

import pytest

from bagel.route import capture_stats


@pytest.mark.parametrize('level', [logging.INFO, logging.DEBUG])
def test_segment_totals_exclude_pause(monkeypatch: pytest.MonkeyPatch, level: int) -> None:
    clock = Mock()
    clock.monotonic.return_value = 10.0
    monkeypatch.setattr(capture_stats, 'time', clock)
    stats = capture_stats.CaptureStats(lambda: None)
    cpu_times = Mock(return_value=(0.0, None))
    monkeypatch.setattr(stats, '_cpu_times', cpu_times)
    info = Mock()
    debug = Mock()
    monkeypatch.setattr(capture_stats.log, 'isEnabledFor', lambda requested: requested >= level)
    monkeypatch.setattr(capture_stats.log, 'info', info)
    monkeypatch.setattr(capture_stats.log, 'debug', debug)
    stats.start()
    stats.record('主线截图', 0.015)
    clock.monotonic.return_value = 10.2
    stats.finish_round('停车转向')
    stats.record('转向截图', 0.025)
    clock.monotonic.return_value = 11.2
    stats.finish_round('直线冲刺')
    info.assert_not_called()
    if level == logging.DEBUG:
        assert debug.call_args.args[0] == '贝果截图统计 窗口 %s'
        assert json.loads(debug.call_args.args[-1])['游戏核'] is None
    stats.finish_round('其他', close=True)
    clock.monotonic.return_value = 20.0
    stats.start()
    stats.record('主线截图', 0.035)
    clock.monotonic.return_value = 20.3
    stats.finish_round('停车转向')
    stats.finish_route()
    if level == logging.INFO:
        debug.assert_not_called()
        cpu_times.assert_not_called()
    assert info.call_count == 1
    assert info.call_args.args[0] == '贝果截图统计 全程 %s'
    total = json.loads(info.call_args.args[-1])['段']
    assert total['停车转向'] == {'秒': 0.5, '主线截图': 2, '主线截图ms': [15, 35]}
    assert total['直线冲刺']['转向截图'] == 1
