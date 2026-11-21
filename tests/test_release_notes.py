"""测试从 CHANGELOG.md 提取 GitHub Release 说明的脚本逻辑。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_DIR / '.github' / 'scripts' / 'extract_release_notes.py'
CHANGELOG_PATH = REPO_DIR / 'CHANGELOG.md'


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('extract_release_notes', SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_prefers_summary_block() -> None:
    """存在 ### 概要 时，只提取概要并附带完整 CHANGELOG 链接。"""
    module = _load_module()
    sample = (
        '# 更新日志\n\n'
        '## 未发布\n\n'
        '- 草稿条目\n\n'
        '## 0.6.0 — 2026-11-21\n\n'
        '### 概要\n\n'
        '#### 新增\n'
        '- 支持滚轮选行。\n\n'
        '#### 修复\n'
        '- 修正超时判定。\n\n'
        '### 新增\n\n'
        '- 内部详细说明\n\n'
        '## 0.5.0 — 2026-11-16\n\n'
        '- 上一版本\n'
    )
    notes = module.extract_release_notes(
        sample,
        tag='v0.6.0',
        repo='Cora093/zzz-bagel-plugin',
    )
    assert notes == (
        '#### 新增\n'
        '- 支持滚轮选行。\n\n'
        '#### 修复\n'
        '- 修正超时判定。\n\n'
        '完整更新记录见 [CHANGELOG.md]'
        '(https://github.com/Cora093/zzz-bagel-plugin/blob/v0.6.0/CHANGELOG.md)。\n'
    )


def test_extract_falls_back_to_full_section() -> None:
    """没有 ### 概要 时，提取该版本的整段内容并止于下一版本标题。"""
    module = _load_module()
    sample = (
        '## 0.3.4 — 2026-11-07\n\n'
        '### 变更\n\n'
        '- 优化默认路线。\n\n'
        '### 修复\n\n'
        '- 修复页签识别。\n\n'
        '## 0.3.3 — 2026-11-04\n\n'
        '- 更早版本\n'
    )
    notes = module.extract_release_notes(sample, tag='v0.3.4', repo='Cora093/zzz-bagel-plugin')
    assert notes == (
        '### 变更\n\n'
        '- 优化默认路线。\n\n'
        '### 修复\n\n'
        '- 修复页签识别。\n'
    )


def test_extract_does_not_match_prerelease_prefix() -> None:
    """版本号 0.5.0 不得误匹配 0.5.0-beta。"""
    module = _load_module()
    sample = (
        '## 0.5.0-beta — 2026-11-15\n\n'
        '- 测试版\n'
    )
    assert module.extract_release_notes(sample, tag='v0.5.0') is None


def test_extract_against_repo_changelog() -> None:
    """能正确提取仓库 CHANGELOG.md 中既有版本的记录，未知版本返回 None。"""
    module = _load_module()
    text = CHANGELOG_PATH.read_text(encoding='utf-8')

    notes_060 = module.extract_release_notes(
        text, tag='v0.6.0', repo='Cora093/zzz-bagel-plugin'
    )
    assert notes_060 is not None
    assert '### 新增' in notes_060
    assert '### 修复' in notes_060
    assert '## 0.5.0' not in notes_060
    assert '140ms' not in notes_060

    notes_034 = module.extract_release_notes(text, tag='v0.3.4')
    assert notes_034 is not None
    assert '优化「白鸽工地站-默认」路线' in notes_034
    assert '## 0.3.3' not in notes_034

    assert module.extract_release_notes(text, tag='v99.99.99') is None
