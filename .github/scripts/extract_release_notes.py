"""从 CHANGELOG.md 提取指定 tag 对应的 GitHub Release 说明。

优先读取该版本段落内的 ``### 概要`` 小节，供插件商店展示简洁的用户向更新日志；
若未单独编写 ``### 概要``，则回退为该版本的整段内容。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SUMMARY_HEADING_PATTERN = re.compile(r'^###\s+概要\s*$')


def _find_version_section(lines: list[str], version: str) -> list[str] | None:
    """定位 ``## <version>`` 段落的正文行，未找到返回 None。"""
    heading_pattern = re.compile(rf'^##\s+{re.escape(version)}(?:\s|$|—)')
    start_idx: int | None = None
    for index, line in enumerate(lines):
        if heading_pattern.match(line):
            start_idx = index + 1
            break
    if start_idx is None:
        return None

    end_idx = len(lines)
    for index in range(start_idx, len(lines)):
        if lines[index].startswith('## '):
            end_idx = index
            break
    return lines[start_idx:end_idx]


def _extract_summary_block(section_lines: list[str]) -> str | None:
    """从版本段落中提取 ``### 概要`` 的正文；不存在或为空时返回 None。"""
    start_idx: int | None = None
    for index, line in enumerate(section_lines):
        if SUMMARY_HEADING_PATTERN.match(line):
            start_idx = index + 1
            break
    if start_idx is None:
        return None

    end_idx = len(section_lines)
    for index in range(start_idx, len(section_lines)):
        if section_lines[index].startswith('### '):
            end_idx = index
            break

    body = '\n'.join(section_lines[start_idx:end_idx]).strip()
    return body or None


def extract_release_notes(
    changelog_text: str,
    tag: str,
    repo: str = '',
) -> str | None:
    """按 tag 从 CHANGELOG 文本提取 Release 说明。

    Args:
        changelog_text: CHANGELOG.md 的完整文本。
        tag: Git 标签名，通常以 ``v`` 开头（如 ``v0.6.0``）。
        repo: GitHub ``owner/repo``，用于在仅提取 ``### 概要`` 时附带完整日志链接。

    Returns:
        提取到的 Markdown 文本（含末尾换行）；未找到对应版本或内容为空时返回 None。
    """
    version = tag.removeprefix('v').strip()
    if not version:
        return None

    section_lines = _find_version_section(changelog_text.splitlines(), version)
    if section_lines is None:
        return None

    summary = _extract_summary_block(section_lines)
    if summary is not None:
        if repo:
            link = f'https://github.com/{repo}/blob/{tag}/CHANGELOG.md'
            return f'{summary}\n\n完整更新记录见 [CHANGELOG.md]({link})。\n'
        return f'{summary}\n'

    full_body = '\n'.join(section_lines).strip()
    if not full_body:
        return None
    return f'{full_body}\n'


def _write_github_output(found: bool) -> None:
    output_path = os.environ.get('GITHUB_OUTPUT')
    if not output_path:
        return
    with open(output_path, 'a', encoding='utf-8') as file:
        file.write(f'found={"true" if found else "false"}\n')


def main() -> int:
    """命令行入口：读取 CHANGELOG.md 并写出指定 tag 的 Release 正文。"""
    parser = argparse.ArgumentParser(
        description='Extract GitHub Release notes for a tag from CHANGELOG.md.'
    )
    parser.add_argument('--changelog', default='CHANGELOG.md', help='CHANGELOG 文件路径。')
    parser.add_argument('--tag', required=True, help='发版 tag，如 v0.6.0。')
    parser.add_argument('--repo', default='', help='GitHub 仓库 owner/repo。')
    parser.add_argument('--out', required=True, help='输出 Markdown 文件路径。')
    args = parser.parse_args()

    changelog_path = Path(args.changelog)
    if not changelog_path.is_file():
        print(f'::warning::未找到 {changelog_path}，跳过覆盖 Release 说明。', file=sys.stderr)
        _write_github_output(False)
        return 0

    notes = extract_release_notes(
        changelog_path.read_text(encoding='utf-8'),
        tag=args.tag,
        repo=args.repo,
    )
    if notes is None:
        version = args.tag.removeprefix('v')
        print(
            f'::warning::{changelog_path} 中未找到版本 {version} 的更新记录，保留默认 Release 说明。',
            file=sys.stderr,
        )
        _write_github_output(False)
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(notes, encoding='utf-8')
    _write_github_output(True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
