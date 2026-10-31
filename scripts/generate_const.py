from __future__ import annotations

import argparse
import sys
from pathlib import Path

from manifest_utils import (
    const_path_for_app,
    find_package_dir,
    iter_declared_apps,
    load_json,
    render_const,
)


def compare_or_write(repo: Path, path: Path, content: str, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    display_path = path.relative_to(repo).as_posix()
    if current == content:
        print(f"PASS const {display_path}")
        return True

    if check:
        print(f"FAIL CONST_OUTDATED {display_path}")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"PASS const generated {display_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate *_const.py from plugin.json.")
    parser.add_argument("--repo", default=".", help="Plugin repository root.")
    parser.add_argument("--check", action="store_true", help="Compare only.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    package_dir = find_package_dir(repo)
    manifest = load_json(package_dir / "plugin.json")

    targets = [(const_path_for_app(package_dir, manifest), render_const(manifest))]
    for app in iter_declared_apps(manifest):
        targets.append(
            (const_path_for_app(package_dir, manifest, app), render_const(manifest, app))
        )

    results = [compare_or_write(repo, path, content, args.check) for path, content in targets]
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL SCRIPT_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
