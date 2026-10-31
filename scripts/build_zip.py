from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

from manifest_utils import find_package_dir, is_excluded, load_json, plugin_id

FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def iter_package_files(package_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in package_dir.rglob("*"):
        relative = path.relative_to(package_dir)
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed: {relative.as_posix()}")
        if path.is_file() and not is_excluded(relative):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(package_dir).as_posix())


def build_zip(package_dir: Path, output: Path) -> None:
    top = package_dir.name
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for path in iter_package_files(package_dir):
            arcname = f"{top}/{path.relative_to(package_dir).as_posix()}"
            info = zipfile.ZipInfo(arcname, date_time=FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def write_github_output(values: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as file:
        for key, value in values.items():
            file.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic <id>.zip release asset.")
    parser.add_argument("--repo", default=".", help="Plugin repository root.")
    parser.add_argument("--out", default="dist", help="Output directory.")
    args = parser.parse_args()

    package_dir = find_package_dir(Path(args.repo).resolve())
    manifest = load_json(package_dir / "plugin.json")
    current_id = plugin_id(manifest)
    if current_id != package_dir.name:
        raise ValueError(f"plugin.json id {current_id!r} != directory {package_dir.name!r}")

    output = Path(args.out).resolve() / f"{current_id}.zip"
    build_zip(package_dir, output)
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    print(f"PASS zip {output}")
    print(f"PASS sha256 {sha256}")
    write_github_output(
        {
            "id": current_id,
            "version": str(manifest.get("version", "")),
            "zip": str(output),
            "sha256": sha256,
        }
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL SCRIPT_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
