from __future__ import annotations

import argparse
import json
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from manifest_utils import (
    COVER_ASPECT_RATIO,
    COVER_ASPECT_TOLERANCE,
    COVER_EXTENSIONS,
    DEFAULT_MAX_COVER_BYTES,
    DEFAULT_MAX_ZIP_BYTES,
    MIN_COVER_HEIGHT,
    MIN_COVER_WIDTH,
    PLUGIN_ID_PATTERN,
    RISK_TAGS,
    const_path_for_app,
    find_package_dirs,
    is_dependency_file,
    iter_declared_apps,
    load_json,
    plugin_id,
    render_const,
)

MAX_UNCOMPRESSED_FACTOR = 10
COVER_MAGIC: dict[str, tuple[bytes, ...]] = {
    ".webp": (b"RIFF",),
}


def read_webp_size(data: bytes) -> tuple[int, int] | None:
    """从 WebP 文件头读出宽高。

    三种子格式的宽高位置不同：有损 VP8、无损 VP8L、扩展 VP8X。

    Args:
        data: WebP 文件内容。

    Returns:
        (宽, 高)；不是合法 WebP 或长度不足时返回 None。
    """
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None

    chunk = data[12:16]
    if chunk == b"VP8 ":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    if chunk == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    return None


class Reporter:
    def __init__(self, quiet: bool) -> None:
        self.quiet = quiet
        self.errors: list[dict[str, str]] = []

    @property
    def failed(self) -> bool:
        return bool(self.errors)

    def pass_(self, label: str, detail: str) -> None:
        if not self.quiet:
            print(f"PASS {label} {detail}")

    def fail(self, code: str, detail: str) -> None:
        self.errors.append({"code": code, "detail": detail})
        if not self.quiet:
            print(f"FAIL {code} {detail}")


def is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_http_url(value: Any) -> bool:
    return isinstance(value, str) and urlparse(value).scheme in {"http", "https"}


def validate_manifest_fields(reporter: Reporter, manifest: dict[str, Any]) -> None:
    current_id = plugin_id(manifest)
    if PLUGIN_ID_PATTERN.fullmatch(current_id):
        reporter.pass_("id", current_id)
    else:
        reporter.fail("INVALID_ID", repr(current_id))

    for field in ("name", "summary", "version"):
        if not is_non_empty_str(manifest.get(field)):
            reporter.fail("INVALID_FIELD", field)

    version_code = manifest.get("versionCode")
    if isinstance(version_code, bool) or not isinstance(version_code, int) or version_code <= 0:
        reporter.fail("INVALID_VERSION_CODE", repr(version_code))
    else:
        reporter.pass_("versionCode", str(version_code))

    for field in ("defaultGroup", "needNotify"):
        if not isinstance(manifest.get(field), bool):
            reporter.fail("INVALID_FIELD", field)

    author = manifest.get("author")
    if not isinstance(author, dict) or not is_non_empty_str(author.get("name")):
        reporter.fail("INVALID_FIELD", "author.name")
    elif not is_http_url(author.get("url")):
        reporter.fail("INVALID_FIELD", "author.url")

    if not is_http_url(manifest.get("homepage")):
        reporter.fail("INVALID_HOMEPAGE", repr(manifest.get("homepage")))

    risks = manifest.get("risk")
    if not isinstance(risks, list) or any(risk not in RISK_TAGS for risk in risks):
        reporter.fail("INVALID_RISK", repr(risks))
    else:
        reporter.pass_("risk", ",".join(risks) or "-")

    compatibility = manifest.get("compatibility")
    if (
        not isinstance(compatibility, dict)
        or not is_non_empty_str(compatibility.get("minAppVersion"))
        or not isinstance(compatibility.get("minCoreApiVersion"), int)
        or not is_non_empty_str(compatibility.get("python"))
    ):
        reporter.fail("INVALID_FIELD", "compatibility")

    apps = manifest.get("apps", [])
    if not isinstance(apps, list):
        reporter.fail("INVALID_FIELD", "apps")
        return
    app_ids: set[str] = set()
    for app in apps:
        app_id = str(app.get("id", "")) if isinstance(app, dict) else ""
        if (
            app_id in app_ids
            or not app_id.startswith(f"{current_id}_")
            or not PLUGIN_ID_PATTERN.fullmatch(app_id)
        ):
            reporter.fail("APP_ID_MISMATCH", app_id or repr(app))
        else:
            app_ids.add(app_id)
            reporter.pass_("app id", app_id)


def validate_cover(
    reporter: Reporter, package_dir: Path, manifest: dict[str, Any], max_cover_bytes: int
) -> Path | None:
    cover = manifest.get("cover")
    if cover in (None, ""):
        return None
    if not isinstance(cover, str):
        reporter.fail("INVALID_COVER", repr(cover))
        return None

    relative = PurePosixPath(cover)
    if relative.is_absolute() or ".." in relative.parts or "\\" in cover:
        reporter.fail("INVALID_COVER", cover)
        return None
    suffix = relative.suffix.lower()
    if suffix not in COVER_EXTENSIONS:
        reporter.fail("INVALID_COVER", f"{cover} (webp only)")
        return None

    path = package_dir / relative
    if not path.is_file():
        reporter.fail("COVER_MISSING", cover)
        return None
    data = path.read_bytes()
    if len(data) > max_cover_bytes:
        reporter.fail("COVER_TOO_LARGE", f"{cover} {len(data)} > {max_cover_bytes}")
        return None
    if not data.startswith(COVER_MAGIC[suffix]) or data[8:12] != b"WEBP":
        reporter.fail("INVALID_COVER", f"{cover} content does not match {suffix}")
        return None

    size = read_webp_size(data)
    if size is None:
        reporter.fail("INVALID_COVER", f"{cover} 读不出宽高")
        return None
    width, height = size
    if width < MIN_COVER_WIDTH or height < MIN_COVER_HEIGHT:
        reporter.fail(
            "COVER_TOO_SMALL",
            f"{cover} {width}x{height} < {MIN_COVER_WIDTH}x{MIN_COVER_HEIGHT}",
        )
        return None
    if abs(width / height - COVER_ASPECT_RATIO) > COVER_ASPECT_TOLERANCE:
        reporter.fail("COVER_BAD_RATIO", f"{cover} {width}x{height} 不是 16:10")
        return None

    reporter.pass_("cover", f"{cover} {width}x{height}")
    return path


def validate_package(
    reporter: Reporter,
    package_dir: Path,
    expect_id: str | None,
    expect_tag: str | None,
    max_cover_bytes: int,
) -> tuple[dict[str, Any], Path | None]:
    manifest_path = package_dir / "plugin.json"
    if not manifest_path.is_file():
        reporter.fail("MANIFEST_MISSING", f"{package_dir.name}/plugin.json")
        return {}, None
    try:
        manifest = load_json(manifest_path)
    except (ValueError, UnicodeDecodeError) as exc:
        reporter.fail("INVALID_MANIFEST", str(exc))
        return {}, None

    validate_manifest_fields(reporter, manifest)
    current_id = plugin_id(manifest)

    if package_dir.name != current_id:
        reporter.fail("PACKAGE_MISMATCH", f"directory {package_dir.name!r} != id {current_id!r}")
    if expect_id is not None and current_id != expect_id:
        reporter.fail("ID_MISMATCH", f"{current_id!r} != expected {expect_id!r}")
    if expect_tag is not None and f"v{manifest.get('version', '')}" != expect_tag:
        reporter.fail("TAG_MISMATCH", f"tag {expect_tag!r} != v{manifest.get('version', '')}")

    required = [
        ("__init__.py", "INIT_MISSING"),
        ("factory.py", "FACTORY_MISSING"),
    ]
    for name, code in required:
        if (package_dir / name).is_file():
            reporter.pass_("file", f"{package_dir.name}/{name}")
        else:
            reporter.fail(code, f"{package_dir.name}/{name}")

    targets = [(const_path_for_app(package_dir, manifest), render_const(manifest))]
    for app in iter_declared_apps(manifest):
        targets.append(
            (const_path_for_app(package_dir, manifest, app), render_const(manifest, app))
        )
    for path, expected in targets:
        display = path.relative_to(package_dir.parent).as_posix()
        if not path.is_file():
            reporter.fail("CONST_MISSING", display)
        elif path.read_text(encoding="utf-8") != expected:
            reporter.fail("CONST_OUTDATED", f"{display} (run scripts/generate_const.py)")
        else:
            reporter.pass_("const", display)

    for path in package_dir.rglob("*"):
        if path.is_file() and is_dependency_file(path.name):
            reporter.fail(
                "DEPENDENCY_FILE",
                f"{path.relative_to(package_dir.parent).as_posix()} (third-party dependencies are not allowed)",
            )

    cover = validate_cover(reporter, package_dir, manifest, max_cover_bytes)
    return manifest, cover


def validate_repo_root(reporter: Reporter, repo: Path, strict: bool) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = find_package_dirs(repo)
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        reporter.fail("PACKAGE_COUNT", f"expected one subdirectory with plugin.json, found: {names}")
        return None, None

    # 旧模板会在根目录留一份同步的 plugin.json：与包内一致时容忍（提示删除），不一致才失败
    root_manifest: dict[str, Any] | None = None
    if (repo / "plugin.json").exists():
        try:
            root_manifest = load_json(repo / "plugin.json")
        except (ValueError, UnicodeDecodeError) as exc:
            reporter.fail("ROOT_MANIFEST", f"repo root plugin.json is not valid JSON: {exc}")

    for prefix, code in (("README", "README_MISSING"), ("LICENSE", "LICENSE_MISSING")):
        if any(child.is_file() and child.name.upper().startswith(prefix) for child in repo.iterdir()):
            reporter.pass_("file", prefix)
        elif strict:
            reporter.fail(code, prefix)
    return candidates[0], root_manifest


def safe_extract(reporter: Reporter, zip_path: Path, target: Path, max_zip_bytes: int) -> Path | None:
    size = zip_path.stat().st_size
    if size > max_zip_bytes:
        reporter.fail("ZIP_TOO_LARGE", f"{size} > {max_zip_bytes}")
        return None
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        reporter.fail("INVALID_ZIP", str(exc))
        return None

    with archive:
        infos = archive.infolist()
        tops: set[str] = set()
        total = 0
        for info in infos:
            name = info.filename
            parts = PurePosixPath(name).parts
            if (
                not name
                or name.startswith("/")
                or "\\" in name
                or ":" in name
                or ".." in parts
            ):
                reporter.fail("UNSAFE_PATH", name)
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                reporter.fail("SYMLINK", name)
                continue
            if info.flag_bits & 0x1:
                reporter.fail("ENCRYPTED_ENTRY", name)
                continue
            tops.add(parts[0])
            if len(parts) == 1 and not info.is_dir():
                reporter.fail("TOP_LEVEL_FILE", name)
            total += info.file_size
        if total > max_zip_bytes * MAX_UNCOMPRESSED_FACTOR:
            reporter.fail("ZIP_TOO_LARGE", f"uncompressed {total}")
        if len(tops) != 1:
            reporter.fail("ZIP_LAYOUT", f"expected one top-level directory, found: {sorted(tops)}")
        if reporter.failed:
            return None
        archive.extractall(target)
    reporter.pass_("zip", f"{zip_path.name} {size} bytes")
    return target / tops.pop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ZZZ OD plugin repo, package dir or release zip.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--repo", default=None, help="Plugin repository root (default: current dir).")
    mode.add_argument("--package-dir", help="Plugin package directory.")
    mode.add_argument("--zip", help="Release asset <id>.zip.")
    parser.add_argument("--strict", action="store_true", help="Require README and LICENSE in repo mode.")
    parser.add_argument("--expect-id", help="Required plugin id.")
    parser.add_argument("--expect-tag", help="Required release tag, must equal v<version>.")
    parser.add_argument("--max-zip-bytes", type=int, default=DEFAULT_MAX_ZIP_BYTES)
    parser.add_argument("--max-cover-bytes", type=int, default=DEFAULT_MAX_COVER_BYTES)
    parser.add_argument("--cover-out", help="Copy the cover to this path without extension; the extension is appended.")
    parser.add_argument("--json", action="store_true", help="Print a JSON report only.")
    args = parser.parse_args()

    reporter = Reporter(quiet=args.json)
    manifest: dict[str, Any] = {}
    cover_file = ""
    has_app_setting = False

    with tempfile.TemporaryDirectory() as temp:
        package_dir: Path | None
        root_manifest: dict[str, Any] | None = None
        if args.zip:
            package_dir = safe_extract(reporter, Path(args.zip), Path(temp), args.max_zip_bytes)
        elif args.package_dir:
            package_dir = Path(args.package_dir).resolve()
        else:
            package_dir, root_manifest = validate_repo_root(reporter, Path(args.repo or ".").resolve(), args.strict)

        if package_dir is not None:
            manifest, cover = validate_package(
                reporter, package_dir, args.expect_id, args.expect_tag, args.max_cover_bytes
            )
            has_app_setting = (package_dir / f"{plugin_id(manifest)}_app_setting.py").is_file()
            if root_manifest is not None and manifest:
                if root_manifest == manifest:
                    reporter.pass_("root manifest", "matches package plugin.json (legacy copy, safe to delete)")
                else:
                    reporter.fail(
                        "ROOT_MANIFEST_MISMATCH",
                        "repo root plugin.json differs from the package plugin.json; delete the root copy",
                    )
            if cover is not None and args.cover_out and not reporter.failed:
                destination = Path(args.cover_out + cover.suffix.lower())
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cover, destination)
                cover_file = str(destination)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not reporter.failed,
                    "errors": reporter.errors,
                    "manifest": manifest,
                    "coverFile": cover_file,
                    "hasAppSetting": has_app_setting,
                }
            )
        )
    return 1 if reporter.failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL SCRIPT_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
