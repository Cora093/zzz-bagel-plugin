from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PLUGIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
RISK_TAGS: set[str] = {
    "filesystem",
    "network",
    "account",
    "automation",
    "ocr",
    "yolo",
    "clipboard",
    "process",
}
COVER_EXTENSIONS: set[str] = {".webp"}
"""封面只收 WebP：同等体积下比 PNG 保留更多细节。"""
MIN_COVER_WIDTH = 600
MIN_COVER_HEIGHT = 376
COVER_ASPECT_RATIO = MIN_COVER_WIDTH / MIN_COVER_HEIGHT
"""卡片是 300×188，封面按两倍起步，高分屏下才不会发虚。"""
COVER_ASPECT_TOLERANCE = 0.03
"""宽高比允许的偏差，避免封面被卡片拉伸变形。"""
DEFAULT_MAX_COVER_BYTES = 512 * 1024
DEFAULT_MAX_ZIP_BYTES = 50 * 1024 * 1024
DEPENDENCY_FILES: set[str] = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "pdm.lock",
    "environment.yml",
    "environment.yaml",
}
EXCLUDED_NAMES: set[str] = {"__pycache__", ".DS_Store", "Thumbs.db"}
EXCLUDED_SUFFIXES: set[str] = {".pyc", ".pyo"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def find_package_dirs(repo: Path) -> list[Path]:
    return sorted(
        child
        for child in repo.iterdir()
        if child.is_dir()
        and not child.name.startswith(".")
        and (child / "plugin.json").is_file()
    )


def find_package_dir(repo: Path) -> Path:
    candidates = find_package_dirs(repo)
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise ValueError(
            f"expected exactly one subdirectory with plugin.json in {repo}, found: {names}"
        )
    return candidates[0]


def plugin_id(manifest: dict[str, Any]) -> str:
    value = manifest.get("id")
    return value if isinstance(value, str) else ""


def is_dependency_file(name: str) -> bool:
    lower = name.lower()
    return lower in DEPENDENCY_FILES or (
        lower.startswith("requirements") and lower.endswith(".txt")
    )


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_NAMES for part in path.parts) or (
        path.suffix in EXCLUDED_SUFFIXES
    )


def const_path_for_app(
    package_dir: Path, manifest: dict[str, Any], app: dict[str, Any] | None = None
) -> Path:
    if app is None:
        return package_dir / f"{plugin_id(manifest)}_const.py"

    app_id = str(app.get("id", ""))
    feature = app_id.removeprefix(f"{plugin_id(manifest)}_")
    return package_dir / feature / f"{feature}_const.py"


def py_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def py_bool(value: Any) -> str:
    return "True" if bool(value) else "False"


def render_const(manifest: dict[str, Any], app: dict[str, Any] | None = None) -> str:
    author = manifest.get("author") if isinstance(manifest.get("author"), dict) else {}
    source = manifest if app is None else app
    app_id = plugin_id(manifest) if app is None else app.get("id", "")

    return (
        "# Generated from plugin.json. Run scripts/generate_const.py after editing manifest.\n"
        f"APP_ID = {py_string(app_id)}\n"
        f"APP_NAME = {py_string(source.get('name', ''))}\n"
        f"DEFAULT_GROUP = {py_bool(source.get('defaultGroup', False))}\n"
        f"NEED_NOTIFY = {py_bool(source.get('needNotify', False))}\n"
        "\n"
        f"PLUGIN_AUTHOR = {py_string(author.get('name', ''))}\n"
        f"PLUGIN_HOMEPAGE = {py_string(manifest.get('homepage', ''))}\n"
        f"PLUGIN_VERSION = {py_string(manifest.get('version', ''))}\n"
        f"PLUGIN_DESCRIPTION = {py_string(source.get('summary', ''))}\n"
    )


def iter_declared_apps(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    apps = manifest.get("apps", [])
    if not isinstance(apps, list):
        return []
    return [app for app in apps if isinstance(app, dict)]
