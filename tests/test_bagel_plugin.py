"""贝果计划插件的基础测试：插件能加载、能读到自己的资源、打包不丢文件。

本文件检查插件结构与接入。提示识别和路线交互分别在同目录的专项测试中检查。
manifest、const 与封面由 ``scripts/validate_plugin.py --strict`` 校验，这里不重复。

运行：`PYTHONPATH=<主仓>/src uv run pytest <本仓库>/tests/`。
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from one_dragon.base.operation.application.application_factory_manager import (
    ApplicationFactoryManager,
)
from one_dragon.base.operation.application.plugin_import_service import (
    PluginImportService,
)
from one_dragon.base.operation.application.plugin_info import PluginSource
from one_dragon.utils import os_utils

REPO_DIR = Path(__file__).resolve().parents[1]
"""插件仓库根，也是导入根：放进 sys.path 后，插件目录以 ``bagel`` 这个包名被加载。"""

PLUGIN_DIR = REPO_DIR / 'bagel'
"""插件目录：目录名必须等于 plugin.json 里的 id，打包后就是 zip 的顶层目录。"""

MODULE_FILES = sorted(PLUGIN_DIR.rglob('*.py'))
"""插件内全部 Python 文件，含子包与 __init__。"""

ROOT_FILES = sorted(PLUGIN_DIR.glob('*.py'))
"""插件根第一层的 Python 文件，扫描器只看这一层。"""


@pytest.fixture(scope='module', autouse=True)
def _plugin_on_sys_path() -> Iterator[None]:
    """插件以仓库根为模块根，导入前先把它放进 sys.path。"""
    entry = str(REPO_DIR)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    yield


@pytest.fixture
def empty_work_dir(tmp_path: Path) -> Iterator[Path]:
    """把工作目录换成一个空目录，用来验证资源不依赖工作目录。"""
    original = os_utils.get_work_dir()
    os_utils.set_work_dir(tmp_path)
    try:
        yield tmp_path
    finally:
        os_utils.set_work_dir(original)


class _StubContext:
    """最小上下文：只实现插件用到的主程序接口 ``get_plugin_data_dir``，规则照主程序来。"""

    def get_plugin_data_dir(self, app_id: str) -> Path:
        return Path(os_utils.get_path_under_work_dir('plugin_data', app_id))


def _module_name(path: Path) -> str:
    """把文件路径换算成导入名，子包按点号展开，__init__ 归到包本身。"""
    parts = list(path.relative_to(REPO_DIR).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def test_plugin_layout() -> None:
    """插件根第一层只允许一个工厂和一个常量文件，其余模块按职责放进子包。"""
    names = [path.name for path in ROOT_FILES]
    assert 'factory.py' in names
    assert 'const.py' in names
    # 一条龙推荐固定文件名，旧的 前缀_factory.py / 前缀_const.py 只是兼容，不再使用
    assert [n for n in names if n.endswith('_factory.py') or n.endswith('_const.py')] == []
    assert '__init__.py' in names


def test_no_legacy_absolute_imports() -> None:
    """插件内不允许出现绝对导入，防止目录改名后整个插件加载不起来。"""
    legacy = (
        'zzz_od.application.bagel',
        'zzz_od.gui.view.bagel',
        'zzz_od.gui.app_setting.bagel_setting',
        'zzz_od.game_data.map_locator',
    )
    offenders: list[str] = []
    for path in MODULE_FILES:
        text = path.read_text(encoding='utf-8')
        for pattern in legacy:
            if pattern in text:
                offenders.append(f'{path.name}: {pattern}')
        if re.search(r'^from bagel\.', text, re.MULTILINE):
            offenders.append(f'{path.name}: 绝对导入 bagel.*')
    assert offenders == []


def test_modules_importable() -> None:
    """所有模块都能用相对导入单独加载，不依赖被搬走的主仓路径。"""
    failures: list[str] = []
    for path in MODULE_FILES:
        try:
            importlib.import_module(_module_name(path))
        except Exception as exc:
            failures.append(f'{_module_name(path)}: {type(exc).__name__}: {exc}')
    assert failures == []


def test_factory_discovered(empty_work_dir: Path) -> None:
    """插件管理器能发现 bagel，且扫描无失败记录。

    用空工作目录：扫描会顺带构造应用配置，不隔离就会写到真实工作目录。
    """
    manager = ApplicationFactoryManager(None, [(REPO_DIR, PluginSource.THIRD_PARTY)])
    factories = manager.discover_factories()

    matched = [factory for factory in factories if factory.app_id == 'bagel']
    assert len(matched) == 1
    assert manager.scan_failures == []

    info = manager.get_plugin_info('bagel')
    assert info is not None
    assert Path(info.plugin_dir).resolve() == PLUGIN_DIR.resolve()


def test_resources_survive_work_dir_change(empty_work_dir: Path) -> None:
    """换一个空工作目录后，出生点、底图、角标、挂锁模板仍然读得到。"""
    store_module = importlib.import_module('bagel.route.route_store')
    store = store_module.BagelRouteStore(_StubContext())
    assert store.spawn_path.is_relative_to(PLUGIN_DIR.resolve())
    assert store.load_spawns()

    locator = importlib.import_module('bagel.vision.map_locator')
    assert locator.load_map_locators()

    vision = importlib.import_module('bagel.vision.item_vision')
    assert Path(vision._badge_dir()).is_relative_to(PLUGIN_DIR.resolve())

    slots = importlib.import_module('bagel.vision.slots')
    assert slots._safe_lock_template() is not None


def test_load_only_requested_map(tmp_path: Path) -> None:
    """未选地图的图片即使不存在，也不应尝试解码。"""
    from bagel.vision.map_locator import default_layer_dir, load_map_locators

    directory = default_layer_dir()
    manifest = json.loads((directory / 'layers.json').read_text(encoding='utf-8'))
    expected = []
    for layer in manifest['layers']:
        if layer['map_id'] == '城郊幻境':
            expected.append(f"城郊幻境/{layer['name']}")
            layer['image'] = str(directory / layer['image'])
        else:
            layer['image'] = '缺失.png'
    (tmp_path / 'layers.json').write_text(json.dumps(manifest), encoding='utf-8')
    assert expected
    assert [layer.name for layer in load_map_locators(tmp_path, map_id='城郊幻境')] == expected
    with pytest.raises(ValueError, match='未知地图'):
        load_map_locators(tmp_path, map_id='未知地图')


def test_default_routes_inside_plugin(empty_work_dir: Path) -> None:
    """默认路线随插件分发：空工作目录下也能按出生点清单初始化出开启选择。

    这条守首次启动：默认路线若还留在工作目录，用户第一次跑就会得到
    「没有开启的贝果路线」，任务根本开不起来。
    """
    store_module = importlib.import_module('bagel.route.route_store')
    store = store_module.BagelRouteStore(_StubContext())

    assert store.sample_dir.resolve().is_relative_to(PLUGIN_DIR.resolve())
    assert sorted(store.sample_dir.glob('*.sample.route.json'))

    catalog = store.load_catalog()
    assert catalog.routes, '默认路线没有读出来'
    active = [ref for ref in catalog.active.values() if ref is not None]
    assert active, '空工作目录下没有初始化出任何开启路线，首次运行会直接判配置无效'


def _zip_plugin(target_zip: Path) -> Path:
    """把插件目录打成商店风格的 zip：包根就是插件目录名。"""
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PLUGIN_DIR.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            arcname = f'{PLUGIN_DIR.name}/{path.relative_to(PLUGIN_DIR).as_posix()}'
            zf.write(path, arcname)
    return target_zip


def test_store_zip_keeps_resources(tmp_path: Path) -> None:
    """打成商店 zip 后，预览能读出元数据，解压清单包含全部资源而不只是 py。

    这条守的是商店/导入分发路径：一旦导入侧按扩展名过滤，
    模板、底图、网页、screen 定义就会在安装后凭空消失。
    """
    zip_path = _zip_plugin(tmp_path / 'bagel.zip')

    # ZIP 解析全程不落盘、不碰 ctx，构造时略过上下文。
    service = PluginImportService.__new__(PluginImportService)

    previews = service.preview_plugins_from_zip(zip_path)
    assert [item.plugin_name for item in previews] == ['bagel']
    manifest = json.loads((PLUGIN_DIR / 'plugin.json').read_text(encoding='utf-8'))
    assert previews[0].version == manifest['version']

    with zipfile.ZipFile(zip_path, 'r') as zf:
        layouts = service._analyze_zip(zf)

    assert len(layouts) == 1
    assert layouts[0].plugin_dir_name == 'bagel'

    members = {path.as_posix() for _, path in layouts[0].members}
    packed = {
        path.relative_to(PLUGIN_DIR).as_posix()
        for path in PLUGIN_DIR.rglob('*')
        if path.is_file() and '__pycache__' not in path.parts
    }
    assert members == packed
