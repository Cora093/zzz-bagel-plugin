"""贝果计划插件：结构约束、自包含资源与接入测试。

这些用例不初始化完整游戏上下文，只守住「搬成插件后仍然自洽」的不变量：
结构满足扫描器要求、模块全部可导入、资源都落在插件目录内、
换个工作目录照样读得到，以及插件管理器和设置扫描器能发现它。

测试随插件走：`uv run pytest plugins/bagel/tests/`。
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
"""插件仓库根：放 README、打包脚本与工作流。"""

PLUGIN_DIR = REPO_DIR / 'bagel'
"""插件目录：目录名必须等于 plugin.json 里的 id，打包后就是 zip 的顶层目录。"""

PLUGINS_DIR = REPO_DIR
"""导入根：把它放进 sys.path 后，插件目录以 ``bagel`` 这个包名被加载。"""

MODULE_FILES = sorted(PLUGIN_DIR.rglob('*.py'))
"""插件内全部 Python 文件，含子包与 __init__。"""

ROOT_FILES = sorted(PLUGIN_DIR.glob('*.py'))
"""插件根第一层的 Python 文件，扫描器只看这一层。"""


@pytest.fixture(scope='module', autouse=True)
def _plugin_on_sys_path() -> Iterator[None]:
    """插件以仓库根为模块根，导入前先把它放进 sys.path。"""
    entry = str(PLUGINS_DIR)
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


def _module_name(path: Path) -> str:
    """把文件路径换算成导入名，子包按点号展开，__init__ 归到包本身。"""
    parts = list(path.relative_to(PLUGINS_DIR).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def test_plugin_layout() -> None:
    """插件根第一层只允许一个工厂和一个常量文件，其余模块按职责放进子包。"""
    names = [path.name for path in ROOT_FILES]
    assert [name for name in names if name.endswith('_factory.py')] == ['bagel_factory.py']
    assert [name for name in names if name.endswith('_const.py')] == ['bagel_const.py']
    assert '__init__.py' in names
    # 模块数量不是契约：删掉不执行的旧流程后数量会变，这里只要求插件不是空壳。
    assert MODULE_FILES


def test_manifest_matches_const() -> None:
    """plugin.json 与生成的 const 保持一致，商店校验比对的就是这两份。

    ``scripts/generate_const.py`` 会按 plugin.json 重写 const；
    手改 const 而不改 manifest，发版时会被校验判成 CONST_OUTDATED。
    """
    manifest = json.loads((PLUGIN_DIR / 'plugin.json').read_text(encoding='utf-8'))
    assert manifest['id'] == PLUGIN_DIR.name

    const_text = (PLUGIN_DIR / 'bagel_const.py').read_text(encoding='utf-8')
    expected = {
        'APP_ID': manifest['id'],
        'APP_NAME': manifest['name'],
        'PLUGIN_VERSION': manifest['version'],
        'PLUGIN_AUTHOR': manifest['author']['name'],
        'PLUGIN_HOMEPAGE': manifest['homepage'],
        'PLUGIN_DESCRIPTION': manifest['summary'],
    }
    for name, value in expected.items():
        assert f'{name} = "{value}"' in const_text, name
    assert f'DEFAULT_GROUP = {manifest["defaultGroup"]}' in const_text
    assert f'NEED_NOTIFY = {manifest["needNotify"]}' in const_text


def test_modules_importable() -> None:
    """所有模块都能用相对导入单独加载，不依赖被搬走的主仓路径。"""
    failures: list[str] = []
    for path in MODULE_FILES:
        try:
            importlib.import_module(_module_name(path))
        except Exception as exc:
            failures.append(f'{_module_name(path)}: {type(exc).__name__}: {exc}')
    assert failures == []


def test_app_identity() -> None:
    """应用标识来自生成的 const，插件根指向插件目录自身。"""
    const = importlib.import_module('bagel.bagel_const')
    defs = importlib.import_module('bagel.bagel_defs')
    assert const.APP_ID == 'bagel'
    assert Path(defs.PLUGIN_DIR).resolve() == PLUGIN_DIR.resolve()


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


@pytest.mark.parametrize('template', ['interact_icon', 'card_key_icon', 'interact_key_f', 'safe_slot_locked'])
def test_templates_inside_plugin(template: str) -> None:
    """模板原图随插件分发，匹配不再经过全局模板库。"""
    assert (PLUGIN_DIR / 'assets' / 'template' / 'bagel' / template / 'raw.png').is_file()


def test_spawn_points_inside_plugin() -> None:
    """出生点清单随插件分发，不依赖工作目录。"""
    spawn = PLUGIN_DIR / 'assets' / 'game_data' / 'bagel' / 'spawn_points.yml'
    assert spawn.is_file()


def test_default_routes_inside_plugin(empty_work_dir: Path) -> None:
    """默认路线随插件分发：空工作目录下也能按出生点清单初始化出开启选择。

    这条守首次启动：默认路线若还留在工作目录，用户第一次跑就会得到
    「没有开启的贝果路线」，任务根本开不起来。
    """
    store_module = importlib.import_module('bagel.route.bagel_route_store')
    store = store_module.BagelRouteStore()

    assert store.sample_dir.resolve().is_relative_to(PLUGIN_DIR.resolve())
    defaults = sorted(store.sample_dir.glob('*.sample.route.json'))
    assert len(defaults) >= 3

    catalog = store.load_catalog()
    assert catalog.routes, '默认路线没有读出来'
    active = [ref for ref in catalog.active.values() if ref is not None]
    assert active, '空工作目录下没有初始化出任何开启路线，首次运行会直接判配置无效'


def test_user_data_outside_plugin(empty_work_dir: Path) -> None:
    """用户数据落在工作目录的 plugin_data/<app_id> 下，不进插件目录。

    插件升级是整个目录替换，数据写在插件目录里会随旧版本一起删掉。
    """
    expected = empty_work_dir / 'plugin_data' / 'bagel'

    store_module = importlib.import_module('bagel.route.bagel_route_store')
    store = store_module.BagelRouteStore()
    assert store.custom_dir.resolve() == (expected / 'routes').resolve()
    assert store.settings_path.resolve() == (expected / 'route_settings.yml').resolve()
    assert not store.custom_dir.resolve().is_relative_to(PLUGIN_DIR.resolve())

    stats_module = importlib.import_module('bagel.route.bagel_route_stats')
    records = stats_module.BagelRouteRecorder().path()
    assert records.resolve() == (expected / 'route_records.jsonl').resolve()

    shot_module = importlib.import_module('bagel.bagel_screenshot')
    assert shot_module.screenshot_dir().resolve() == (expected / 'screenshots').resolve()


def test_readme_images_exist() -> None:
    """README 引用的截图都随插件分发，避免说明里的图裂掉。"""
    readme = REPO_DIR / 'README.md'
    assert readme.is_file()
    text = readme.read_text(encoding='utf-8')

    refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    assert refs, 'README 里没有引用任何截图'
    for ref in refs:
        assert (REPO_DIR / ref).is_file(), ref


def test_map_layers_load_from_plugin() -> None:
    """分层底图与清单从插件目录加载，层名沿用游戏内叫法。"""
    locator = importlib.import_module('bagel.vision.map_locator')
    layer_dir = Path(locator.default_layer_dir()).resolve()
    assert layer_dir.is_relative_to(PLUGIN_DIR.resolve())
    assert (layer_dir / 'layers.json').is_file()

    locators = locator.load_map_locators()
    assert locators
    assert any(item.name.startswith('雅努斯幻境/') for item in locators)


def test_item_badges_inside_plugin() -> None:
    """物品角标模板齐全，且不落到工作目录。"""
    vision = importlib.import_module('bagel.vision.bagel_item_vision')
    badge_dir = Path(vision._badge_dir()).resolve()
    assert badge_dir.is_relative_to(PLUGIN_DIR.resolve())
    for _, stem in vision._BADGE_FILES:
        assert (badge_dir / f'{stem}.png').is_file()


def test_safe_lock_template_readable() -> None:
    """保险箱挂锁模板能读出有效纹理，缺失时判空会全部失效。"""
    slots = importlib.import_module('bagel.vision.bagel_slots')
    template = slots._safe_lock_template()
    assert template is not None
    assert template.shape == (56, 48)


def test_resources_survive_work_dir_change(empty_work_dir: Path) -> None:
    """换一个空工作目录后，出生点、底图、角标、模板仍然读得到。"""
    store_module = importlib.import_module('bagel.route.bagel_route_store')
    store = store_module.BagelRouteStore()
    assert store.spawn_path.is_relative_to(PLUGIN_DIR.resolve())
    assert store.load_spawns()

    locator = importlib.import_module('bagel.vision.map_locator')
    assert locator.load_map_locators()

    vision = importlib.import_module('bagel.vision.bagel_item_vision')
    assert Path(vision._badge_dir()).is_relative_to(PLUGIN_DIR.resolve())

    slots = importlib.import_module('bagel.vision.bagel_slots')
    assert slots._safe_lock_template() is not None


def test_screen_yaml_inside_plugin() -> None:
    """screen 定义随插件分发，app_id 统一指向插件自身。"""
    files = sorted((PLUGIN_DIR / 'screen_info').glob('*.yml'))
    assert len(files) >= 9

    screen_ids: set[str] = set()
    for path in files:
        text = path.read_text(encoding='utf-8')
        assert 'app_id: bagel' in text, path.name
        match = re.search(r'^screen_id:\s*(\S+)$', text, re.MULTILINE)
        assert match is not None, path.name
        screen_ids.add(match.group(1))
    assert len(screen_ids) == len(files)

    names = {path.name for path in files}
    assert {'bagel_hud.yml', 'bagel_hub.yml', 'bagel_warehouse.yml'} <= names


def test_web_asset_refs_exist() -> None:
    """路线编辑器网页按插件根解析相对路径，引用到的资源必须真实存在。"""
    index = PLUGIN_DIR / 'web' / 'index.html'
    assert index.is_file()
    text = index.read_text(encoding='utf-8')

    refs = set(re.findall(r'(?:src|fetch\()\s*[\'"]?((?:assets|web)/[\w./-]+)', text))
    assert refs, '网页里没有找到相对资源引用，检查改写是否回退'

    for ref in refs:
        candidate = PLUGIN_DIR / ref
        if candidate.suffix in ('.js', '.css', '.html'):
            continue
        # 目录型引用（底图目录）只要求目录存在
        assert candidate.parent.is_dir() or candidate.is_dir(), ref


def test_factory_discovered(empty_work_dir: Path) -> None:
    """插件管理器能发现 bagel，且扫描无失败记录。

    用空工作目录：扫描会顺带构造应用配置，不隔离就会写到真实工作目录。
    """
    manager = ApplicationFactoryManager(None, [(PLUGINS_DIR, PluginSource.THIRD_PARTY)])
    factories = manager.discover_factories()

    matched = [factory for factory in factories if factory.app_id == 'bagel']
    assert len(matched) == 1
    assert manager.scan_failures == []

    info = manager.get_plugin_info('bagel')
    assert info is not None
    assert info.source == PluginSource.THIRD_PARTY
    assert Path(info.plugin_dir).resolve() == PLUGIN_DIR.resolve()


def test_setting_provider_registered() -> None:
    """设置界面提供者随插件被发现，app_id 与应用一致。"""
    from one_dragon_qt.services.app_setting.app_setting_provider import (
        AppSettingProvider,
    )

    provider_module = importlib.import_module('bagel.bagel_app_setting')
    providers = [
        value for value in vars(provider_module).values()
        if isinstance(value, type)
        and issubclass(value, AppSettingProvider)
        and value is not AppSettingProvider
        and value.__module__ == provider_module.__name__
    ]
    assert len(providers) == 1
    assert providers[0].app_id == 'bagel'


def test_plugin_metadata_declared(empty_work_dir: Path) -> None:
    """插件元数据齐全，插件管理界面与商店覆盖安装都能读出作者和版本。"""
    manager = ApplicationFactoryManager(None, [(PLUGINS_DIR, PluginSource.THIRD_PARTY)])
    manager.discover_factories()

    info = manager.get_plugin_info('bagel')
    assert info is not None
    manifest = json.loads((PLUGIN_DIR / 'plugin.json').read_text(encoding='utf-8'))
    assert info.version == manifest['version']
    assert info.author.startswith('Cora')
    assert 'kawayiYokami' in info.author
    assert info.description
    assert info.homepage


def test_default_group_app(empty_work_dir: Path) -> None:
    """贝果进一条龙默认应用组，同时不限定只能由代码入口调用。"""
    manager = ApplicationFactoryManager(None, [(PLUGINS_DIR, PluginSource.THIRD_PARTY)])
    manager.discover_factories()

    factory = next(item for item in manager.discover_factories() if item.app_id == 'bagel')
    assert factory.default_group is True
    assert factory.internal_app is False


def test_cover_declared(empty_work_dir: Path) -> None:
    """封面随插件分发，格式与尺寸符合商店要求：WebP、至少 600×376、16:10。"""
    from PIL import Image

    cover = PLUGIN_DIR / 'assets' / 'cover.webp'
    assert cover.is_file()
    assert cover.read_bytes()[:4] == b'RIFF'

    with Image.open(cover) as image:
        assert image.format == 'WEBP'
        assert image.width >= 600
        assert image.height >= 376
        assert abs(image.width / image.height - 600 / 376) < 0.03

    manager = ApplicationFactoryManager(None, [(PLUGINS_DIR, PluginSource.THIRD_PARTY)])
    manager.discover_factories()
    info = manager.get_plugin_info('bagel')
    assert info is not None
    assert info.cover_path is not None
    assert Path(info.cover_path).resolve() == cover.resolve()


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
    assert previews[0].author.startswith('Cora')

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

    assert any(name.startswith('assets/game_data/map_locator/') for name in members)
    assert any(name.startswith('assets/template/bagel/') for name in members)
    assert any(name.startswith('screen_info/') and name.endswith('.yml') for name in members)
    assert 'web/index.html' in members
    assert len([name for name in members if not name.endswith('.py')]) > 20
