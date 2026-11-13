"""贝果路线与出生点的存储：加载、写入保护与开启选择。
文件布局（前两项随插件分发，后两项写在插件数据目录）：

- ``assets/game_data/bagel/spawn_points.yml``：出生点清单。
- ``assets/game_data/bagel/routes/<标识>.sample.route.json``：默认路线，内容只读。
- ``plugin_data/<APP_ID>/routes/custom_<标识>.route.json``：自定义路线。
- ``assets/game_data/bagel/keycard_rooms.yml``：开房卡房间清单。
- ``plugin_data/<APP_ID>/route_settings.yml``：每个出生点当前开启的跑刀路线，
  以及每个房间当前开启的开房卡房间段和各出生点的入口段。

开启选择每个位置只存一个路线引用或 ``null``，天然保证同一位置最多开启一条。
路线文件不保存开关状态。界面、录制器和执行器都通过本模块读写，不自行拼路径。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from one_dragon.utils.log_utils import log

from ..defs import PLUGIN_DIR, get_plugin_data_dir
from .route_model import (
    SEGMENT_ENTRY,
    SEGMENT_ROOM,
    SOURCE_CUSTOM,
    SOURCE_DEFAULT,
    BagelKeycardRoom,
    BagelKeycardTag,
    BagelRoute,
    BagelRoutePoint,
    BagelRouteRef,
    BagelSpawnPoint,
    check_entry_segment,
    check_room_segment,
    check_route_start,
    check_run_points,
    join_keycard_route,
    now_iso,
    validate_route_name,
)

SPAWN_FILE: tuple[str, ...] = ('assets', 'game_data', 'bagel', 'spawn_points.yml')
"""出生点清单相对插件目录的路径。"""

ROOM_FILE: tuple[str, ...] = ('assets', 'game_data', 'bagel', 'keycard_rooms.yml')
"""开房卡房间清单相对插件目录的路径。"""

SAMPLE_DIR: tuple[str, ...] = ('assets', 'game_data', 'bagel', 'routes')
"""默认路线目录相对插件目录的路径；默认路线随插件分发，不依赖工作目录。"""

SAMPLE_SUFFIX: str = '.sample.route.json'
"""默认路线文件后缀。"""

ROUTE_SUFFIX: str = '.route.json'
"""路线文件后缀；自定义路线只用这个后缀。"""

CUSTOM_PREFIX: str = 'custom_'
"""自定义路线标识前缀。"""

CUSTOM_ID_PATTERN: re.Pattern[str] = re.compile(r'^custom_[0-9a-z_]{1,48}$')
"""自定义路线标识的合法形式；写入前检查，防止借标识写到默认路线或目录外。"""

SETTINGS_VERSION: int = 1
"""开启选择配置的格式版本。"""

class BagelRouteConfigError(Exception):
    """路线配置无法使用：配置损坏、引用失效或写入被拒绝。"""


@dataclass
class BagelKeycardSelection:
    """一个房间的开房卡开启选择：一条房间段，加每个出生点一条入口段。"""

    room: BagelRouteRef | None = None
    """当前开启的房间段；None 表示该房间没有可用房间段。"""
    entries: dict[str, BagelRouteRef | None] = field(default_factory=dict)
    """出生点标识 → 当前开启的入口段；缺少记录按关闭处理。"""

    def to_json(self) -> dict[str, Any]:
        """写入配置的形式。"""
        return {
            'room': None if self.room is None else self.room.to_json(),
            'entries': {
                spawn_id: None if ref is None else ref.to_json()
                for spawn_id, ref in self.entries.items()
            },
        }

    @classmethod
    def from_json(cls, data: Any) -> BagelKeycardSelection:
        """从配置还原。

        Raises:
            ValueError: 结构不对或引用无效。
        """
        if not isinstance(data, dict):
            raise ValueError('房间开启选择不是映射')
        room = data.get('room')
        entries = data.get('entries') or {}
        if not isinstance(entries, dict):
            raise ValueError('entries 不是映射')
        return cls(
            room=None if room is None else BagelRouteRef.from_json(room),
            entries={
                str(spawn_id): None if ref is None else BagelRouteRef.from_json(ref)
                for spawn_id, ref in entries.items()
            },
        )


@dataclass
class BagelRouteCatalog:
    """一次加载得到的全部出生点、房间、路线与开启选择。"""

    spawns: list[BagelSpawnPoint]
    routes: list[BagelRoute]
    active: dict[str, BagelRouteRef | None]
    """出生点标识 → 当前开启的跑刀路线引用；缺少记录按关闭处理。"""
    errors: list[str] = field(default_factory=list)
    """加载时跳过的损坏文件等问题，供界面提示。"""
    rooms: list[BagelKeycardRoom] = field(default_factory=list)
    """已登记的开房卡房间。"""
    keycard: dict[str, BagelKeycardSelection] = field(default_factory=dict)
    """房间标识 → 开房卡开启选择；缺少记录按全部关闭处理。"""

    def spawn(self, spawn_id: str) -> BagelSpawnPoint | None:
        """按标识找出生点。"""
        return next((item for item in self.spawns if item.spawn_id == spawn_id), None)

    def room(self, room_id: str) -> BagelKeycardRoom | None:
        """按标识找房间。"""
        return next((item for item in self.rooms if item.room_id == room_id), None)

    def route(self, ref: BagelRouteRef) -> BagelRoute | None:
        """按引用找路线。"""
        return next((item for item in self.routes if item.ref == ref), None)

    def active_ref(self, spawn_id: str) -> BagelRouteRef | None:
        """某个出生点当前开启的跑刀路线引用。"""
        return self.active.get(spawn_id)

    def keycard_selection(self, room_id: str) -> BagelKeycardSelection:
        """某个房间的开房卡开启选择；没有记录时返回空选择（不写回）。"""
        return self.keycard.get(room_id) or BagelKeycardSelection()

    def is_active(self, route: BagelRoute) -> bool:
        """路线是否处于开启状态。

        跑刀路线看出生点的开启选择；开房卡的段看所属房间的开启选择。
        """
        if route.keycard is None:
            return self.active.get(route.spawn_id) == route.ref
        selection = self.keycard_selection(route.keycard.room_id)
        if route.keycard.segment == SEGMENT_ROOM:
            return selection.room == route.ref
        return selection.entries.get(route.spawn_id) == route.ref

    def segment_problem(self, route: BagelRoute) -> str | None:
        """检查一条开房卡段本身能否使用：房间已登记、首末点能接上。

        Returns:
            问题原因；不是开房卡段或没有问题时返回 None。
        """
        if route.keycard is None:
            return None
        room = self.room(route.keycard.room_id)
        if room is None:
            return f'「{route.name}」所属房间 {route.keycard.room_id} 未登记'
        if route.keycard.segment == SEGMENT_ROOM:
            problem = check_room_segment(route, room)
        else:
            spawn = self.spawn(route.spawn_id)
            if spawn is None:
                return f'「{route.name}」绑定的出生点 {route.spawn_id} 未登记'
            problem = check_route_start(route, spawn) or check_entry_segment(route, room)
        return None if problem is None else f'「{route.name}」{problem}'

    def keycard_problem(self, room_id: str) -> str | None:
        """检查某个房间的开房卡开启选择能否执行。

        Returns:
            选择无效时的原因；全部关闭或有效时返回 None。
        """
        room = self.room(room_id)
        if room is None:
            return f'房间 {room_id} 未登记'
        selection = self.keycard_selection(room_id)
        problems: list[str] = []
        refs: list[tuple[BagelRouteRef, str, str]] = []
        if selection.room is not None:
            refs.append((selection.room, SEGMENT_ROOM, ''))
        refs.extend((ref, SEGMENT_ENTRY, spawn_id) for spawn_id, ref in selection.entries.items() if ref is not None)
        for ref, segment, spawn_id in refs:
            route = self.route(ref)
            if route is None:
                problems.append(f'{room.name}开启的路线 {ref} 不存在或已损坏')
                continue
            if route.keycard is None or route.keycard.room_id != room_id or route.keycard.segment != segment:
                problems.append(f'「{route.name}」不是{room.name}的{"房间段" if segment == SEGMENT_ROOM else "入口段"}')
                continue
            if segment == SEGMENT_ENTRY and route.spawn_id != spawn_id:
                problems.append(f'「{route.name}」绑定的出生点与开启位置不一致')
                continue
            problem = self.segment_problem(route)
            if problem is not None:
                problems.append(problem)
        if selection.room is None and any(ref is not None for ref in selection.entries.values()):
            problems.append(f'{room.name}开启了入口段，但没有开启房间段')
        return '；'.join(problems) if problems else None

    def selection_problem(self, spawn_id: str) -> str | None:
        """检查某个出生点的开启选择能否执行。

        Returns:
            选择无效时的原因；未开启或有效时返回 None。
        """
        ref = self.active.get(spawn_id)
        if ref is None:
            return None
        spawn = self.spawn(spawn_id)
        if spawn is None:
            return f'出生点 {spawn_id} 未登记'
        route = self.route(ref)
        if route is None:
            return f'{spawn.name}开启的路线 {ref} 不存在或已损坏'
        if route.spawn_id != spawn_id:
            return f'{spawn.name}开启的路线「{route.name}」绑定的是其他出生点'
        problem = check_route_start(route, spawn) or check_run_points(route)
        if problem is not None:
            return f'{spawn.name}开启的路线「{route.name}」无法开始：{problem}'
        return None

    def selection_problems(self) -> list[str]:
        """全部出生点开启选择中的问题。"""
        problems = [self.selection_problem(spawn_id) for spawn_id in self.active]
        return [item for item in problems if item is not None]

    def active_count(self) -> int:
        """开启了路线的出生点个数。"""
        return sum(1 for ref in self.active.values() if ref is not None)

    def keycard_route(self, room_id: str, spawn_id: str) -> BagelRoute | None:
        """按开启选择拼出某个出生点去某个房间的完整路线。

        Returns:
            拼好的路线；该出生点没有开启入口段时返回 None（调用方按「出生点不支持」退出重开）。

        Raises:
            BagelRouteConfigError: 房间未登记、房间段未开启或任一段无效。
        """
        problem = self.keycard_problem(room_id)
        if problem is not None:
            raise BagelRouteConfigError(problem)
        selection = self.keycard_selection(room_id)
        entry_ref = selection.entries.get(spawn_id)
        if entry_ref is None:
            return None
        entry = self.route(entry_ref)
        room_route = self.route(selection.room) if selection.room is not None else None
        if entry is None or room_route is None:
            raise BagelRouteConfigError('开房卡路线引用失效')
        return join_keycard_route(entry, room_route)


class BagelRouteStore:
    """路线文件与开启选择的读写入口。"""

    def __init__(self, data_dir: Path | None = None) -> None:
        """
        Args:
            data_dir: 数据目录；None 表示插件数据目录 plugin_data/<APP_ID>。离线核对时可传临时目录。
        """
        self.data_dir: Path = data_dir if data_dir is not None else get_plugin_data_dir()
        self.custom_dir: Path = self.data_dir / 'routes'
        self.sample_dir: Path = PLUGIN_DIR.joinpath(*SAMPLE_DIR)
        self.settings_path: Path = self.data_dir / 'route_settings.yml'
        self.spawn_path: Path = PLUGIN_DIR.joinpath(*SPAWN_FILE)
        self.room_path: Path = PLUGIN_DIR.joinpath(*ROOM_FILE)

    # ---------- 读取 ----------

    def load_spawns(self) -> list[BagelSpawnPoint]:
        """读出出生点清单。

        Raises:
            BagelRouteConfigError: 清单缺失或格式错误。
        """
        try:
            data = yaml.safe_load(self.spawn_path.read_text(encoding='utf-8'))
            spawns = [
                BagelSpawnPoint(
                    spawn_id=str(item['id']),
                    name=str(item['name']),
                    layer=str(item['layer']),
                    x=float(item['x']),
                    y=float(item['y']),
                    radius=float(item['radius']),
                    initial_route=item.get('initial_route') or None,
                )
                for item in data['spawn_points']
            ]
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as error:
            raise BagelRouteConfigError(f'出生点清单无法读取：{error}') from error
        ids = [item.spawn_id for item in spawns]
        if len(set(ids)) != len(ids):
            raise BagelRouteConfigError('出生点清单中有重复标识')
        return spawns

    def load_rooms(self) -> list[BagelKeycardRoom]:
        """读出开房卡房间清单；文件不存在时返回空列表。

        Raises:
            BagelRouteConfigError: 清单格式错误或标识重复。
        """
        if not self.room_path.is_file():
            return []
        try:
            data = yaml.safe_load(self.room_path.read_text(encoding='utf-8'))
            rooms = [
                BagelKeycardRoom(
                    room_id=str(item['id']),
                    name=str(item['name']),
                    layer=str(item['layer']),
                    x=float(item['x']),
                    y=float(item['y']),
                    radius=float(item['radius']),
                    initial_room_route=item.get('initial_room_route') or None,
                    initial_entry_routes=dict(item.get('initial_entry_routes') or {}),
                )
                for item in (data or {}).get('rooms') or []
            ]
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError, AttributeError) as error:
            raise BagelRouteConfigError(f'开房卡房间清单无法读取：{error}') from error
        ids = [item.room_id for item in rooms]
        if len(set(ids)) != len(ids):
            raise BagelRouteConfigError('开房卡房间清单中有重复标识')
        return rooms

    def load_routes(self) -> tuple[list[BagelRoute], list[str]]:
        """读出默认与自定义路线，损坏的文件跳过并记录原因。

        Returns:
            (路线列表, 问题列表)。默认路线在前，各自按名称排列。
        """
        routes: list[BagelRoute] = []
        errors: list[str] = []
        # 默认路线随插件走，自定义路线跟着用户的数据目录。
        for directory, source in ((self.sample_dir, SOURCE_DEFAULT), (self.custom_dir, SOURCE_CUSTOM)):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob(f'*{ROUTE_SUFFIX}')):
                try:
                    route = self._read_route(path, source)
                except (OSError, ValueError) as error:
                    log.warning('跳过损坏的贝果路线文件 %s：%s', path.name, error)
                    errors.append(f'{path.name}：{error}')
                    continue
                routes.append(route)
        routes.sort(key=lambda item: (item.source != SOURCE_DEFAULT, item.name))
        return routes, errors

    def load_catalog(self) -> BagelRouteCatalog:
        """读出出生点、路线与开启选择，并为未配置的房间开启默认段。

        Raises:
            BagelRouteConfigError: 出生点清单或开启选择配置损坏。
        """
        spawns = self.load_spawns()
        rooms = self.load_rooms()
        self._ensure_settings(spawns, rooms)
        routes, errors = self.load_routes()
        active, keycard = self._read_settings()
        # 已登记但没有记录的出生点按关闭处理；新发布的默认路线不抢占用户选择。
        for spawn in spawns:
            active.setdefault(spawn.spawn_id, None)
        catalog = BagelRouteCatalog(spawns, routes, active, errors, rooms, keycard)
        for spawn_id in list(active):
            if catalog.spawn(spawn_id) is None:
                errors.append(f'开启选择中的出生点 {spawn_id} 未登记，已忽略')
                del active[spawn_id]
        for room_id in list(keycard):
            if catalog.room(room_id) is None:
                errors.append(f'开房卡开启选择中的房间 {room_id} 未登记，已忽略')
                del keycard[room_id]
        return catalog

    def load_route(self, ref: BagelRouteRef) -> BagelRoute:
        """按引用读出一条路线。

        Raises:
            BagelRouteConfigError: 路线不存在或损坏。
        """
        path = self._route_path(ref)
        if not path.is_file():
            raise BagelRouteConfigError(f'路线不存在：{ref}')
        try:
            return self._read_route(path, ref.source)
        except (OSError, ValueError) as error:
            raise BagelRouteConfigError(f'路线 {ref} 无法读取：{error}') from error

    def active_route(self, spawn_id: str) -> BagelRoute | None:
        """读出某个出生点当前开启的路线快照。

        Returns:
            开启的路线；该出生点没有开启路线时返回 None。

        Raises:
            BagelRouteConfigError: 配置损坏，或开启的路线无效。
        """
        catalog = self.load_catalog()
        problem = catalog.selection_problem(spawn_id)
        if problem is not None:
            raise BagelRouteConfigError(problem)
        ref = catalog.active_ref(spawn_id)
        if ref is None:
            return None
        route = catalog.route(ref)
        return route.copy() if route is not None else None

    def keycard_route(self, room_id: str, spawn_id: str) -> BagelRoute | None:
        """读出某个出生点去某个房间的开房卡路线（入口段 + 房间段）。

        Returns:
            拼好的路线；该出生点没有开启入口段时返回 None。

        Raises:
            BagelRouteConfigError: 配置损坏，或开启的段无效。
        """
        return self.load_catalog().keycard_route(room_id, spawn_id)

    def trial_route(self, ref: BagelRouteRef) -> BagelRoute:
        """读出试跑用的路线。

        入口段单独走到房间入口就停，没有意义，所以试跑入口段时接上该房间当前开启的
        房间段一起跑；房间段与跑刀路线原样返回。

        Raises:
            BagelRouteConfigError: 路线不存在，或入口段所属房间没有可用房间段。
        """
        route = self.load_route(ref)
        if not route.is_entry_segment or route.keycard is None:
            return route
        catalog = self.load_catalog()
        problem = catalog.segment_problem(route)
        if problem is not None:
            raise BagelRouteConfigError(problem)
        room_id = route.keycard.room_id
        room_ref = catalog.keycard_selection(room_id).room
        room_route = catalog.route(room_ref) if room_ref is not None else None
        if room_route is None:
            room = catalog.room(room_id)
            raise BagelRouteConfigError(f'{room.name if room else room_id}没有开启房间段，入口段无法单独试跑')
        problem = catalog.segment_problem(room_route)
        if problem is not None:
            raise BagelRouteConfigError(problem)
        return join_keycard_route(route, room_route)

    # ---------- 写入 ----------

    def set_active_route(self, spawn_id: str, ref: BagelRouteRef | None) -> None:
        """设置某个出生点当前开启的路线；None 表示关闭该出生点的全部路线。

        Raises:
            BagelRouteConfigError: 出生点未登记、路线不存在或绑定的是其他出生点。
        """
        catalog = self.load_catalog()
        if catalog.spawn(spawn_id) is None:
            raise BagelRouteConfigError(f'出生点未登记：{spawn_id}')
        if ref is not None:
            route = catalog.route(ref)
            if route is None:
                raise BagelRouteConfigError(f'路线不存在：{ref}')
            if route.keycard is not None:
                raise BagelRouteConfigError(f'「{route.name}」是开房卡{route.keycard.label}，不能作为跑刀路线开启')
            if route.spawn_id != spawn_id:
                raise BagelRouteConfigError(f'路线「{route.name}」绑定的是其他出生点')
            problem = check_route_start(route, catalog.spawn(spawn_id)) or check_run_points(route)
            if problem is not None:
                raise BagelRouteConfigError(f'路线「{route.name}」无法开启：{problem}')
        catalog.active[spawn_id] = ref
        self._write_settings(catalog.active, catalog.keycard)

    def set_keycard_route(self, room_id: str, spawn_id: str | None, ref: BagelRouteRef | None) -> None:
        """设置某个房间的开房卡开启选择。

        Args:
            room_id: 房间标识。
            spawn_id: None 表示设置房间段；否则设置该出生点的入口段。
            ref: 要开启的段；None 表示关闭该位置。

        Raises:
            BagelRouteConfigError: 房间或出生点未登记、段类型不符或段本身无法使用。
        """
        catalog = self.load_catalog()
        room = catalog.room(room_id)
        if room is None:
            raise BagelRouteConfigError(f'房间未登记：{room_id}')
        if spawn_id is not None and catalog.spawn(spawn_id) is None:
            raise BagelRouteConfigError(f'出生点未登记：{spawn_id}')
        segment = SEGMENT_ROOM if spawn_id is None else SEGMENT_ENTRY
        if ref is not None:
            route = catalog.route(ref)
            if route is None:
                raise BagelRouteConfigError(f'路线不存在：{ref}')
            label = '房间段' if segment == SEGMENT_ROOM else '入口段'
            if route.keycard is None or route.keycard.room_id != room_id or route.keycard.segment != segment:
                raise BagelRouteConfigError(f'「{route.name}」不是{room.name}的{label}')
            if spawn_id is not None and route.spawn_id != spawn_id:
                raise BagelRouteConfigError(f'入口段「{route.name}」绑定的是其他出生点')
            problem = catalog.segment_problem(route)
            if problem is not None:
                raise BagelRouteConfigError(f'无法开启：{problem}')
        selection = catalog.keycard.setdefault(room_id, BagelKeycardSelection())
        if spawn_id is None:
            selection.room = ref
        else:
            selection.entries[spawn_id] = ref
        self._write_settings(catalog.active, catalog.keycard)

    def create_custom_route(
        self, name: str, spawn_id: str,
        points: list[BagelRoutePoint] | None = None, activate: bool = False,
        started_at: str | None = None, keycard: BagelKeycardTag | None = None,
    ) -> BagelRoute:
        """新建一条自定义路线。

        Args:
            name: 显示名称，不能与已有路线重名。
            spawn_id: 绑定的出生点；开房卡房间段不绑定出生点，传空串。
            points: 点位；None 表示只放一个起点途径点：跑刀路线与入口段放在出生点，
                房间段放在房间入口。
            activate: 创建后是否开启，开启会替换同一位置原来开启的路线。
                开房卡的段要能接上房间入口才开得了，所以通常在补完点位后再开启。
            started_at: 路线开始时间；None 表示当前时间。
            keycard: 开房卡段标记；None 表示跑刀路线。

        Returns:
            已保存的路线。

        Raises:
            ValueError: 名称无效或重名。
            BagelRouteConfigError: 出生点或房间未登记、写入失败。
        """
        catalog = self.load_catalog()
        anchor: BagelSpawnPoint | None
        if keycard is not None:
            room = catalog.room(keycard.room_id)
            if room is None:
                raise BagelRouteConfigError(f'房间未登记：{keycard.room_id}')
            if keycard.segment == SEGMENT_ROOM:
                spawn_id = ''
                anchor = room.as_spawn()
            else:
                anchor = catalog.spawn(spawn_id)
        else:
            anchor = catalog.spawn(spawn_id)
        if anchor is None:
            raise BagelRouteConfigError(f'出生点未登记：{spawn_id}')
        name = self._unique_name_check(catalog, name)
        if points is None:
            points = [BagelRoutePoint(round(anchor.x), round(anchor.y), anchor.layer, 'beacon')]
        route = BagelRoute(
            route_id=self._new_custom_id(),
            name=name,
            spawn_id=spawn_id,
            source=SOURCE_CUSTOM,
            started_at=started_at or now_iso(),
            ended_at=now_iso(),
            points=list(points),
            keycard=keycard,
        )
        self._write_route(route)
        if activate:
            if keycard is None:
                self.set_active_route(spawn_id, route.ref)
            else:
                # 开房卡的段要能接上房间入口才开得了；开不了时保留路线、保持关闭，不让保存失败。
                try:
                    self.set_keycard_route(
                        keycard.room_id, None if keycard.segment == SEGMENT_ROOM else spawn_id, route.ref,
                    )
                except BagelRouteConfigError as error:
                    log.warning('贝果开房卡%s「%s」已保存，但未开启：%s', keycard.label, route.name, error)
        return route

    def copy_route(self, ref: BagelRouteRef, name: str) -> BagelRoute:
        """复制一条路线为新的自定义路线，副本初始关闭。

        Raises:
            ValueError: 名称无效或重名。
            BagelRouteConfigError: 源路线不存在或写入失败。
        """
        source = self.load_route(ref)
        return self.create_custom_route(
            name, source.spawn_id, points=list(source.points), keycard=source.keycard,
        )

    def save_custom_route(self, route_id: str, points: list[BagelRoutePoint]) -> BagelRoute:
        """保存自定义路线的节点；默认路线拒绝写入。

        Raises:
            BagelRouteConfigError: 不是自定义路线、路线不存在或写入失败。
            ValueError: 节点无效。
        """
        ref = self._custom_ref(route_id)
        route = self.load_route(ref)
        spawn = self._start_anchor(route)
        route.points = list(points)
        problem = check_route_start(route, spawn) or check_run_points(route)
        if problem is not None:
            raise ValueError(problem)
        route.ended_at = now_iso()
        self._write_route(route)
        return route

    def rename_custom_route(self, route_id: str, name: str) -> BagelRoute:
        """修改自定义路线的显示名称；标识与文件名不变。

        Raises:
            ValueError: 名称无效或重名。
            BagelRouteConfigError: 不是自定义路线或路线不存在。
        """
        ref = self._custom_ref(route_id)
        catalog = self.load_catalog()
        route = catalog.route(ref)
        if route is None:
            raise BagelRouteConfigError(f'路线不存在：{ref}')
        route.name = self._unique_name_check(catalog, name, ignore=ref)
        self._write_route(route)
        return route

    def delete_custom_route(self, route_id: str) -> None:
        """删除一条自定义路线。

        路线正在开启时先把对应出生点置为关闭，再删除文件；不自动开启其他路线。
        删除失败时开启选择保持关闭，避免留下指向已删除文件的引用。

        Raises:
            BagelRouteConfigError: 不是自定义路线、路线不存在或删除失败。
        """
        ref = self._custom_ref(route_id)
        path = self._route_path(ref)
        if not path.is_file():
            raise BagelRouteConfigError(f'路线不存在：{ref}')
        catalog = self.load_catalog()
        changed = False
        for spawn_id, active in catalog.active.items():
            if active == ref:
                catalog.active[spawn_id] = None
                changed = True
        for selection in catalog.keycard.values():
            if selection.room == ref:
                selection.room = None
                changed = True
            for spawn_id, active in selection.entries.items():
                if active == ref:
                    selection.entries[spawn_id] = None
                    changed = True
        if changed:
            self._write_settings(catalog.active, catalog.keycard)
        try:
            path.unlink()
        except OSError as error:
            raise BagelRouteConfigError(f'删除路线文件失败：{error}') from error

    # ---------- 初始化 ----------

    def _ensure_settings(self, spawns: list[BagelSpawnPoint], rooms: list[BagelKeycardRoom]) -> None:
        """首次使用时初始化；升级时为尚未配置的房间开启默认段。

        已有房间记录保留，包括明确关闭的选择。损坏配置由读取时报错，不重新初始化。
        """
        if self.settings_path.exists():
            active, keycard = self._read_settings()
            missing_rooms = [room for room in rooms if room.room_id not in keycard]
            if not missing_rooms:
                return
            defaults = self._initial_keycard_selection(spawns, missing_rooms)
            if defaults:
                keycard.update(defaults)
                self._write_settings(active, keycard)
                log.info('贝果路线：为 %d 个未配置的房间开启默认段', len(defaults))
            return
        keycard = self._initial_keycard_selection(spawns, rooms)
        self._write_settings(self._initial_selection(spawns), keycard)

    def _initial_keycard_selection(
        self, spawns: list[BagelSpawnPoint], rooms: list[BagelKeycardRoom],
    ) -> dict[str, BagelKeycardSelection]:
        """从房间清单生成通过段校验的默认开启选择。"""
        routes, _errors = self.load_routes()
        catalog = BagelRouteCatalog(spawns, routes, {}, rooms=rooms)
        selections: dict[str, BagelKeycardSelection] = {}
        for room in rooms:
            if not room.initial_room_route:
                continue
            room_ref = BagelRouteRef(SOURCE_DEFAULT, room.initial_room_route)
            room_route = catalog.route(room_ref)
            if (
                room_route is None
                or room_route.keycard != BagelKeycardTag(room.room_id, SEGMENT_ROOM)
                or catalog.segment_problem(room_route) is not None
            ):
                continue
            selection = BagelKeycardSelection(room=room_ref)
            for spawn_id, route_id in room.initial_entry_routes.items():
                ref = BagelRouteRef(SOURCE_DEFAULT, route_id)
                route = catalog.route(ref)
                if (
                    route is not None
                    and route.spawn_id == spawn_id
                    and route.keycard == BagelKeycardTag(room.room_id, SEGMENT_ENTRY)
                    and catalog.segment_problem(route) is None
                ):
                    selection.entries[spawn_id] = ref
            selections[room.room_id] = selection
        return selections

    def _initial_selection(self, spawns: list[BagelSpawnPoint]) -> dict[str, BagelRouteRef | None]:
        """首次安装的开启选择：只引用存在的默认路线。"""
        routes, _errors = self.load_routes()
        defaults = {route.route_id: route for route in routes if route.source == SOURCE_DEFAULT}
        active: dict[str, BagelRouteRef | None] = {}
        for spawn in spawns:
            route = defaults.get(spawn.initial_route) if spawn.initial_route else None
            if route is not None and route.spawn_id == spawn.spawn_id:
                active[spawn.spawn_id] = route.ref
            else:
                active[spawn.spawn_id] = None
        log.info('贝果路线：首次使用，按出生点清单初始化开启选择')
        return active

    # ---------- 内部 ----------

    def _start_anchor(self, route: BagelRoute) -> BagelSpawnPoint:
        """路线首点要对齐的位置：房间段对齐房间入口，其余对齐绑定的出生点。

        Raises:
            BagelRouteConfigError: 出生点或房间未登记。
        """
        if route.is_room_segment and route.keycard is not None:
            room = next((item for item in self.load_rooms() if item.room_id == route.keycard.room_id), None)
            if room is None:
                raise BagelRouteConfigError(f'房间段所属房间未登记：{route.keycard.room_id}')
            return room.as_spawn()
        spawn = next((item for item in self.load_spawns() if item.spawn_id == route.spawn_id), None)
        if spawn is None:
            raise BagelRouteConfigError(f'路线绑定的出生点未登记：{route.spawn_id}')
        return spawn

    def _route_path(self, ref: BagelRouteRef) -> Path:
        """引用对应的文件路径；标识不合法时拒绝。"""
        if ref.source == SOURCE_CUSTOM:
            if not CUSTOM_ID_PATTERN.match(ref.route_id):
                raise BagelRouteConfigError(f'自定义路线标识不合法：{ref.route_id}')
            return self.custom_dir / f'{ref.route_id}{ROUTE_SUFFIX}'
        if any(ch in ref.route_id for ch in '\\/:*?"<>|') or ref.route_id in ('.', '..'):
            raise BagelRouteConfigError(f'默认路线标识不合法：{ref.route_id}')
        return self.sample_dir / f'{ref.route_id}{SAMPLE_SUFFIX}'

    def _custom_ref(self, route_id: str) -> BagelRouteRef:
        """把标识当作自定义路线引用；默认路线或非法标识一律拒绝写入。"""
        if not CUSTOM_ID_PATTERN.match(route_id):
            raise BagelRouteConfigError('默认路线只读，复制后再修改')
        return BagelRouteRef(SOURCE_CUSTOM, route_id)

    def _read_route(self, path: Path, source: str) -> BagelRoute:
        """读出一个路线文件，并核对文件名与标识一致。"""
        route = BagelRoute.from_dict(json.loads(path.read_text(encoding='utf-8')), source)
        suffix = SAMPLE_SUFFIX if source == SOURCE_DEFAULT else ROUTE_SUFFIX
        if path.name != f'{route.route_id}{suffix}':
            raise ValueError(f'文件名与路线标识 {route.route_id} 不一致')
        if source == SOURCE_CUSTOM and not CUSTOM_ID_PATTERN.match(route.route_id):
            raise ValueError(f'自定义路线标识不合法：{route.route_id}')
        return route

    def _write_route(self, route: BagelRoute) -> None:
        """写入自定义路线：同目录临时文件写入、读回核对后替换。"""
        if route.source != SOURCE_CUSTOM:
            raise BagelRouteConfigError('默认路线只读，复制后再修改')
        path = self._route_path(route.ref)
        text = json.dumps(route.to_dict(), ensure_ascii=False, indent=1)

        def check(tmp: Path) -> None:
            reloaded = BagelRoute.from_dict(json.loads(tmp.read_text(encoding='utf-8')), SOURCE_CUSTOM)
            if len(reloaded.points) != len(route.points):
                raise ValueError(
                    f'读回点数不一致：写入 {len(route.points)}，读回 {len(reloaded.points)}',
                )

        _atomic_write(path, text, check)

    def _read_settings(
        self,
    ) -> tuple[dict[str, BagelRouteRef | None], dict[str, BagelKeycardSelection]]:
        """读出跑刀与开房卡的开启选择；损坏时报错，不回退到默认值。

        ``keycard_routes`` 是后加的可选项，缺少的房间由初始化与升级步骤补齐默认选择。
        """
        try:
            data = yaml.safe_load(self.settings_path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or data.get('version') != SETTINGS_VERSION:
                raise ValueError('格式或版本不正确')
            raw = data.get('active_routes') or {}
            if not isinstance(raw, dict):
                raise ValueError('active_routes 不是映射')
            raw_keycard = data.get('keycard_routes') or {}
            if not isinstance(raw_keycard, dict):
                raise ValueError('keycard_routes 不是映射')
            active = {
                str(spawn_id): None if value is None else BagelRouteRef.from_json(value)
                for spawn_id, value in raw.items()
            }
            keycard = {
                str(room_id): BagelKeycardSelection.from_json(value)
                for room_id, value in raw_keycard.items()
            }
            return active, keycard
        except (OSError, yaml.YAMLError, ValueError) as error:
            raise BagelRouteConfigError(
                f'路线开启配置 {self.settings_path.name} 已损坏：{error}',
            ) from error

    def _write_settings(
        self, active: dict[str, BagelRouteRef | None], keycard: dict[str, BagelKeycardSelection],
    ) -> None:
        """一次写入整个开启选择；没有任何开房卡选择时不写 ``keycard_routes``。"""
        data: dict[str, Any] = {
            'version': SETTINGS_VERSION,
            'active_routes': {
                spawn_id: None if ref is None else ref.to_json()
                for spawn_id, ref in active.items()
            },
        }
        if keycard:
            data['keycard_routes'] = {room_id: selection.to_json() for room_id, selection in keycard.items()}
        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)

        def check(tmp: Path) -> None:
            if yaml.safe_load(tmp.read_text(encoding='utf-8')) != data:
                raise ValueError('读回内容不一致')

        _atomic_write(self.settings_path, text, check)

    def _new_custom_id(self) -> str:
        """生成一个未被占用的自定义路线标识。"""
        while True:
            route_id = f'{CUSTOM_PREFIX}{datetime.now().strftime("%Y%m%d")}_{uuid.uuid4().hex[:8]}'
            if not (self.custom_dir / f'{route_id}{ROUTE_SUFFIX}').exists():
                return route_id

    @staticmethod
    def _unique_name_check(
        catalog: BagelRouteCatalog, name: str, ignore: BagelRouteRef | None = None,
    ) -> str:
        """校验名称合法且不与其他路线重名。"""
        name = validate_route_name(name)
        if any(route.name == name and route.ref != ignore for route in catalog.routes):
            raise ValueError(f'已经有同名路线：{name}')
        return name


def _atomic_write(path: Path, text: str, check: Any) -> None:
    """同目录临时文件写入，核对后原子替换；失败时保留原文件。

    Args:
        path: 目标文件。
        text: 文件内容。
        check: 接收临时文件路径的核对函数，不通过时抛异常。

    Raises:
        BagelRouteConfigError: 写入或核对失败。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        check(tmp)
        os.replace(tmp, path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        tmp.unlink(missing_ok=True)
        raise BagelRouteConfigError(f'写入 {path.name} 失败：{error}') from error
