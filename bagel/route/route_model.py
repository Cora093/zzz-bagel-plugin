"""贝果路线与出生点的数据定义及文件格式转换。

本模块只描述数据，不读写磁盘、不依赖 Qt 或录制器。读写与校验规则在
``route_store``，出生点匹配在 ``spawn``。

- 出生点：入场时角色所在的固定位置，独立于路线首点登记。
- 路线：一串有序点位，顺序本身就是路线；每条路线绑定一个出生点。
- 路线来源：默认路线随程序发布、内容只读；自定义路线由用户新增、录制或复制。
- 开房卡路线分两段：入口段从出生点走到房间入口，绑定出生点；房间段从房间入口
  刷卡进门搜完容器，绑定房间、不绑定出生点。执行前按「入口段 + 房间段」拼成一条。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from math import hypot
from typing import Any

ROUTE_VERSION: int = 2
"""路线文件格式版本。v2 起路线必须绑定出生点，开关状态不再写进路线文件。"""

SOURCE_DEFAULT: str = 'default'
"""默认路线：随程序发布，内容只读。"""

SOURCE_CUSTOM: str = 'custom'
"""自定义路线：用户新增、录制或复制，可编辑。"""

SOURCES: tuple[str, ...] = (SOURCE_DEFAULT, SOURCE_CUSTOM)

BEACON: str = 'beacon'
"""途径点：纯路点，路过即可，不交互。"""

ELEVATOR: str = 'elevator'
"""电梯类型标识。进门后可能出楼层选项，也可能不出。"""

ELEVATOR_EXIT_FORWARD: str = 'forward'
ELEVATOR_EXIT_BACKWARD: str = 'backward'
"""电梯出来时朝哪个方向走。"""

KEYCARD_DOOR: str = 'keycard_door'
"""房卡门：要刷门禁卡（长按 F）才开的门。

和普通门分开：只有这类点识别刷卡提示并长按刷卡；其它点遇到刷卡提示一律忽略，
免得跑刀路线零携带经过刷卡处时读到「0/1」把任务停掉。
"""

POINT_KINDS: dict[str, tuple[str, str]] = {
    '1': (BEACON, '途径点'),
    '2': ('electronic_safe', '电子保险箱'),
    '3': ('mechanical_safe', '机械保险箱'),
    '4': ('box', '普通箱子'),
    '5': ('door', '门'),
    '6': (ELEVATOR, '电梯'),
    '7': (KEYCARD_DOOR, '房卡门'),
}
"""打点键到（类型标识, 中文名）的映射。"""

BOX_CATEGORIES: dict[str, str] = {
    'cardboard_box': '纸箱',
    'crate': '板条箱',
    'trash_can': '垃圾桶',
    'medical_station': '医疗站',
    'suitcase': '行李箱',
    'handbag': '提包',
    'cabinet': '单斗柜',
}
"""普通箱子的具体容器类别，只用于在路线内容里标注。

这几种容器的搜索面板、格子和「全部拾取」按钮位置都一样，执行时与普通箱子
完全相同；分类只记下这里是什么容器，不改变走法和收集规则。
"""

KIND_ALIASES: dict[str, str] = {
    'briefcase': 'box',
}
"""旧路线文件里的类型名到当前类型名；改名后仍能读回。"""

KIND_LABELS: dict[str, str] = dict(POINT_KINDS.values())
KIND_LABELS.update(BOX_CATEGORIES)
"""类型标识到中文名，供界面与网页显示；普通箱子的具体类别也在这里。"""

KIND_COLORS: dict[str, str] = {
    BEACON: '#2233b4',
    'electronic_safe': '#d97400',
    'mechanical_safe': '#8b3fc4',
    'box': '#1f8a4c',
    **dict.fromkeys(BOX_CATEGORIES, '#1f8a4c'),
    'door': '#238495',
    ELEVATOR: '#bc2f5e',
    KEYCARD_DOOR: '#518a10',
}
"""类型标识到标注色，与网页画点用的颜色是同一套。

电子=橙、机械=紫、普通箱=绿是语义色，其余四类按色相环拉开，
任意两类的色相至少差 41 度，肉眼可辨。房卡门取黄绿（约 88 度），
落在电子橙（约 32 度）与普通箱绿（约 145 度）之间。普通箱子的具体类别
沿用普通箱的绿色：执行上没有区别，不需要另配颜色。
"""

POINT_MIN_DISTANCE: float = 8.0
"""同类型点与上一个点的最小间距；原地重复按不产生新点。"""

COORDINATE_LIMIT: float = 4000.0
"""点坐标的绝对值上限，防止网页或损坏文件传来离谱数值把路线写坏。"""

ROUTE_NAME_LIMIT: int = 40
"""路线显示名称的最大字符数。"""

SEGMENT_ENTRY: str = 'entry'
"""开房卡入口段：出生点 → 房间入口，绑定出生点。"""

SEGMENT_ROOM: str = 'room'
"""开房卡房间段：房间入口 → 刷卡进门 → 搜完容器，绑定房间，不绑定出生点。"""

SEGMENT_LABELS: dict[str, str] = {SEGMENT_ENTRY: '入口段', SEGMENT_ROOM: '房间段'}
"""段类型到中文名。"""


def now_iso() -> str:
    """当前本地时间的 ISO 文本，精确到秒。"""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def map_id_of(layer: str) -> str:
    """完整分层名中的地图标识，如 ``雅努斯幻境/Custom`` → ``雅努斯幻境``。"""
    return layer.split('/')[0]


@dataclass(frozen=True)
class BagelRoutePoint:
    """一个打点：全局坐标、所在分层与类型。

    点按打点顺序排列，顺序本身就是路线，不需要额外的连线数据。
    电梯点额外带两项进出电梯的参数。
    """

    x: float
    y: float
    layer: str
    kind: str
    elevator_option: str = ''
    """电梯要选的楼层文字，如「负二层站台」；空表示不选（直接进门）。"""
    elevator_exit: str = ELEVATOR_EXIT_FORWARD
    """出电梯往哪个方向走：forward 前进、backward 后退。"""

    def to_dict(self) -> dict[str, Any]:
        """生成可写入 JSON 的点。"""
        data: dict[str, Any] = {
            'x': round(self.x),
            'y': round(self.y),
            'layer': self.layer,
            'kind': self.kind,
        }
        if self.kind == ELEVATOR:
            data['elevator_option'] = self.elevator_option
            data['elevator_exit'] = self.elevator_exit
        return data

    @classmethod
    def from_dict(cls, item: Any) -> BagelRoutePoint:
        """校验并还原一个点；文件与网页传来的点都走这里。

        Args:
            item: 点位对象。

        Returns:
            校验后的点，坐标取整。

        Raises:
            ValueError: 字段缺失、类型不支持或坐标越界。
        """
        if not isinstance(item, dict):
            raise ValueError('点位不是对象')
        try:
            kind = KIND_ALIASES.get(str(item['kind']), str(item['kind']))
            layer = str(item['layer'])
            x = float(item['x'])
            y = float(item['y'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'点位字段缺失或不是数字：{error}') from error
        if kind not in KIND_LABELS:
            raise ValueError(f'不支持的点的类型：{kind}')
        if not layer or '/' not in layer:
            raise ValueError(f'点位分层不完整：{layer}')
        if not (-COORDINATE_LIMIT <= x <= COORDINATE_LIMIT):
            raise ValueError(f'x 越界：{x}')
        if not (-COORDINATE_LIMIT <= y <= COORDINATE_LIMIT):
            raise ValueError(f'y 越界：{y}')
        option = str(item.get('elevator_option', '')).strip()
        exit_direction = str(item.get('elevator_exit', ELEVATOR_EXIT_FORWARD))
        if exit_direction not in (ELEVATOR_EXIT_FORWARD, ELEVATOR_EXIT_BACKWARD):
            raise ValueError(f'电梯出梯方向不支持：{exit_direction}')
        if kind != ELEVATOR:
            option = ''
            exit_direction = ELEVATOR_EXIT_FORWARD
        return cls(
            float(round(x)), float(round(y)), layer, kind,
            elevator_option=option, elevator_exit=exit_direction,
        )


@dataclass(frozen=True)
class BagelSpawnPoint:
    """一个已登记的出生点。

    出生点独立于路线首点：移动路线首点不改变出生点位置。
    """

    spawn_id: str
    """稳定标识，路线和开启选择都用它引用出生点。"""
    name: str
    """显示名称。"""
    layer: str
    """完整分层名，如 ``雅努斯幻境/Custom``。"""
    x: float
    y: float
    """全局坐标。"""
    radius: float
    """识别半径（地图像素）：定位结果与出生点坐标的距离不超过它才算命中。"""
    initial_route: str | None = None
    """首次安装时默认开启的默认路线标识；None 表示首次安装时不开启任何路线。"""

    @property
    def map_id(self) -> str:
        """出生点所在地图。"""
        return map_id_of(self.layer)

    def distance_to(self, x: float, y: float) -> float:
        """与某个全局坐标的距离。"""
        return hypot(self.x - x, self.y - y)

    def to_json(self) -> dict[str, Any]:
        """网页使用的出生点数据。"""
        return {
            'spawn_id': self.spawn_id,
            'name': self.name,
            'layer': self.layer,
            'map_id': self.map_id,
            'x': round(self.x),
            'y': round(self.y),
            'radius': self.radius,
        }


@dataclass(frozen=True)
class BagelKeycardRoom:
    """一个已登记的开房卡房间。

    房间入口是入口段与房间段的接缝：入口段末点、房间段首点都要落在入口附近。
    入口取在房卡门外一小段的途径点上，不取房卡门本身，让刷卡点留在房间段里。
    """

    room_id: str
    """稳定标识，路线文件与开启选择都用它引用房间。"""
    name: str
    """显示名称，与游戏刷卡提示的房间名一致，如「风蚀公路汽车旅馆二层」。"""
    layer: str
    """房间入口所在完整分层。"""
    x: float
    y: float
    """房间入口的全局坐标。"""
    radius: float
    """接缝半径（地图像素）：入口段末点、房间段首点与入口的距离不超过它才算接上。
    单独试跑房间段时，角色也要站在入口这个半径内。"""
    initial_room_route: str | None = None
    """新装或升级后尚未配置的房间自动开启的默认房间段；None 表示不开启。"""
    initial_entry_routes: dict[str, str] = field(default_factory=dict)
    """随默认房间段自动开启的入口段：出生点标识 → 路线标识。"""

    @property
    def map_id(self) -> str:
        """房间所在地图。"""
        return map_id_of(self.layer)

    def distance_to(self, x: float, y: float) -> float:
        """与某个全局坐标的距离。"""
        return hypot(self.x - x, self.y - y)

    def as_spawn(self) -> BagelSpawnPoint:
        """把房间入口当作出生点，供单独试跑和录制房间段时锚定起点。"""
        return BagelSpawnPoint(
            spawn_id=room_anchor_id(self.room_id),
            name=f'{self.name}入口',
            layer=self.layer,
            x=self.x,
            y=self.y,
            radius=self.radius,
        )

    def to_json(self) -> dict[str, Any]:
        """网页使用的房间数据。"""
        return {
            'room_id': self.room_id,
            'name': self.name,
            'layer': self.layer,
            'map_id': self.map_id,
            'x': round(self.x),
            'y': round(self.y),
            'radius': self.radius,
        }


def room_anchor_id(room_id: str) -> str:
    """房间入口当作出生点时使用的标识；与真实出生点标识不会重名。"""
    return f'keycard_room:{room_id}'


@dataclass(frozen=True)
class BagelKeycardTag:
    """开房卡路线的段标记，写在路线文件的 ``keycard`` 字段。"""

    room_id: str
    """所属房间。"""
    segment: str
    """段类型：``SEGMENT_ENTRY`` 或 ``SEGMENT_ROOM``。"""

    def to_json(self) -> dict[str, str]:
        """写入文件或推给网页的形式。"""
        return {'room': self.room_id, 'segment': self.segment}

    @classmethod
    def from_json(cls, data: Any) -> BagelKeycardTag:
        """从文件或网页还原段标记。

        Raises:
            ValueError: 字段缺失或段类型不支持。
        """
        if not isinstance(data, dict):
            raise ValueError('开房卡段标记不是对象')
        room_id = str(data.get('room', '')).strip()
        segment = str(data.get('segment', ''))
        if not room_id:
            raise ValueError('开房卡段标记缺少房间')
        if segment not in SEGMENT_LABELS:
            raise ValueError(f'不支持的开房卡段类型：{segment}')
        return cls(room_id, segment)

    @property
    def label(self) -> str:
        """段类型中文名。"""
        return SEGMENT_LABELS[self.segment]


@dataclass(frozen=True)
class BagelRouteRef:
    """对一条路线的引用：来源加路线标识。

    默认与自定义路线分开存放，同一个标识只在各自来源内唯一，所以引用必须带来源。
    """

    source: str
    route_id: str

    def to_json(self) -> dict[str, str]:
        """写入配置或推给网页的形式。"""
        return {'source': self.source, 'route_id': self.route_id}

    @classmethod
    def from_json(cls, data: Any) -> BagelRouteRef:
        """从配置或网页还原引用。

        Raises:
            ValueError: 来源不支持或标识为空。
        """
        if not isinstance(data, dict):
            raise ValueError('路线引用不是对象')
        source = str(data.get('source', ''))
        route_id = str(data.get('route_id', '')).strip()
        if source not in SOURCES:
            raise ValueError(f'不支持的路线来源：{source}')
        if not route_id:
            raise ValueError('路线引用缺少标识')
        return cls(source, route_id)

    def __str__(self) -> str:
        return f'{self.source}:{self.route_id}'


@dataclass
class BagelRoute:
    """一条打点路线：有序点位，顺序即路径。"""

    route_id: str
    """路线标识，创建后不变，与文件名主干一致。"""
    name: str
    """显示名称；自定义路线可以改名，改名不改标识。"""
    spawn_id: str
    """绑定的出生点标识；开房卡房间段不绑定出生点，为空串。"""
    source: str = SOURCE_CUSTOM
    """来源，由实际文件所在位置决定，不写进文件。"""
    started_at: str = field(default_factory=now_iso)
    ended_at: str | None = None
    points: list[BagelRoutePoint] = field(default_factory=list)
    keycard: BagelKeycardTag | None = None
    """开房卡段标记；跑刀路线为 None。"""

    @property
    def is_room_segment(self) -> bool:
        """是否为开房卡房间段。"""
        return self.keycard is not None and self.keycard.segment == SEGMENT_ROOM

    @property
    def is_entry_segment(self) -> bool:
        """是否为开房卡入口段。"""
        return self.keycard is not None and self.keycard.segment == SEGMENT_ENTRY

    @property
    def ref(self) -> BagelRouteRef:
        """指向本路线的引用。"""
        return BagelRouteRef(self.source, self.route_id)

    @property
    def map_id(self) -> str:
        """路线首点所在地图；没有点时为空串。"""
        return map_id_of(self.points[0].layer) if self.points else ''

    def append_point(self, point: BagelRoutePoint) -> bool:
        """追加一个点；同类型且与上一个点太近时忽略。

        Args:
            point: 待追加的点。

        Returns:
            真正加入时返回 True。
        """
        if self.points:
            last = self.points[-1]
            if (
                last.kind == point.kind
                and last.layer == point.layer
                and hypot(last.x - point.x, last.y - point.y) < POINT_MIN_DISTANCE
            ):
                return False
        self.points.append(point)
        return True

    def pop_point(self) -> BagelRoutePoint | None:
        """删掉最后一个点，没有点时返回 None。"""
        if not self.points:
            return None
        return self.points.pop()

    def copy(self) -> BagelRoute:
        """生成独立副本，点位列表不与原路线共享。"""
        return replace(self, points=list(self.points))

    def to_dict(self) -> dict[str, Any]:
        """生成待写入文件的数据；来源由文件位置决定，不写入。跑刀路线不写 ``keycard``。"""
        data: dict[str, Any] = {
            'version': ROUTE_VERSION,
            'route_id': self.route_id,
            'name': self.name,
            'spawn_id': self.spawn_id,
            'started_at': self.started_at,
            'ended_at': self.ended_at,
        }
        if self.keycard is not None:
            data['keycard'] = self.keycard.to_json()
        data['points'] = [point.to_dict() for point in self.points]
        return data

    def to_json(self, active: bool) -> dict[str, Any]:
        """网页使用的路线数据。

        Args:
            active: 是否为所绑定出生点当前开启的路线。
        """
        return {
            'source': self.source,
            'route_id': self.route_id,
            'name': self.name,
            'spawn_id': self.spawn_id,
            'keycard': None if self.keycard is None else self.keycard.to_json(),
            'active': active,
            'points': [point.to_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, data: Any, source: str) -> BagelRoute:
        """从 v2 文件数据还原路线。

        Args:
            data: 文件内容。
            source: 文件所在位置决定的来源。

        Raises:
            ValueError: 版本不支持或数据不完整。
        """
        if not isinstance(data, dict):
            raise ValueError('路线文件不是对象')
        if data.get('version') != ROUTE_VERSION:
            raise ValueError(f'不支持的路线文件版本：{data.get("version")}')
        route_id = str(data.get('route_id', '')).strip()
        name = str(data.get('name', '')).strip()
        spawn_id = str(data.get('spawn_id', '') or '').strip()
        raw_keycard = data.get('keycard')
        keycard = None if raw_keycard is None else BagelKeycardTag.from_json(raw_keycard)
        if not route_id or not name:
            raise ValueError('路线缺少标识或名称')
        # 房间段不绑定出生点，其余路线必须绑定。
        if keycard is not None and keycard.segment == SEGMENT_ROOM:
            if spawn_id:
                raise ValueError('开房卡房间段不能绑定出生点')
        elif not spawn_id:
            raise ValueError('路线缺少出生点')
        raw_points = data.get('points')
        if not isinstance(raw_points, list):
            raise ValueError('路线点位不是数组')
        return cls(
            route_id=route_id,
            name=name,
            spawn_id=spawn_id,
            source=source,
            started_at=str(data.get('started_at') or now_iso()),
            ended_at=data.get('ended_at'),
            points=[BagelRoutePoint.from_dict(item) for item in raw_points],
            keycard=keycard,
        )


def validate_route_name(name: str) -> str:
    """校验路线显示名称并去掉首尾空白。

    Raises:
        ValueError: 名称为空或过长。
    """
    name = name.strip()
    if not name:
        raise ValueError('路线名称不能为空')
    if len(name) > ROUTE_NAME_LIMIT:
        raise ValueError(f'路线名称不能超过 {ROUTE_NAME_LIMIT} 个字符')
    return name


def check_route_start(route: BagelRoute, spawn: BagelSpawnPoint) -> str | None:
    """检查路线首点能否从出生点开始执行。

    首点必须是途径点，与出生点在同一完整分层，且位于出生点识别半径内。

    Args:
        route: 待检查的路线。
        spawn: 路线绑定的出生点。

    Returns:
        不满足时返回原因；满足时返回 None。
    """
    if not route.points:
        return '路线没有点'
    first = route.points[0]
    if first.kind != BEACON:
        return f'首点必须是途径点，当前是{KIND_LABELS.get(first.kind, first.kind)}'
    if first.layer != spawn.layer:
        return f'首点分层 {first.layer} 与{spawn.name}分层 {spawn.layer} 不一致'
    distance = spawn.distance_to(first.x, first.y)
    if distance > spawn.radius:
        return f'首点距{spawn.name} {distance:.0f}px，超出识别半径 {spawn.radius:.0f}px'
    return None


def check_run_points(route: BagelRoute) -> str | None:
    """检查跑刀路线的点类型：不允许房卡门点。

    房卡门点会识别刷卡提示，零携带时读到持卡计数 0 就按没带卡停止任务；
    刷卡只属于开房卡的房间段。

    Returns:
        不满足时返回原因；满足或不是跑刀路线时返回 None。
    """
    if route.keycard is None and any(point.kind == KEYCARD_DOOR for point in route.points):
        return '跑刀路线里不能有房卡门点，刷卡只放在开房卡的房间段'
    return None


def check_room_segment(route: BagelRoute, room: BagelKeycardRoom) -> str | None:
    """检查房间段能否接在房间入口之后执行。

    首点必须是途径点、与入口同层且在接缝半径内；段内必须有房卡门点，否则进不了房间。

    Returns:
        不满足时返回原因；满足时返回 None。
    """
    problem = check_route_start(route, room.as_spawn())
    if problem is not None:
        return problem
    if not any(point.kind == KEYCARD_DOOR for point in route.points):
        return '房间段里没有房卡门点'
    return None


def check_entry_segment(route: BagelRoute, room: BagelKeycardRoom) -> str | None:
    """检查入口段的末点能否接上房间入口。

    入口段不允许出现房卡门点：刷卡属于房间段，免得同一扇门刷两次。

    Returns:
        不满足时返回原因；满足时返回 None。
    """
    if not route.points:
        return '入口段没有点'
    if any(point.kind == KEYCARD_DOOR for point in route.points):
        return '入口段里不能有房卡门点，刷卡放在房间段'
    last = route.points[-1]
    if last.layer != room.layer:
        return f'入口段末点分层 {last.layer} 与房间入口分层 {room.layer} 不一致'
    distance = room.distance_to(last.x, last.y)
    if distance > room.radius:
        return f'入口段末点距房间入口 {distance:.0f}px，超出接缝半径 {room.radius:.0f}px'
    return None


def join_keycard_route(entry: BagelRoute, room_route: BagelRoute) -> BagelRoute:
    """把入口段和房间段拼成一条可执行路线。

    拼出的路线只在内存里使用，不写盘。标识与名称由两段组合，日志与统计据此区分组合。
    只有两段都是默认路线时，拼出的路线才算默认来源。

    Args:
        entry: 入口段，提供出生点绑定。
        room_route: 房间段。

    Returns:
        新路线，点位依次为入口段全部点、房间段全部点。入口段末点与房间段首点
        完全相同（同一个入口途径点）时只保留一个。
    """
    room_points = list(room_route.points)
    if entry.points and room_points and entry.points[-1] == room_points[0]:
        room_points = room_points[1:]
    both_default = entry.source == room_route.source == SOURCE_DEFAULT
    return BagelRoute(
        route_id=f'{entry.route_id}+{room_route.route_id}',
        name=f'{entry.name} + {room_route.name}',
        spawn_id=entry.spawn_id,
        source=SOURCE_DEFAULT if both_default else SOURCE_CUSTOM,
        started_at=entry.started_at,
        ended_at=entry.ended_at,
        points=list(entry.points) + room_points,
    )
