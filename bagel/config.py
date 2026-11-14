from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from one_dragon.base.config.config_item import ConfigItem
from one_dragon.base.operation.application.application_config import ApplicationConfig

from .const import APP_ID
from .defs import (
    CLEAN_MODE_CUSTOM,
    CLEAN_MODE_DEFAULT,
    CLEAN_QUALITIES,
    CLEAN_TYPES,
    DEFAULT_CLEAN_FILTER_GROUPS,
    DEFAULT_DIFFICULTY,
    DEFAULT_MODE,
    DIFFICULTIES,
    DIFFICULTY_HARD,
    DIFFICULTY_HAZARD,
    DIFFICULTY_NORMAL,
    KEYCARD_DIFFICULTY,
    KEYCARD_KEEP_TYPES,
    MAP_JANUS,
    MAP_OUTSKIRTS,
    MAPS,
    MODE_DEFAULT_MAPS,
    MODE_KEYCARD,
    MODE_RUN,
    MODES,
)


@dataclass(frozen=True)
class BagelCleanStep:
    """一次出售：类型和品质同时命中才卖。"""

    types: tuple[str, ...]
    qualities: tuple[str, ...]


class BagelConfig(ApplicationConfig):
    """贝果运行上限、入箱和仓库清理设置；按账号、应用组隔离。"""

    def __init__(self, instance_idx: int, group_id: str) -> None:
        """加载当前账号和应用组的配置。"""
        super().__init__(APP_ID, instance_idx, group_id)

    @property
    def mode(self) -> str:
        """运行模式：run 跑刀，keycard 开房卡；默认跑刀，旧配置缺项也按跑刀。"""
        value = self.get('mode', DEFAULT_MODE)
        if value not in MODES:
            return DEFAULT_MODE
        return value

    @mode.setter
    def mode(self, value: str) -> None:
        """保存运行模式。"""
        if value not in MODES:
            raise ValueError(f'mode 只能是 {"、".join(MODES)}')
        self.update('mode', value)

    @property
    def is_keycard(self) -> bool:
        """当前是否为开房卡模式。"""
        return self.mode == MODE_KEYCARD

    @property
    def map_name(self) -> str:
        """进入的地图。缺项或不认识时按模式取默认地图，旧配置的跑刀仍是雅努斯幻境。

        跑刀两张图都能选；开房卡要与所选房间同一张图，由启动检查对照房间清单确认。
        """
        value = self.get('map', None)
        if value not in MAPS:
            return MODE_DEFAULT_MAPS[self.mode]
        return value

    @map_name.setter
    def map_name(self, value: str) -> None:
        """保存地图。"""
        if value not in MAPS:
            raise ValueError(f'map 只能是 {"、".join(MAPS)}')
        self.update('map', value)

    def stored_mode_map_problem(self) -> str | None:
        """检查存下的 ``mode`` 与 ``map`` 原值；缺项按默认，不算问题。

        读取属性时不认识的值会按默认处理，界面据此显示；启动校验看原值，免得拼错的配置静默按默认运行。

        Returns:
            有不认识的值时返回原因，否则返回 None。
        """
        mode = self.get('mode', DEFAULT_MODE)
        if mode not in MODES:
            return f'mode「{mode}」无法识别，只能是 {"、".join(MODES)}'
        map_name = self.get('map', None)
        if map_name is not None and map_name not in MAPS:
            return f'map「{map_name}」无法识别，只能是 {"、".join(MAPS)}'
        return None

    @property
    def keycard_room(self) -> str:
        """开房卡模式的目标房间标识，对应 ``keycard_rooms.yml`` 的 ``id``；未选为空串。"""
        value = self.get('keycard_room', '')
        return value if type(value) is str else ''

    @property
    def keycard_template(self) -> str:
        """目标房间门禁卡的图标模板 id。"""
        return f'keycard_{self.keycard_room}'

    @keycard_room.setter
    def keycard_room(self, value: str) -> None:
        """保存目标房间；是否已登记由启动检查对照房间清单确认。"""
        if type(value) is not str:
            raise ValueError('keycard_room 必须是文本')
        self.update('keycard_room', value)

    @property
    def keycard_reserve(self) -> int:
        """开房卡保留张数：挑卡前清点仓库，本房间的卡不多于这个数就不再挑、结束任务；0 表示用到没卡为止。"""
        value = self.get('keycard_reserve', 0)
        return value if type(value) is int and 0 <= value <= 999 else 0

    @keycard_reserve.setter
    def keycard_reserve(self, value: int) -> None:
        """保存保留张数。"""
        if type(value) is not int or not 0 <= value <= 999:
            raise ValueError('keycard_reserve 必须是 0 至 999 的整数')
        self.update('keycard_reserve', value)

    @property
    def auto_buy_keycard(self) -> bool:
        """开房卡模式：仓库没有可用的门禁卡时是否自动在商店购买房卡。"""
        value = self.get('auto_buy_keycard', False)
        return value if type(value) is bool else False

    @auto_buy_keycard.setter
    def auto_buy_keycard(self, value: bool) -> None:
        """保存是否自动购买房卡。"""
        if type(value) is not bool:
            raise ValueError('auto_buy_keycard 必须是布尔值')
        self.update('auto_buy_keycard', value)

    @property
    def buy_keycard_count(self) -> int:
        """自动购买房卡时的购买数量，上限 30 张；默认 1。"""
        value = self.get('buy_keycard_count', 1)
        return value if type(value) is int and 1 <= value <= 30 else 1

    @buy_keycard_count.setter
    def buy_keycard_count(self, value: int) -> None:
        """保存自动购买房卡数量。"""
        if type(value) is not int or not 1 <= value <= 30:
            raise ValueError('buy_keycard_count 必须是 1 至 30 的整数')
        self.update('buy_keycard_count', value)

    @property
    def run_difficulty(self) -> str:
        """本次实际进入的难度：开房卡固定高危，跑刀按 ``difficulty``。"""
        return KEYCARD_DIFFICULTY if self.is_keycard else self.difficulty

    @property
    def difficulty(self) -> str:
        """跑刀的挑战难度：高危、困难、常规；默认高危。开房卡模式不读这一项。"""
        value = self.get('difficulty', DEFAULT_DIFFICULTY)
        if value not in DIFFICULTIES:
            return DEFAULT_DIFFICULTY
        return value

    @difficulty.setter
    def difficulty(self, value: str) -> None:
        """保存挑战难度。"""
        if value not in DIFFICULTIES:
            raise ValueError(f'difficulty 只能是 {"、".join(DIFFICULTIES)}')
        self.update('difficulty', value)

    @property
    def max_success_rounds(self) -> int:
        """成功入仓多少局后结束；0 不限制成功次数。"""
        return self.get('max_success_rounds', 1)

    @max_success_rounds.setter
    def max_success_rounds(self, value: int) -> None:
        """保存成功局数上限。"""
        if type(value) is not int or not 0 <= value <= 1000:
            raise ValueError('max_success_rounds 必须是 0 至 1000 的整数')
        self.update('max_success_rounds', value)

    @property
    def max_consecutive_failures(self) -> int:
        """全局或同一路线连续失败达到多少局时停止；成功入仓后清零。"""
        return self.get('max_consecutive_failures', 5)

    @max_consecutive_failures.setter
    def max_consecutive_failures(self, value: int) -> None:
        """保存连续失败上限，1 表示首次失败结算后停止。"""
        if type(value) is not int or not 1 <= value <= 10:
            raise ValueError('max_consecutive_failures 必须是 1 至 10 的整数')
        self.update('max_consecutive_failures', value)

    @property
    def blueprint_priority(self) -> bool:
        """蓝图是否优先于全部非蓝图入箱；旧配置缺项也开启。"""
        return self.get('blueprint_priority', True)

    @blueprint_priority.setter
    def blueprint_priority(self, value: bool) -> None:
        """保存蓝图优先开关，不影响仓库出售。"""
        if type(value) is not bool:
            raise ValueError('blueprint_priority 必须是布尔值')
        self.update('blueprint_priority', value)

    @property
    def auto_clean_warehouse(self) -> bool:
        """是否按当前出售方案清理仓库。"""
        return self.get('auto_clean_warehouse', True)

    @auto_clean_warehouse.setter
    def auto_clean_warehouse(self, value: bool) -> None:
        """保存仓库清理开关。"""
        if type(value) is not bool:
            raise ValueError('auto_clean_warehouse 必须是布尔值')
        self.update('auto_clean_warehouse', value)

    @property
    def sell_interval(self) -> int:
        """本任务多少次非空入仓后出售；失败局带回物资也计入。"""
        return self.get('sell_interval', 5)

    @sell_interval.setter
    def sell_interval(self, value: int) -> None:
        """保存出售间隔，至少 1 局。"""
        if type(value) is not int or value < 1 or value > 999:
            raise ValueError('sell_interval 必须是 1 至 999 的整数')
        self.update('sell_interval', value)

    @property
    def clean_mode(self) -> str:
        """default 用两步默认方案；custom 按 clean_steps 顺序多次出售。"""
        return self._text_mode('clean_mode', CLEAN_MODE_DEFAULT)

    @clean_mode.setter
    def clean_mode(self, value: str) -> None:
        """只接受默认或自定义。"""
        if value not in (CLEAN_MODE_DEFAULT, CLEAN_MODE_CUSTOM):
            raise ValueError('clean_mode 只能是 default 或 custom')
        self.update('clean_mode', value)

    @property
    def clean_steps(self) -> list[BagelCleanStep]:
        """自定义时按列表顺序各出售一次。没有配置就是还没有步骤。"""
        raw = self.get('clean_steps', [])
        if type(raw) is not list:
            raise ValueError('clean_steps 必须是列表')
        return [self._parse_step(item) for item in raw]

    @clean_steps.setter
    def clean_steps(self, value: list[BagelCleanStep]) -> None:
        """保存出售步骤。"""
        if type(value) is not list:
            raise ValueError('clean_steps 必须是列表')
        self.update('clean_steps', [self._step_dict(step) for step in value])

    def add_clean_step(self, step: BagelCleanStep | None = None) -> None:
        """在末尾追加一步。"""
        steps = list(self.clean_steps)
        steps.append(step or BagelCleanStep((), ()))
        self.clean_steps = steps

    def update_clean_step(self, index: int, step: BagelCleanStep) -> None:
        """替换指定步骤。"""
        steps = list(self.clean_steps)
        if not 0 <= index < len(steps):
            raise ValueError('清理步骤序号超出范围')
        steps[index] = step
        self.clean_steps = steps

    def delete_clean_step(self, index: int) -> None:
        """删除指定步骤。"""
        steps = list(self.clean_steps)
        if not 0 <= index < len(steps):
            raise ValueError('清理步骤序号超出范围')
        del steps[index]
        self.clean_steps = steps

    def move_clean_step(self, index: int, target: int) -> None:
        """把一步挪到目标位置。"""
        steps = list(self.clean_steps)
        if not 0 <= index < len(steps) or not 0 <= target < len(steps):
            raise ValueError('清理步骤序号超出范围')
        step = steps.pop(index)
        steps.insert(target, step)
        self.clean_steps = steps

    def clean_filter_groups(self) -> tuple[tuple[str, ...], ...]:
        """每步快速选择要点中的区域名，默认与自定义都保留步骤边界。

        开房卡模式去掉 ``KEYCARD_KEEP_TYPES`` 里的类型；去掉后一步不剩任何类型时整步跳过，
        不能只留品质，否则会卖掉该品质的全部类型。
        """
        if self.clean_mode != CLEAN_MODE_CUSTOM:
            return DEFAULT_CLEAN_FILTER_GROUPS
        steps = self.clean_steps
        if self.is_keycard:
            steps = [
                BagelCleanStep(tuple(name for name in step.types if name not in KEYCARD_KEEP_TYPES), step.qualities)
                for step in steps
                if not step.types or any(name not in KEYCARD_KEEP_TYPES for name in step.types)
            ]
        return tuple(
            tuple(f'筛选-{name}' for name in (*step.types, *step.qualities))
            for step in steps
        )

    def keycard_skipped_steps(self) -> list[int]:
        """开房卡模式下因只含门禁卡而整步跳过的自定义步骤序号（从 1 起）。"""
        if not self.is_keycard or self.clean_mode != CLEAN_MODE_CUSTOM:
            return []
        return [
            index for index, step in enumerate(self.clean_steps, start=1)
            if step.types and all(name in KEYCARD_KEEP_TYPES for name in step.types)
        ]

    def _parse_step(self, value: object) -> BagelCleanStep:
        """读取一步；名单外或重复的名字留给启动校验拒绝。"""
        if type(value) is not dict:
            raise ValueError('清理步骤必须是对象')
        return BagelCleanStep(tuple(self._stored_names(value.get('types'))), tuple(self._stored_names(value.get('qualities'))))

    def _step_dict(self, step: BagelCleanStep) -> dict[str, list[str]]:
        """写入前校验类型和品质。"""
        types = list(step.types)
        qualities = list(step.qualities)
        self._set_names_checked(types, CLEAN_TYPES, '清理类型')
        self._set_names_checked(qualities, CLEAN_QUALITIES, '清理品质')
        return {'types': types, 'qualities': qualities}

    def _stored_names(self, value: object) -> list[str]:
        """读取一步里的名字列表。"""
        if type(value) is not list or any(type(item) is not str for item in value):
            raise ValueError('清理步骤的类型和品质必须是字符串列表')
        return list(value)

    def _text_mode(self, name: str, default: str) -> str:
        """读取文本配置；手改成非文本时交给启动校验拒绝。"""
        value = self.get(name, default)
        if type(value) is not str:
            raise ValueError(f'{name} 必须是文本')
        return value

    def _set_names_checked(self, value: list[str], allowed: tuple[str, ...], label: str) -> None:
        """拒绝非列表、重复和名单外的名字。"""
        if any(item not in allowed for item in value):
            raise ValueError(f'{label}只能从规定名单里选择')
        if len(set(value)) != len(value):
            raise ValueError(f'{label}不能重复')

    def validate(self) -> None:
        """启动前校验配置。房间是否已登记、房间与地图是否一致、是否开启了路线由启动检查对照路线配置确认。"""
        problem = self.stored_mode_map_problem()
        if problem is not None:
            raise ValueError(f'{problem}；打开设置页会改回默认值，确认后再运行')
        if self.is_keycard and not self.keycard_room:
            raise ValueError('开房卡模式还没有选择房间')
        reserve = self.get('keycard_reserve', 0)
        if type(reserve) is not int or not 0 <= reserve <= 999:
            raise ValueError('keycard_reserve 必须是 0 至 999 的整数')
        if type(self.auto_buy_keycard) is not bool:
            raise ValueError('auto_buy_keycard 必须是布尔值')
        buy_count = self.get('buy_keycard_count', 1)
        if type(buy_count) is not int or not 1 <= buy_count <= 30:
            raise ValueError('buy_keycard_count 必须是 1 至 30 的整数')
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f'difficulty 只能是 {"、".join(DIFFICULTIES)}')
        interval = self.sell_interval
        if type(interval) is not int or not 1 <= interval <= 999:
            raise ValueError('sell_interval 必须是 1 至 999 的整数')
        value = self.max_success_rounds
        if type(value) is not int or not 0 <= value <= 1000:
            raise ValueError('max_success_rounds 必须是 0 至 1000 的整数')
        failures = self.max_consecutive_failures
        if type(failures) is not int or not 1 <= failures <= 10:
            raise ValueError('max_consecutive_failures 必须是 1 至 10 的整数')
        if type(self.blueprint_priority) is not bool:
            raise ValueError('blueprint_priority 必须是布尔值')
        if type(self.auto_clean_warehouse) is not bool:
            raise ValueError('auto_clean_warehouse 必须是布尔值')
        if self.clean_mode not in (CLEAN_MODE_DEFAULT, CLEAN_MODE_CUSTOM):
            raise ValueError('clean_mode 只能是 default 或 custom')
        steps = self.clean_steps
        for step in steps:
            if any(name not in CLEAN_TYPES for name in step.types):
                raise ValueError('清理类型不在快速选择里')
            if any(name not in CLEAN_QUALITIES for name in step.qualities):
                raise ValueError('清理品质不在快速选择里')
            if len(set(step.types)) != len(step.types) or len(set(step.qualities)) != len(step.qualities):
                raise ValueError('清理筛选不能重复')
        if (
            self.auto_clean_warehouse
            and self.clean_mode == CLEAN_MODE_CUSTOM
            and (not steps or any(not step.types or not step.qualities for step in steps))
        ):
            raise ValueError('自定义清理每一步都要同时选择类型和品质')


class BagelMode(Enum):
    """运行模式。"""

    RUN = ConfigItem('跑刀', MODE_RUN, '零携带、零投资入场')
    KEYCARD = ConfigItem('开房卡', MODE_KEYCARD, '安全箱带 1 张门禁卡')


class BagelMap(Enum):
    """地图。"""

    JANUS = ConfigItem('雅努斯幻境', MAP_JANUS)
    OUTSKIRTS = ConfigItem('城郊幻境', MAP_OUTSKIRTS)


class BagelDifficulty(Enum):
    """挑战难度。"""

    HAZARD = ConfigItem('高危', DIFFICULTY_HAZARD)
    HARD = ConfigItem('困难', DIFFICULTY_HARD)
    NORMAL = ConfigItem('常规', DIFFICULTY_NORMAL)


class BagelCleanMode(Enum):
    """清理方案。默认分两步保留金色蓝图，自定义按已保存的步骤出售。"""

    DEFAULT = ConfigItem('默认', CLEAN_MODE_DEFAULT, '贵重物品、战术棱镜卖 C/B/A/S；其他只卖 C/B/A')
    CUSTOM = ConfigItem('自定义', CLEAN_MODE_CUSTOM, '按勾选的类型和品质出售')


class BagelCleanType(Enum):
    """仓库快速选择里可卖的类型。不含「全部」。"""

    VALUABLE = ConfigItem('贵重物品', '贵重物品')
    PRISM = ConfigItem('战术棱镜', '战术棱镜')
    OTHER = ConfigItem('其他', '其他')
    EQUIP = ConfigItem('装备', '装备')
    TACTIC = ConfigItem('战术道具', '战术道具')
    KEYCARD = ConfigItem('门禁卡', '门禁卡')


class BagelCleanQuality(Enum):
    """仓库快速选择里可卖的品质。"""

    C = ConfigItem('C', 'C')
    B = ConfigItem('B', 'B')
    A = ConfigItem('A', 'A')
    S = ConfigItem('S', 'S')
    Z = ConfigItem('Z', 'Z')
