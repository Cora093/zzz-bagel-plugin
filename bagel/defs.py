"""贝果插件的路径基准与业务常量。

插件元数据（``APP_ID``、``APP_NAME``、``PLUGIN_*``）在 ``const``，
那份文件由 ``scripts/generate_const.py`` 从 ``plugin.json`` 生成，不要手改。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .const import APP_ID

if TYPE_CHECKING:
    from zzz_od.context.zzz_context import ZContext

PLUGIN_DIR: Path = Path(__file__).resolve().parent
"""插件根目录；模板、出生点、地图底图等自带资源都从这里取，不依赖工作目录。"""


def get_plugin_data_dir(ctx: ZContext) -> Path:
    """插件运行期数据目录，存放自定义路线、开启选择、路线记录与现场截图。

    放在插件目录之外，插件更新替换整个目录时用户数据不会被删掉。
    路径由主程序按 ``APP_ID`` 分配（当前是 ``plugin_data/<APP_ID>``），插件不自己拼，
    主程序改了目录规则，插件跟着走。

    Args:
        ctx: 运行上下文，用它提供的接口取目录。

    Returns:
        主程序分配的插件数据目录。
    """
    return ctx.get_plugin_data_dir(APP_ID)

DIFFICULTY_NORMAL: str = '常规'
DIFFICULTY_HARD: str = '困难'
DIFFICULTY_HAZARD: str = '高危'
DIFFICULTIES: tuple[str, ...] = (DIFFICULTY_NORMAL, DIFFICULTY_HARD, DIFFICULTY_HAZARD)
DEFAULT_DIFFICULTY: str = DIFFICULTY_HAZARD

# 各难度推荐装备价值：常规 12000，困难 72100，高危 258000
DIFFICULTY_RECOMMENDED_VALUES: dict[str, str] = {
    DIFFICULTY_NORMAL: '12000',
    DIFFICULTY_HARD: '72100',
    DIFFICULTY_HAZARD: '258000',
}

RECOMMENDED_VALUE: str = DIFFICULTY_RECOMMENDED_VALUES[DIFFICULTY_HAZARD]

# 电子保险箱完成四轮点按后等待搜索结果，点按时机按光圈状态判断。
SAFE_UNLOCK_HITS: int = 4

# 仓库清理：default 按下面的两步出售；custom 按 clean_steps 顺序出售。
CLEAN_MODE_DEFAULT: str = 'default'
CLEAN_MODE_CUSTOM: str = 'custom'
CLEAN_TYPES: tuple[str, ...] = ('贵重物品', '战术棱镜', '其他', '装备', '战术道具', '门禁卡')
CLEAN_QUALITIES: tuple[str, ...] = ('C', 'B', 'A', 'S', 'Z')
# 每步分别保存类型和品质，不能合并成一组，否则会卖掉 S 品质其他物品。
DEFAULT_CLEAN_STEPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (('贵重物品', '战术棱镜'), ('C', 'B', 'A', 'S')),
    (('其他',), ('C', 'B', 'A')),
)
DEFAULT_CLEAN_FILTER_GROUPS: tuple[tuple[str, ...], ...] = tuple(
    tuple(f'筛选-{name}' for name in (*types, *qualities))
    for types, qualities in DEFAULT_CLEAN_STEPS
)

# 运行模式：跑刀零携带进雅努斯幻境；开房卡带 1 张门禁卡进城郊幻境高危，只搜一个房间。
MODE_RUN: str = 'run'
MODE_KEYCARD: str = 'keycard'
MODES: tuple[str, ...] = (MODE_RUN, MODE_KEYCARD)
DEFAULT_MODE: str = MODE_RUN

MAP_JANUS: str = '雅努斯幻境'
MAP_OUTSKIRTS: str = '城郊幻境'
MAPS: tuple[str, ...] = (MAP_JANUS, MAP_OUTSKIRTS)
#: 开房卡模式固定的难度。
KEYCARD_DIFFICULTY: str = DIFFICULTY_HAZARD
#: 每种模式当前支持的地图，第一项是默认；其他组合拒绝启动。
MODE_MAPS: dict[str, tuple[str, ...]] = {MODE_RUN: (MAP_JANUS,), MODE_KEYCARD: (MAP_OUTSKIRTS,)}
#: 每种模式当前可选的难度。
MODE_DIFFICULTIES: dict[str, tuple[str, ...]] = {MODE_RUN: DIFFICULTIES, MODE_KEYCARD: (KEYCARD_DIFFICULTY,)}
#: 城郊幻境高危的推荐装备价值。取自用户录屏的城郊高危备战页（推荐价值：258000），选图页读数尚未实机核对；
#: 改成 None 时入场按「点一次高危后接受读数」处理。
OUTSKIRTS_RECOMMENDED_VALUE: str | None = '258000'
#: 开房卡模式出售时一律跳过的类型，防止卖掉要用的卡。
KEYCARD_KEEP_TYPES: tuple[str, ...] = ('门禁卡',)
