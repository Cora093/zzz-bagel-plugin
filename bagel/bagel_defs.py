"""贝果插件的路径基准与业务常量。

插件元数据（``APP_ID``、``APP_NAME``、``PLUGIN_*``）在 ``bagel_const``，
那份文件由 ``scripts/generate_const.py`` 从 ``plugin.json`` 生成，不要手改。
"""

from pathlib import Path

from one_dragon.utils import os_utils

from .bagel_const import APP_ID

PLUGIN_DIR: Path = Path(__file__).resolve().parent
"""插件根目录；模板、出生点、地图底图等自带资源都从这里取，不依赖工作目录。"""

def get_plugin_data_dir() -> Path:
    """插件运行期数据目录，存放自定义路线、开启选择、路线记录与现场截图。

    放在插件目录之外，插件更新替换整个目录时用户数据不会被删掉。
    每次调用按当前工作目录现算，不缓存。

    Returns:
        工作目录下的 ``plugin_data/<app_id>``。
    """
    return Path(os_utils.get_work_dir()).joinpath('plugin_data', APP_ID)

RECOMMENDED_VALUE: str = '258000'

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
