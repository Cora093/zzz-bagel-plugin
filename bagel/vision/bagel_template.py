"""读插件自带的模板图并做匹配，不经过主仓的全局模板库。

模板放在 ``assets/template/bagel/<模板 id>/``，跟着插件目录走：
插件装在两层目录里、换一台机器，都照样读得到。
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from cv2.typing import MatLike
from one_dragon.base.geometry.rectangle import Rect
from one_dragon.utils import cv2_utils
from one_dragon.utils.log_utils import log

from ..bagel_defs import PLUGIN_DIR

TEMPLATE_ROOT: Path = PLUGIN_DIR / 'assets' / 'template' / 'bagel'
"""插件自带模板根目录，按模板 id 分文件夹。"""


@cache
def load_template(template_id: str) -> tuple[MatLike, MatLike | None] | None:
    """读模板原图与掩码，同一个模板只读一次。

    Args:
        template_id: 模板 id，对应模板根目录下的子文件夹。

    Returns:
        (原图, 掩码)；缺原图时返回 None，掩码缺失时第二项为 None。
    """
    directory = TEMPLATE_ROOT / template_id
    raw = cv2_utils.read_image(str(directory / 'raw.png'))
    if raw is None:
        log.warning('缺少插件模板：%s', directory)
        return None
    return raw, cv2_utils.read_image(str(directory / 'mask.png'))


def has_template(
    screen: MatLike, pc_rect: Rect, template_id: str, threshold: float,
) -> bool:
    """在截图的指定区域内找插件自带模板。

    Args:
        screen: 整屏截图，RGB。
        pc_rect: 匹配区域。
        template_id: 模板 id。
        threshold: 相关系数阈值。

    Returns:
        区域里存在达到阈值的匹配时为 True。
    """
    loaded = load_template(template_id)
    if loaded is None or screen is None:
        return False
    raw, mask = loaded
    crop = cv2_utils.crop_image_only(screen, pc_rect)
    if crop is None or crop.size == 0:
        return False
    result = cv2_utils.match_template(crop, raw, threshold, mask=mask)
    return result.max is not None
