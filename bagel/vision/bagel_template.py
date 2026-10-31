"""插件自带模板的读取与区域匹配。

主程序的 TemplateLoader 只读工作目录 ``assets/template``，读不到插件目录里的模板。
screen 区域即使配置了 ``template_sub_dir: bagel``，用 ``round_by_find_area`` 也只会报
「未加载模板」并判为没找到。插件模板一律经这里读取和匹配。
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from one_dragon.utils import cv2_utils
from one_dragon.utils.log_utils import log

from ..bagel_defs import PLUGIN_DIR

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from zzz_od.context.zzz_context import ZContext


@cache
def load_plugin_template(sub_dir: str, template_id: str) -> tuple[MatLike, MatLike | None] | None:
    """读插件自带模板的原图与掩码；缺图时返回 None。"""
    directory = PLUGIN_DIR / 'assets' / 'template' / sub_dir / template_id
    raw = cv2_utils.read_image(str(directory / 'raw.png'))
    if raw is None:
        log.warning(f'缺少插件模板：{directory}')
        return None
    return raw, cv2_utils.read_image(str(directory / 'mask.png'))


def find_plugin_template_area(ctx: ZContext, screen: MatLike, screen_name: str, area_name: str) -> bool:
    """按 screen 区域的范围、模板和阈值，用插件模板在截图中匹配。"""
    area = ctx.screen_loader.get_area(screen_name, area_name)
    if area is None or not area.is_template_area:
        return False
    template = load_plugin_template(area.template_sub_dir, area.template_id)
    if template is None:
        return False
    crop = cv2_utils.crop_image_only(screen, area.rect)
    if crop is None or crop.size == 0:
        return False
    raw, mask = template
    result = cv2_utils.match_template(
        crop, raw, area.template_match_threshold, mask=mask, only_best=True, ignore_inf=True,
    )
    return result.max is not None
