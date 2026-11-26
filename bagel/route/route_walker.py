"""贝果路线执行状态机：按打点顺序走到最后。

入场后先识别出生点：连续 2 次新截图命中同一个已登记出生点才确认，最多
观察 10 秒。正式运行执行该出生点当前开启的路线；试跑执行指定路线，但角色
也必须站在该路线绑定的出生点。确认后从路线第 1 点开始，游标顺序推进：

- 下一格是途径点：正常导航，进入 3px 到达半径即算路过。
- 下一格是箱类：按距离慢步或碎步靠近。当前目标是电子保险箱时，进入
  15px 后保持 0.10 秒短步；可信提示出现后松键，按类型交互。
- 路线走完：原地判断现场有无交互物，没有就结束；由应用层走既有退出结算链。

出生点不在已登记范围内、或出生点没有开启路线时，返回 ``STATUS_ANCHOR_MISS``，
由应用层退出重开。

导航与交互复用旧实现里与旧流程解耦的部分：分层定位、朝向识别、转向
补偿、光圈解锁、逐件入箱、关闭面板。旧流程（flow.yml 步骤、写死的
动作模板）不再参与。
"""

from __future__ import annotations

import re
import time
from math import atan2, cos, degrees, hypot, radians, sin
from typing import TYPE_CHECKING

import numpy as np
from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.base.operation.operation_round_result import OperationRoundResultEnum
from one_dragon.utils import cal_utils, cv2_utils, str_utils
from one_dragon.utils.i18_utils import gt
from one_dragon.utils.log_utils import log

from ..defs import INTERACT_TAP_SECONDS, MAP_JANUS
from ..diag import diag
from ..operation.close_search import BagelCloseSearch
from ..operation.container import INTERACT_KEY_THRESHOLD
from ..operation.operation import BagelOperation
from ..operation.organize_backpack import BagelOrganizeBackpack
from ..operation.store import BagelStoreSafe
from ..operation.unlock_safe import BagelUnlockSafe
from ..screenshot import save_screenshot
from ..stats import (
    BagelSafeLedger,
    format_value,
    parse_value,
)
from ..vision.map_locator import (
    MapLocator,
    MultiMapLocator,
    load_map_locators,
)
from ..vision.minimap_angle import analyze
from ..vision.prompt_list import PromptFrame, PromptSelection
from ..vision.slots import slot_occupied
from ..vision.template import load_template
from .capture_stats import CaptureStats, movement_stage
from .input_worker import (
    STEP_REST_SECONDS,
    TURN_DEADBAND,
    TURN_DEADBAND_FAST,
    BagelInputWorker,
    HeadingSample,
    TurnRatioMemory,
)
from .round_perf import RepeatMerger, RoundPerf
from .route_model import (
    BEACON,
    BOX_CATEGORIES,
    CONTAINER_KINDS,
    ELEVATOR,
    ELEVATOR_EXIT_BACKWARD,
    KEYCARD_DOOR,
    KIND_LABELS,
    BagelRoute,
    BagelRoutePoint,
    BagelRouteRef,
    BagelSpawnPoint,
    check_route_start,
    map_id_of,
)
from .route_recorder import is_foreground
from .route_store import (
    BagelRouteConfigError,
    BagelRouteStore,
)
from .spawn import (
    SPAWN_AMBIGUOUS,
    SPAWN_HIT,
    SPAWN_MISS,
    BagelSpawnMatch,
    match_spawn,
)

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from one_dragon.base.geometry.point import Point
    from one_dragon.base.matcher.ocr.ocr_match_result import OcrMatchResult
    from one_dragon.base.operation.operation_base import OperationResult
    from one_dragon.base.operation.operation_round_result import OperationRoundResult
    from zzz_od.context.zzz_context import ZContext

ELEVATOR_OPTION_TAP_SECONDS: float = 1.0
"""电梯选项还没出现时，点一次屏幕中间推进对话的间隔（用户给定每秒一次）。"""

ELEVATOR_ENTRY_TIMEOUT: float = 10.0
"""从电梯门口往前走多久还没进电梯就判失败（秒）。

只限制「门口走向电梯」这一段，此时还在场景里，失败后能正常退出。
进电梯之后的三段一律不设超时：卡在里面连游戏都退不出来。
"""

ELEVATOR_PHASE_ENTER: str = 'enter'
ELEVATOR_PHASE_WAIT_OPTION: str = 'wait_option'
ELEVATOR_PHASE_CLICK_OPTION: str = 'click_option'
ELEVATOR_PHASE_EXIT: str = 'exit'
"""电梯四个阶段的标识：进门、等选项、点选项、出电梯。"""

SPAWN_CONFIRM_FRAMES: int = 2
"""出生点识别需要连续几次新截图得出相同结论。"""

SPAWN_OBSERVE_SECONDS: float = 10.0
"""出生点识别的最长观察时间（秒）；期间不移动角色。"""

ARRIVE_RADIUS: float = 3.0
"""到达半径：3px 内算到点（用户给定）。"""

INTERACTION_TRUST_DISTANCE: float = 40.0
"""交互采信距离：提示出现时距目标点超过此值不采信。"""

SPRINT_DISTANCE: float = 30.0
"""多少像素外开冲刺（用户给定 30px）。

30px 外就冲刺，边跑边点闪避；30px 内不冲刺，松键改用走路逼近。
"""

SPRINT_DODGE_INTERVAL: float = 0.3
"""冲刺期间两次点闪避的最短间隔（秒，用户给定）。

每轮都点时间隔约 37ms，游戏会当成按住闪避，触发星见雅等角色的长闪。
"""

SPRINT_START_DELAY: float = 0.3
"""W 实际按下后多久才点第一下闪避（秒，用户给定）。

大幅转向后角色朝向还停在转向前，立刻点闪避会沿旧朝向冲出约 0.4 秒；
实测约 90 度转向时先走 0.1 秒就已转过身；接近 180 度时转身慢的角色要 0.25~0.3 秒，
实机 0.2 秒仍偏短，取 0.3 秒。
"""

BEACON_SLOW_RADIUS: float = 10.0
"""途径点慢步半径：进入它改用慢步走走停停，不再一路按住前进（用户给定）。"""

NEAR_STEP_TURN_RADIUS: float = 8.0
"""距离目标点小于它就不再按角度停车转。

这个距离上目标方位完全由 1~2px 的定位抖动决定：实测 5px 处相邻两轮算出的偏差
是 78° 和 59°，按它停车转只会原地打摆。改成像束角内那样边走边转，转向幅度受
CRUISE_TURN_CAP 限制。
"""

NEAR_STOP_TURN_ANGLE: float = 60.0
"""近点内朝向偏差超过它时仍停车转到位，再慢步。

近点不停车转，偏差大时一步慢步会朝错误方向走出好几像素：实机里角色面朝东、
途径点在西边 7px，一步慢步往东走了 8px；折回后面朝西到达途径点，下一个公文包点
在正南 6px，一步慢步又往西走了 6px，余势再滑出 5px，出了门卡在墙边。
改用 0.08 秒短步也不行：短按几乎走不动，边走边转每轮又限幅 15 度，实机在 4.5px
处原地蹭了 4 秒，方向不对时还越蹭越远。

阈值取 60 度：5px 处定位抖动造成的偏差实测 59–78 度，转到位后残差落在阈值内，
不会因抖动反复停车。
"""

BOX_SMALL_STEP_RADIUS: float = 3.0
"""箱子碎步半径：进到它以内用碎步（用户给定 3px）。"""

ELECTRONIC_SMALL_STEP_RADIUS: float = 15.0
"""电子保险箱短步半径：进入后保持短步，直到当前目标结束。

只用于雅努斯幻境的电子保险箱：那一个箱子要碎步才贴得准。其他地图的电子
保险箱和普通箱子一样慢步接近、3px 内碎步。
"""

BOX_WAIT_RADIUS: float = 1.9
"""箱子原地等待半径：距离小于 1.9px 就停步原地等提示（用户给定）。

用户要求「0px 才停」，但坐标不一定正好落在目标点上：
编辑器拖点会保留 1 位小数，定位又是整数像素，实际常停在 0.5px。
按 <= 0 判定永远进不去原地等待，只能一路碎步撞到 75 秒兜底超时。
所以用「小于 1.9」作判据，0.5px 这种就能正确进入等待。
"""

SMALL_STEP_TIMEOUT: float = 5.0
"""碎步超时（秒）：3px 内碎步前进超过它还没贴到目标就判超时（用户给定）。

碎步是不看角度硬迈的，方向偏了会一直贴着原地蹭。给一个短时限，
免得它在这个距离上耗到 75 秒的单点兜底。
"""

SLOW_STEP_PRESS: float = 0.5
"""慢步单次按压时长（秒）：比碎步长，一次走得多一点（用户给定 0.5）。"""

SLOW_STEP_WAIT: float = 0.1
"""慢步投递后重新观察的间隔（秒）；短按未结束时不叠发下一步。"""

ANGLE_FAIL_SECONDS: float = 10.0
"""连续读不到镜头朝向超过这个时长才判失败（用户给定 10 秒）。

单帧读不到是常事（转视角、贴墙、扇形被挡），不能立刻中断整条路线。
"""

WALK_TURN_ANGLE: float = 10.0
"""停车转向退出阈值（度）：停车等转向时偏差降回它以内即恢复前进。"""

WALK_STOP_TURN_ANGLE: float = 16.0
"""行进中触发停车转向的偏差阈值（度）：与 ``WALK_TURN_ANGLE`` 组成滞回带。

行进中偏差在 16 度以内继续边走边修，超过 16 度才松键停车，停车后转回 10 度内再走，
避免 10 度边缘因朝向读数 ±2 度抖动反复松键停顿和打断冲刺。

行进中要连续两张不同截图都超过 16 度才停车：朝向识别偶有单帧跳变（扇区被图标、
暖色背景短暂污染），单帧就停会白白松键一次。真实大偏差只晚一张截图停车。"""

SMALL_STEP_PRESS: float = 0.10
"""碎步单次按压时长（秒）。两次短按之间的停顿由 ``STEP_REST_SECONDS`` 管。

实测按 0.05、0.08 秒每步都约 0.4px，0.12 秒约 0.8px；0.10 秒落在两档之间，
尚未单独标定，按线性推算约 0.6px。"""

STEP_REST_POLL: float = 0.03
"""停顿期间主线的轮询上限（秒）：停顿结束后最多晚这么久拍到决策帧。"""

HUD_CHECK_INTERVAL: float = 0.5
"""定位成功时，暂停菜单检查的最小间隔（秒）；定位失败那一轮总是检查。"""

LOCATE_FAIL_SECONDS: float = 10.0
"""连续定位失败超过这个时长才判失败（用户给定 10 秒）。

给 MultiMapLocator 的最后一档降级留足时间：它要丢失满 1 秒
才开始搜索全部层，全层搜索本身也慢。
"""

STUCK_SECONDS: float = 3.0
STUCK_DISTANCE: float = 2.0
"""按住前进时判卡位：窗口内位移不足此值即视为走不动。"""

SIDE_STEP_SECONDS: float = 0.2
"""卡位时侧移一小步的按键时长（秒）。短按一下，不挪过头。"""

STUCK_STEP_LIMIT: int = 5
"""同一个点连续侧移的上限；挪这么多步还脱不了身，说明不是偏线而是真走不通。"""
STUCK_ON_LINE_SEQUENCE: tuple[int, ...] = (-1, -1, 1, 1, 1)
"""卡位时已站在路径线上、判不出往哪侧挪，就按这个顺序侧移：-1 左（A），1 右（D）。

站在线上仍走不动，说明挡路的是线上的东西（如刚开过的电子保险箱的箱角），
等着不动只会把侧移次数耗光。先左两步，再右三步，回到原位后再向右一步，
两侧都试到。长度与 ``STUCK_STEP_LIMIT`` 相同。
"""

BLOCKED_DOOR_TAP_LIMIT: int = 2
"""同一个点受阻时为开门点按 F 的最多次数。

路上的普通门有时开着、有时关着，录路线时开着的门不会打成门点。关着时角色顶在门板上，
提示列表中有门（如「木质单开门」）。只在受阻时才按，门开着时不会去碰它。
"""

BLOCKED_DOOR_WAIT_SECONDS: float = 0.8
"""受阻时点按 F 开门后，等门打开的时间（秒）。"""

SLOW_STUCK_SECONDS: float = 5.0
"""慢步或短步逼近时，位置连续这么久变化不足 ``STUCK_DISTANCE`` 就算受阻。"""

RETREAT_LIMIT: int = 2
"""同一个点受阻后最多退回上一个点重走几次；用完再受阻就按失败处理。"""

RETREAT_ARRIVE_RADIUS: float = 1.9
"""退回上一个点时的到点判据，比途径点的 3px 严。

受阻多半是提前 3px 判到点、还没绕过拐角就转向了；退回时要站到录制位置上，
重走才有意义。定位是整数像素，小于 1.9 即落在录制点或其上下左右、斜对角一格内。
"""

RETREAT_SECONDS: float = 10.0
"""退回上一个点的时限；到时还没站上就从当前位置重走。"""

POINT_TIMEOUT: dict[str, float] = {BEACON: 45.0, 'default': 75.0}
"""单点超时（秒）：途径点 45，箱类 75。"""

ACTION_LIMIT: int = 100
"""单个点允许的输入次数上限。"""

BOX_PROMPT_SECONDS: float = 1.0
"""箱子点贴到 0px 后原地等提示的上限（用户给定 1 秒）。

箱子一定会刷出交互提示，所以到 0px 还读不到，只可能是这个点本身
没东西，直接跳到下一个点，不判失败、也不原地死等。
"""

SURROUND_CHECK_SECONDS: float = 0.5
"""走完路线后原地观察交互物的时长。

游标推进已经等到局内提示恢复，末尾再看两三帧即可；原先 2 秒在实机里只看到已搜过的绿勾。
"""

MECHANICAL_HOLD_SECONDS: float = 5.0
"""机械保险箱长按 F 的最长时间（秒）。"""

MECHANICAL_PROMPT_LOST_SECONDS: float = 0.5
"""机械保险箱长按中 F 键连续消失多久（秒），就松手回碎步移动重新靠近。

正常长按时「长按 F 机械保险箱」条目和 F 键一直在，面板一出才消失。
角色带着余势滑到采信距离边缘时就开始长按，可能接着滑出交互范围，
条目消失后长按不会生效，干等只会两次超时判失败。
"""

MECHANICAL_PROMPT_LOST_LIMIT: int = 2
"""同一个机械保险箱点因长按中提示消失而重新靠近的最多次数，用完后按原超时逻辑处理。"""


PRESS_F_RESULT_SECONDS: float = 0.4
"""点按 F 后这段时间内只轮询结果、不重按（秒）。

实机录屏里普通容器面板在点按后 0.2 秒内出现；面板开着时再按 F 是「全部拾取」，所以不能提前重按。
"""

PRESS_F_POLL_SECONDS: float = 0.1
"""点按 F 后等面板或门提示消失的轮询间隔（秒）。"""

PRESS_F_MAX_TAPS: int = 20
"""同一个点最多点按 F 的次数（每次约 0.4 秒）。点这么多次提示还在、面板也没出，
多半是类型判错（点按对机械保险箱、刷卡处无效），按本局失败处理，不空转到看门狗介入。"""

CARD_HOLD_SECONDS: float = 4.0
"""刷卡处单次长按 F 的上限（秒）。录像里进度圈约 1.5 秒填满。"""

CARD_HOLD_MIN_SECONDS: float = 1.5
"""长按满这么久后，F 键消失才算刷完；更早消失多半是读帧抖动。

录像里进度圈约 1.5 秒填满。实机出现过长按 1.1 秒时 F 键读不到、提前松手、门没开。
"""

CARD_STABLE_SECONDS: float = 0.5
"""刷卡条目与 F 键连续对准这么久（秒）才开始长按。

碎步刚到点或试探侧移后角色还带着余势，提示一出现就长按，角色可能在长按中滑出
刷卡器范围，长按不再生效。先原地等余势停下：仍在范围内就提示不变，滑出去了就
提示消失、转去试探。0.5 秒是推断值，尚未标定。
"""

CARD_HOLD_LOST_LIMIT: int = 2
"""长按中刷卡条目整条消失、松手回去试探的最多次数，用完后按本局失败处理。

正常刷完时条目一直在，计数先变成 0，松手后提示才消失。长按开始时角色还带着
余势滑动，可能滑出交互范围，条目和 F 键一起消失；这时只看「满 1.5 秒 F 键消失」
会把没刷上当成刷完，后面顶着关着的门走。
"""

CARD_VERIFY_WAIT_SECONDS: float = 0.5
"""判定刷完并松手后，等这么久再看第一帧，确认 F 键没有重新对准刷卡条目（秒）。"""

CARD_SWIPE_SETTLE_SECONDS: float = 2.0
"""松手后 F 键仍对准刷卡条目时，最多等多久让提示消失，才判没刷上（秒，从松手起算）。

实机刷开门时都是长按约 1.6 秒读到计数「0」就松手，松手约 0.56 秒后提示已消失。
刷完的瞬间提示还显示「0/1」，若只看松手 0.5 秒后的一帧就判没刷上，会多按一次。
所以给提示留时间消失；等满仍在才重按，重按前读到计数 0 的处理见 ``CARD_SPENT_WAIT_SECONDS``。
"""

CARD_SPENT_WAIT_SECONDS: float = 4.0
"""本点长按过后读到计数 0 时，从松手起最多等多久让提示消失（秒）。

长按过后计数 0 有两种可能：卡刚被这次刷掉（门已开、提示稍后消失），或者本来就没带卡。
两种情况再按都没用，所以不再重按，只等提示消失：消失就判刷开；满时仍在，本点读到过
非 0 计数就判本局失败（带了卡，门的状态不明），从没读到过就按没带卡停止任务。
4 秒是推断值：实机观察到松手约 0.56 秒后提示就消失，尚未标定。
"""

CARD_PROMPT_LOST_SECONDS: float = 0.5
"""还没长按时刷卡条目连续消失多久（秒），就回碎步移动左右试探。

碎步刚到点时角色带着余势，提示出现后可能又滑出刷卡器范围。只在交互节点
原地等不会把角色带回去，所以提示稳定消失后改回移动节点试探。留 0.5 秒是为了
不把单帧闪烁当成消失。
"""

CARD_KEY_ROW_TOLERANCE: int = 12
"""F 键与刷卡条目算同一行的纵向容差（像素）。实测对齐时相差 2 像素。"""

CARD_PROBE_SEQUENCE: tuple[int, ...] = (-1, -1, 1, 1, 1, 1)
"""房卡门到点没有刷卡提示时的左右试探顺序：-1 向左（A），1 向右（D）。

小地图只有整数像素精度，坐标对上时角色仍可能站在门板正中，离刷卡器差一个身位，
提示不出现。实机截图里镜头跟在身后对着门，刷卡器在左侧，所以先向左两步，
再向右走回原位并继续向右两步。
"""

CARD_PROBE_STEP_SECONDS: float = 0.2
"""试探时每次侧移的按键时长（秒）。"""

CARD_COUNT_PATTERN: re.Pattern[str] = re.compile(r'(\d+)\s*/\s*(\d+)')
"""刷卡条目第二行的持卡计数，如「1/1」；首位为 0 表示没带对应门禁卡。"""

FAIL_SURVEY_SECONDS: float = 3.0
"""装箱或关闭面板失败后，等 HUD 调查收益读数停稳的最长时间（秒）。

战斗中普通攻击按键可能被动画遮住，涨幅浮字也可能干扰读数，所以要多等几帧。
"""

FAIL_SURVEY_WAIT: float = 0.3
"""失败后补读调查收益时，两次读数的间隔（秒）。"""

SURVEY_LINE_TOLERANCE: int = 6
"""读调查收益时，同一行文字中心的纵向容差（像素）。

物品入箱后，HUD 数字上方可能浮出该件涨幅（如「+10K」），会落进读数区域。
"""

PICKUP_TAP_LIMIT: int = 3
"""同一条拾取物最多点按几次 F。还在说明捡不起来，本局之后不再按它。"""

PICKUP_FAIL_LIMIT: int = 3
"""本局累计这么多条拾取物捡不起来，按背包已满处理，不再拾取。"""

PICKUP_TAP_WAIT: float = 0.4
"""点按 F 拾取后，同一条目这么久内不再重按（秒）。

只限同名条目：游戏反应慢时，前几帧可能还是刚按过的那件，重按会白白计入
``PICKUP_TAP_LIMIT``。换成别的拾取物时不等，两帧判稳后就按。
"""

PICK_ALL_INITIAL_WAIT: float = 0.10
"""点按 F「全部拾取」后首次回读前的等待（秒），覆盖 0.05s 点按与游戏首帧响应。"""

PICK_ALL_POLL_SECONDS: float = 0.05
"""点按 F「全部拾取」后轮询结果格是否清空的间隔（秒）。"""

PICK_ALL_MAX_WAIT: float = PICKUP_TAP_WAIT
"""点按 F「全部拾取」后的最长等待上限（秒）；背包已满导致结果格仍有剩余时到此放行。"""

PICKUP_FAST_SECONDS: float = 0.6
"""移动中连续走「拾取快轮」的最长时长（秒）。

快轮只截图、找高亮、读选中行，跳过定位、提示列表和朝向。到时做一轮
完整判断，更新到点、转向目标和卡位计时，然后可以再开快轮。
必须小于转向线程的提交新鲜度 ``TURN_TARGET_FRESH_SECONDS``（1 秒），否则转向线程会自己停。
"""

PICKUP_FAST_AFTER_TAP: float = 0.6
"""点按 F 拾取后这么久内，选中行读不到也继续走快轮（秒），用来跨过列表重排的过渡帧。"""

PICKUP_FAST_BEACON_DISTANCE: float = 20.0
"""按住前进走向途径点、且上一轮完整判断时距离不超过这个值（像素）时，不走快轮。

冲刺约 24px/秒，快轮期间不做到点判定，离得太近会冲过途径点。
"""

MERGED_LOG_NODES: frozenset[str] = frozenset({'导航移动', '碎步移动'})
"""等待轮的决策日志可以合并的节点。"""

NUMBER_PATTERN: re.Pattern[str] = re.compile(r'-?\d+(?:\.\d+)?')
"""合并决策日志时，状态文字里的数字一律视为相同。"""

SELECT_TIMEOUT: float = 8.0
"""同一目标连续无法选中时的最长等待，包含滚轮后的重新识别。"""

STRAY_UNLOCK_ESC_LIMIT: int = 2
"""移动中误开了非目标电子保险箱的解锁界面时，同一个点最多按几次 esc 退出。"""


def _closest_on_segment(
    start: BagelRoutePoint, end: BagelRoutePoint, position: tuple[float, float],
) -> tuple[float, float]:
    """线段上离角色位置最近的点，夹在线段两端之间。

    Args:
        start: 线段起点，路径上的上一个点。
        end: 线段终点，当前目标点。
        position: 角色当前坐标。

    Returns:
        线段上的最近点坐标。
    """
    dx, dy = end.x - start.x, end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return (float(start.x), float(start.y))
    t = ((position[0] - start.x) * dx + (position[1] - start.y) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return (start.x + t * dx, start.y + t * dy)


class BagelRouteWalker(BagelOperation):
    """按打点顺序执行路线，箱类按类型交互。"""

    STATUS_ANCHOR_MISS: str = '出生点没有可执行路线，退出重开'
    STATUS_SPAWN_FAILED: str = '出生点识别失败'
    STATUS_CONFIG_INVALID: str = '路线配置无效'
    STATUS_NOT_FOREGROUND: str = '游戏不在前台，已停止'
    STATUS_DONE: str = '路线走完'
    STATUS_KEYCARD_MISSING: str = '刷卡处显示没有对应门禁卡'
    STATUS_FAIL_SURVEY: str = '失败前补读调查收益'

    def __init__(
        self, ctx: ZContext, blueprint_priority: bool = True,
        route_ref: BagelRouteRef | None = None, keycard_room: str | None = None,
        turn_memory: TurnRatioMemory | None = None,
        organize_backpack: bool = False,
        map_layers: list[MapLocator] | None = None,
    ) -> None:
        """加载出生点清单并准备定位。

        Args:
            ctx: 运行上下文。
            blueprint_priority: 入箱时蓝图是否优先。
            route_ref: 试跑时指定的路线，关闭的路线也可以试跑；None 表示正式运行，
                识别出生点后执行该出生点当前开启的路线。试跑开房卡入口段时接上
                该房间开启的房间段；试跑房间段时以房间入口代替出生点锚定起点。
            keycard_room: 正式运行开房卡模式时的目标房间；识别出生点后执行该出生点
                去这个房间的「入口段 + 房间段」。None 表示跑刀。试跑时忽略。
            turn_memory: 同一次应用运行内的跨局转向校准记忆；本局从它的值起步，
                局末由 remember_turn_ratio 记回。None 表示从 1.0 起步、局末不记。
            organize_backpack: 是否允许在路线最后一个容器的搜查面板内就地整理背包；
                正式应用传入 True，路线试跑默认 False。
            map_layers: 调用方已加载的只读地图层，多局共用；None 表示本执行器自己加载。
                每层约 6 MiB，23 层合计约 135 MiB，逐局加载会让内存随局数上涨。

        Raises:
            ValueError: 出生点清单无法读取，或指定路线不存在、损坏、绑定的出生点未登记。
        """
        # 不加「检测游戏窗口」节点：执行器只在确认局内后创建，这个节点用不上；
        # 而它的回调闭包引用执行器自身，执行器结束后要等完整垃圾回收才能释放。
        super().__init__(ctx, op_name='贝果-路线执行', timeout_seconds=-1, need_check_game_win=False)
        self.store: BagelRouteStore = BagelRouteStore(self.ctx)
        self.keycard_room: str | None = keycard_room if route_ref is None else None
        """正式运行开房卡模式的目标房间；跑刀与试跑为 None。"""
        try:
            self.spawns: list[BagelSpawnPoint] = self.store.load_spawns()
            self.trial_route: BagelRoute | None = (
                None if route_ref is None else self.store.trial_route(route_ref)
            )
            if self.trial_route is not None and self.trial_route.is_room_segment:
                # 房间段没有出生点：把房间入口当作唯一的出生点来锚定起点。
                self.spawns = [self._room_anchor(self.trial_route)]
        except BagelRouteConfigError as error:
            raise ValueError(str(error)) from error
        if self.trial_route is not None and not self.trial_route.is_room_segment and not any(
            spawn.spawn_id == self.trial_route.spawn_id for spawn in self.spawns
        ):
            raise ValueError(f'路线绑定的出生点未登记：{self.trial_route.spawn_id}')
        self.routes: list[BagelRoute] = []
        """本局执行的路线快照；识别出生点后才确定，只含一条。"""
        self.spawn: BagelSpawnPoint | None = None
        """本局识别到的出生点。"""
        self.spawn_started_at: float | None = None
        """出生点识别的起始时刻。"""
        self.spawn_last_match: BagelSpawnMatch | None = None
        """上一帧的出生点匹配结论；定位失败时清空。"""
        self.spawn_streak: int = 0
        """连续得出相同出生点结论的新截图次数。"""
        self.blueprint_priority: bool = blueprint_priority
        self.locator: MultiMapLocator = MultiMapLocator(
            load_map_locators() if map_layers is None else map_layers,
        )
        self.route_index: int = 0
        self.cursor: int = 0
        self.current_kind: str | None = None
        self.f_pressed: bool = False
        self.door_missing_at: float | None = None
        self.press_f_taps: int = 0
        self.press_f_at: float = float('-inf')
        """本点上次点按 F 的时刻（monotonic）。"""
        self.mechanical_hold_until: float | None = None
        self.mechanical_retried: bool = False
        self.mechanical_lost_since: float | None = None
        """机械保险箱长按中 F 键开始消失的时刻；F 键在时为 None。"""
        self.mechanical_lost_count: int = 0
        """当前点因长按中提示消失而重新靠近的次数。"""
        self.card_hold_started: float | None = None
        """刷卡处开始长按 F 的时刻；没在长按时为 None。"""
        self.card_retried: bool = False
        """刷卡处是否已经超时重按过一次。"""
        self.card_held: bool = False
        """当前刷卡处是否已经长按过；长按过后提示消失按刷完处理。长按中提示消失、回去试探时清掉。"""
        self.card_pressed: bool = False
        """当前刷卡处是否按下过长按；试探时不清，只在换点时清。读到计数 0 时据此区分门口一开始就是 0 还是刷后变 0。"""
        self.card_seen_loaded: bool = False
        """当前刷卡处是否读到过非 0 的持卡计数。"""
        self.card_released_at: float | None = None
        """当前刷卡处最近一次长按松手的时刻。"""
        self.card_spent_since: float | None = None
        """长按过后读到计数 0、开始等提示消失的时刻；没在等时为 None。"""
        self.card_lost_since: float | None = None
        """还没长按时刷卡条目消失的起始时刻。"""
        self.card_stable_since: float | None = None
        """刷卡条目与 F 键开始连续对准的时刻；未对准时为 None。"""
        self.card_hold_lost_since: float | None = None
        """长按中刷卡条目整条消失的起始时刻；条目在时为 None。"""
        self.card_hold_lost_count: int = 0
        """当前房卡门因长按中条目消失而松手回去试探的次数。"""
        self.stray_unlock_escapes: int = 0
        """当前点为退出误开的解锁界面已按 esc 的次数。"""
        self.blocked_door_taps: int = 0
        """当前点受阻时为开门已点按 F 的次数。"""
        self.pickup_text: str | None = None
        """正在点按 F 拾取的条目文字；没在拾取时为 None。整局有效，换点不清。"""
        self.pickup_taps: int = 0
        """对 ``pickup_text`` 已点按 F 的次数。"""
        self.pickup_ignored: set[str] = set()
        """本局捡不起来的拾取物条目文字，之后路过不再按。"""
        self.pickup_failures: int = 0
        """本局捡不起来的拾取物条数。"""
        self.pickups: int = 0
        """本局按提示消失计的已拾取条数，只用于日志。"""
        self.pickup_tap_at: float = float('-inf')
        """最近一次点按 F 拾取的单调时钟时刻。"""
        self.pickup_pending: bool = False
        self.pickup_fast_since: float | None = None
        """本段连续拾取快轮的起始时刻；没在走快轮时为 None。"""
        self.pickup_fast_rounds: int = 0
        """本段已走的快轮轮数，只用于日志。"""
        self.pickup_fast_presses: int = 0
        """本段开始时的 ``pickup_presses``，用于日志统计本段点按次数。"""
        self.perf: RoundPerf = RoundPerf()
        """本轮分段计时，只用于决策日志。"""
        self.log_merger: RepeatMerger = RepeatMerger()
        """合并连续相同的移动等待行，只用于决策日志。"""
        self.prompt_frame: PromptFrame | None = None
        self.selection: PromptSelection = PromptSelection()
        self.select_since: float | None = None
        self.cleanup_since: float | None = None
        self.scanned_rows: set[tuple[str, str, int]] = set()
        self.opening_blocking_door: bool = False
        self.pickup_presses: int = 0
        """本局为地面拾取物和容器「全部拾取」点按 F 的总次数。
        大于 ``organized_at_presses`` 时背包里可能有未整理的新物品。"""
        self.organize_backpack: bool = organize_backpack
        """是否允许在最后一个容器的搜查面板内就地整理背包。"""
        self.organized_at_presses: int = 0
        """上次成功整理背包时的 ``pickup_presses``；后续若再拾取，退出前仍会兜底整理。"""
        self._last_result_centers: tuple[Point, ...] = ()
        """最近一次入箱已定位的结果格心，用于跳过空箱全部拾取及轮询清空。"""
        self._pick_all_sent_at: float | None = None
        """本次「全部拾取」点按 F 的时刻（monotonic）；等待生效期间不为 None。"""
        self._last_pick_all_at: float | None = None
        """最近一次「全部拾取」点按 F 的时刻，供容器内整理空包复核使用。"""
        self.card_probe_step: int | None = None
        """房卡门左右试探已执行到第几步；没在试探时为 None。"""
        self.card_verify_since: float | None = None
        """判定刷完松手的时刻；松手后确认刷卡结果期间不为 None。"""
        self.card_swiped: bool = False
        """本局是否已经刷开房卡门；刷开后门禁卡已用掉，结算时据此决定卡的去留。"""
        self.position: tuple[float, float] | None = None
        self.last_map_id: str | None = None
        self.locate_misses: int = 0
        self.locate_miss_since: float | None = None
        """连续定位失败的起始时刻，恢复定位后清空。"""
        self.hud_checked_at: float = float('-inf')
        """上次检查暂停菜单的时刻（单调时钟），用于定位成功时的节流。"""
        self.angle_miss_since: float | None = None
        """连续读不到镜头朝向的起始时刻，读到朝向后清空。"""
        self.last_position: tuple[float, float] | None = None
        self.steps: int = 0
        self.stuck_since: tuple[tuple[float, float], float] | None = None
        self.stuck_steps: int = 0
        """当前点已经侧移了几次，进入新点时清零。"""
        self.slow_stuck_since: tuple[tuple[float, float], float] | None = None
        """慢步段位置开始停滞的位置与时刻；慢步每步都松键，不能用 ``stuck_since``。"""
        self.retreat_point: BagelRoutePoint | None = None
        """受阻后正在退回的上一个点；只走到它的坐标，不重做它的交互。"""
        self.retreat_since: float | None = None
        self.retreats: int = 0
        """当前点已经退回重走的次数，进入新点时清零。"""
        self.target_wait_since: float | None = None
        self.electronic_small_steps: bool = False
        """当前电子保险箱已进入短步阶段，定位回到半径外时也不恢复长步。"""
        self.small_step_since: float | None = None
        """进入 3px 碎步段的起始时刻，用于碎步超时判定。"""
        self.point_started_at: float | None = None
        self.last_input_frame: float | None = None
        self.decision: str = '准备开始'
        """最近一轮的决策文字，供执行态悬浮窗显示。"""
        self.sprinting: bool = False
        """当前是否在冲刺。"""
        self.turning_in_place: bool = False
        """巡航中是否已进入停车等转向；用于 ``WALK_STOP_TURN_ANGLE`` 与 ``WALK_TURN_ANGLE`` 滞回判断。"""
        self.cruise_over_frame: float | None = None
        """巡航中偏差首次超过停车阈值的截图时间戳；下一张不同截图仍超过才停车，过滤单帧跳变。"""
        self.last_dodge_at: float | None = None
        """上次点闪避的时刻（monotonic），用于冲刺点闪避的间隔控制。"""
        self.turn_memory: TurnRatioMemory | None = turn_memory
        """跨局转向校准记忆；试跑时为 None，每次从 1.0 起步。"""
        initial_ratio = turn_memory.hint if turn_memory is not None else 1.0
        trust_initial = bool(turn_memory is not None and turn_memory.trusted)
        log.info(
            '贝果转向本局 turn_dx=%.4f，起步校准值 %.3f（%s），局内由转向线程学习；不写回 turn_dx',
            self._turn_dx(), initial_ratio, '跨局可信' if trust_initial else '未校准',
        )
        self.elevator_phase: str | None = None
        """电梯当前阶段：进门 / 选楼层 / 出电梯；不在电梯里时为 None。"""
        self.elevator_since: float | None = None
        """当前电梯阶段的起始时刻，用于计时。"""
        self.elevator_last_tap: float = 0.0
        """上次为推进对话而点按的时刻，控制每秒一次。"""
        self.capture_stats: CaptureStats = CaptureStats(ctx.controller.game_win.get_hwnd)
        self._captured_frame: MatLike | None = None
        self._capture_start: float = 0.0
        self._turn_resume_at: float = float('-inf')
        self.input_worker: BagelInputWorker = BagelInputWorker(
            ctx, initial_turn_ratio=initial_ratio, trust_initial_ratio=trust_initial, stats=self.capture_stats,
        )
        """按键单独线程：主线投递即返回；同种操作不打断自己，不同操作打断旧的。"""
        self.ledger: BagelSafeLedger = BagelSafeLedger()
        """本局安全箱物品，只用于统计日志。"""
        self.survey_value: int = 0
        """本局调查收益的最后一次读数，只用于统计日志。

        零携带入场，开局为 0；每次关闭搜查面板后读 HUD 更新，装箱或关闭面板失败时再补读一次。
        """
        self.fail_reason: str | None = None
        """装箱或关闭面板失败的原因；补读调查收益后按它结束本局。"""
        self.fail_screenshot: tuple[float, MatLike] | None = None
        """失败时的截图时间和截图；补读结束后恢复，现场截图仍是失败画面。"""
        self.fail_survey_started: float | None = None
        """开始补读调查收益的时刻。"""
        self.fail_survey_last: int | None = None
        """上一帧补读到的 HUD 读数；连续两帧相同才采用。"""

    def state_snapshot(self) -> dict[str, object]:
        """给悬浮窗用的执行状态快照。

        每次调用都按当前游标现算，不维护第二份真值，避免两边对不上。

        Returns:
            含决策、路线进度、点位与当前位置的字典。
        """
        route = self.routes[self.route_index] if self.routes else None
        point = self._current_point()
        distance: float | None = None
        if point is not None and self.position is not None:
            distance = hypot(point.x - self.position[0], point.y - self.position[1])
        return {
            'decision': self.decision,
            'route_id': route.route_id if route is not None else '',
            'route_name': route.name if route is not None else '',
            'spawn_name': self.spawn.name if self.spawn is not None else '',
            'route_index': self.route_index,
            'route_count': len(self.routes),
            'cursor': self.cursor,
            'total': len(route.points) if route is not None else 0,
            'points': [(point.kind, (point.x, point.y)) for point in route.points]
                      if route is not None else [],
            'layer': route.points[0].layer if route is not None and route.points else '',
            'position': self.position,
            'map_id': self.last_map_id,
            'distance': distance,
        }

    def _execute_one_round(self) -> OperationRoundResult:
        """每轮前先确认游戏在前台，结束后更新决策文字。

        用户切出游戏时，按键会打进别的窗口，必须先松键并停下，
        否则会在后台一直空转。
        """
        self.capture_stats.start()
        self.perf.start()
        if not self._game_in_foreground():
            self._release_forward()
            self.decision = '游戏不在前台 · 已停止'
            return self.round_fail(self.STATUS_NOT_FOREGROUND)
        result = super()._execute_one_round()
        node_name = '未知' if self._current_node is None else self._current_node.cn
        status = result.status or result.status_display or ''
        self.decision = f'{node_name} · {status}' if status else node_name
        mergeable = node_name in MERGED_LOG_NODES and result.result == OperationRoundResultEnum.WAIT
        self.capture_stats.finish_round(movement_stage(node_name, status))
        self._log_decision(node_name, status, mergeable=mergeable)
        return result

    def _log_decision(self, node_name: str, status: str, mergeable: bool = False) -> None:
        """每轮一条决策日志，带点位、距离、位置和分段耗时，便于核对停在哪一步。

        导航与碎步节点的等待轮，连续相同的决策合并输出（见 ``RepeatMerger``）：
        首行照打，之后只累计，键变了或满 1 秒再打一条汇总。键是节点、
        去掉数字后的状态文字和游标，所以「停车等转向 61度」与「停车等转向 18度」算同一条。
        卡住时同一条反复出现，每秒仍留一条完整行和一条汇总，「一直短步」「一直转圈」照样看得到。

        到点那一轮不能把「当前游标」和「位置」直接配对：游标已经推进到
        下一个点，位置还是走到上一个点时的位置，两边一算就得到一个
        几十像素的假距离，看起来像「离老远就说到点」。所以到点这一轮
        只打位置，判定距离由 navigate 自己在到点时打一条独立的 log。
        """
        point = self._current_point()
        parts = [f'[{node_name}]']
        if status:
            parts.append(status)
        if point is not None:
            parts.append(f'第{self.cursor + 1}点 {KIND_LABELS.get(point.kind, point.kind)}')
        arrived = node_name == '导航移动' and status.startswith('已到点')
        if point is not None and self.position is not None and not arrived:
            distance = hypot(point.x - self.position[0], point.y - self.position[1])
            parts.append(f'距 {distance:.1f}px')
        if self.position is not None:
            parts.append(f'位置 ({self.position[0]:.0f},{self.position[1]:.0f})')
        if self.current_kind is not None:
            parts.append(f'提示 {self.current_kind}')
        key = f'{node_name}|{NUMBER_PATTERN.sub("#", status)}|{self.route_index}|{self.cursor}' if mergeable else None
        for line in self.log_merger.submit(key, ' · '.join(parts), self.perf):
            log.info('贝果执行 %s', line)

    def _flush_decision_log(self) -> None:
        """打出尚未汇总的合并轮次，用于暂停和结束时。"""
        for line in self.log_merger.flush():
            log.info('贝果执行 %s', line)

    def screenshot(self) -> MatLike:
        """框架每轮开头的截图，计入本轮耗时。"""
        with self.perf.measure('截图'):
            start = time.perf_counter()
            frame = super().screenshot()
            self._capture_start = start
            self._captured_frame = frame
            self.capture_stats.record('主线截图', time.perf_counter() - start)
            return frame

    def _after_round_wait(self, wait: float | None = None, wait_round_time: float | None = None) -> None:
        """轮末睡眠，计入本轮耗时。

        区域核对等内部调用也会经过这里，但不等待；只在真的要睡时计入，免得次数虚高。
        """
        if (wait is None or wait <= 0) and (wait_round_time is None or wait_round_time <= 0):
            super()._after_round_wait(wait=wait, wait_round_time=wait_round_time)
            return
        with self.perf.measure('睡眠'):
            super()._after_round_wait(wait=wait, wait_round_time=wait_round_time)

    def _has(self, area: str, screen_name: str = '贝果-局内') -> bool:
        """画面区域核对多是 OCR，计入本轮耗时。"""
        with self.perf.measure('区域'):
            return super()._has(area, screen_name)

    def _game_in_foreground(self) -> bool:
        """游戏窗口是否在前台；拿不到窗口句柄时按前台处理，避免误停。"""
        game_win = self.ctx.controller.game_win
        hwnd = game_win.get_hwnd()
        if hwnd is None:
            return True
        return is_foreground(hwnd)

    # ---------- 定位 ----------

    def minimap(self) -> MatLike:
        """按原生区域裁出小地图；框架截图为 RGB，定位与朝向直接使用。"""
        area = self.ctx.screen_loader.get_area('贝果-局内', '定位小地图')
        if area is None:
            raise ValueError('缺少贝果定位小地图区域')
        crop = cv2_utils.crop_image_only(self.last_screenshot, area.pc_rect)
        if crop is None or crop.size == 0:
            raise ValueError('小地图裁剪为空')
        if crop.ndim == 2:
            crop = np.stack([crop] * 3, axis=-1)
        return crop

    def locate(self) -> tuple[float, float] | None:
        """定位一次并更新所在地图。"""
        with self.perf.measure('定位'):
            result = self.locator.locate(self.minimap())
        if result is None:
            return None
        locator, _score, center = result
        self.last_map_id = locator.map_id
        self.position = (float(center[0]), float(center[1]))
        return self.position

    def player_angle(self) -> float | None:
        """识别镜头朝向角，正右为 0、逆时针增加。

        使用 ``analyze`` 返回的扇形方向，与独立补采使用的角度一致。
        """
        with self.perf.measure('朝向'):
            _marker_vertices, _marker_edges, sector, _radius, _center = analyze(self.minimap())
        return sector

    # ---------- 节点 ----------

    @operation_node(name='出生点识别', is_start_node=True)
    def anchor(self) -> OperationRoundResult:
        """识别出生点并确定本局路线，期间不移动角色。

        每轮使用新截图定位一次。连续 ``SPAWN_CONFIRM_FRAMES`` 次得出相同结论才采用；
        定位缺失或命中多个出生点时等下一帧，超过 ``SPAWN_OBSERVE_SECONDS`` 仍无结论则失败，
        不按路线顺序任取一条。
        """
        now = time.monotonic()
        if self.spawn_started_at is None:
            self.spawn_started_at = now
        if now - self.spawn_started_at > SPAWN_OBSERVE_SECONDS:
            reason = '定位失败' if self.spawn_last_match is None else self.spawn_last_match.describe()
            log.warning('贝果出生点识别：%.0f 秒内没有稳定结论（%s）', SPAWN_OBSERVE_SECONDS, reason)
            return self.round_fail(self.STATUS_SPAWN_FAILED, data=reason)
        position = self.locate()
        if position is None:
            self.spawn_last_match = None
            self.spawn_streak = 0
            return self.round_wait('定位失败，等下一帧', wait=0.2)
        match = match_spawn(self.spawns, self.last_map_id or '', position)
        self.spawn_streak = self.spawn_streak + 1 if match.same_as(self.spawn_last_match) else 1
        self.spawn_last_match = match
        if match.kind == SPAWN_AMBIGUOUS or self.spawn_streak < SPAWN_CONFIRM_FRAMES:
            return self.round_wait(match.describe(), wait=0.2)
        if match.kind == SPAWN_MISS:
            log.info(
                '贝果出生点识别：位置 (%.0f,%.0f) 不在任何已登记出生点范围内',
                position[0], position[1],
            )
            return self.round_fail(self.STATUS_ANCHOR_MISS)
        if match.kind != SPAWN_HIT or match.spawn is None:
            return self.round_fail(self.STATUS_SPAWN_FAILED, data=match.describe())
        return self._start_route_at(match.spawn)

    def _room_anchor(self, route: BagelRoute) -> BagelSpawnPoint:
        """房间段试跑用的起点：所属房间的入口。

        Raises:
            BagelRouteConfigError: 房间未登记。
        """
        room_id = route.keycard.room_id if route.keycard is not None else ''
        room = next((item for item in self.store.load_rooms() if item.room_id == room_id), None)
        if room is None:
            raise BagelRouteConfigError(f'房间段所属房间未登记：{room_id}')
        return room.as_spawn()

    def _start_route_at(self, spawn: BagelSpawnPoint) -> OperationRoundResult:
        """按识别到的出生点确定路线快照，并从第 1 点开始。

        Args:
            spawn: 已确认的出生点；单独试跑房间段时是房间入口。
        """
        self.spawn = spawn
        # 一局只在出生点所在地图里：锁定后，界面挡住小地图时不会误锁到其它地图。
        self.locator.lock_map(spawn.map_id)
        if self.trial_route is not None and self.trial_route.is_room_segment:
            route = self.trial_route
        elif self.trial_route is not None:
            route = self.trial_route
            if route.spawn_id != spawn.spawn_id:
                bound = next(
                    (item.name for item in self.spawns if item.spawn_id == route.spawn_id),
                    route.spawn_id,
                )
                return self.round_fail(
                    self.STATUS_SPAWN_FAILED,
                    data=f'角色在{spawn.name}，路线「{route.name}」绑定的是{bound}',
                )
        elif self.keycard_room is not None:
            try:
                route = self.store.keycard_route(self.keycard_room, spawn.spawn_id)
            except BagelRouteConfigError as error:
                log.error('贝果开房卡房间 %s 的路线配置无效：%s', self.keycard_room, error)
                return self.round_fail(self.STATUS_CONFIG_INVALID, data=str(error))
            if route is None:
                log.info('贝果出生点识别：%s 没有开启去房间 %s 的入口段', spawn.name, self.keycard_room)
                return self.round_fail(self.STATUS_ANCHOR_MISS)
        else:
            try:
                route = self.store.active_route(spawn.spawn_id)
            except BagelRouteConfigError as error:
                log.error('贝果出生点 %s 的路线配置无效：%s', spawn.name, error)
                return self.round_fail(self.STATUS_CONFIG_INVALID, data=str(error))
            if route is None:
                log.info('贝果出生点识别：%s 没有开启的路线', spawn.name)
                return self.round_fail(self.STATUS_ANCHOR_MISS)
        problem = check_route_start(route, spawn)
        if problem is not None:
            return self.round_fail(self.STATUS_CONFIG_INVALID, data=f'路线「{route.name}」{problem}')
        self.routes = [route]
        self._enter_point(0, 0)
        log.info(
            '贝果出生点识别：%s，执行路线「%s」（%s），共 %s 点',
            spawn.name, route.name, route.ref, len(route.points),
        )
        return self.round_success('已锚定')

    @node_from(from_name='游标推进', status='已推进')
    @node_from(from_name='导航移动', status='已到点')
    @node_from(from_name='导航移动', status='跳过此点')
    @node_from(from_name='碎步移动', status='跳过此点')
    @node_from(from_name='进出电梯', status='已离开电梯')
    @operation_node(name='点调度', mute=True)
    def dispatch(self) -> OperationRoundResult:
        """按游标当前点分派到导航、碎步或现场判断。"""
        point = self._current_point()
        if point is None:
            return self.round_success('路线走完')
        if point.kind == BEACON:
            return self.round_success('下一格是途径点')
        if point.kind == ELEVATOR:
            return self.round_success('下一格是电梯')
        return self.round_success('下一格是箱类')

    @node_from(from_name='点调度', status='路线走完')
    @node_from(from_name='导航移动', status='路线走完')
    @node_from(from_name='碎步移动', status='路线走完')
    @operation_node(name='现场交互判断')
    def check_surroundings(self) -> OperationRoundResult:
        """在路线末尾原地处理可见条目，忽略绿勾和刷卡处。"""
        recovered = self._recover_pickup_panel()
        if recovered is not None:
            return recovered
        self._release_forward()
        self.input_worker.submit_turn(None)
        now = time.monotonic()
        if self.target_wait_since is None:
            self.target_wait_since = now
        elapsed = now - self.target_wait_since
        if elapsed < SELECT_TIMEOUT:
            pickup = self._try_pickup(all_rows=True)
            if pickup is not None:
                return pickup
            self.current_kind = self._nearby_kind()
            if self.current_kind is not None:
                self.target_wait_since = None
                return self.round_success('现场有交互物')
            scan = self._scan_more()
            if scan is not None:
                return scan
            if elapsed < SURROUND_CHECK_SECONDS:
                return self.round_wait('原地观察交互提示', wait=0.1)
        self.target_wait_since = None
        self._log_leftover_prompts()
        return self.round_success('现场无交互物')

    def _nearby_kind(self) -> str | None:
        """路线只接当前点对应的类型；路线末尾可处理可见的新容器或门。"""
        point = self._current_point()
        if point is None:
            kinds = ('box', 'door', 'mechanical_safe', 'electronic_safe')
        elif point.kind in BOX_CATEGORIES:
            kinds = ('box',)
        else:
            kinds = ('card_reader' if point.kind == KEYCARD_DOOR else point.kind,)
        return next((row.kind for row in self._prompts().rows if row.kind in kinds), None)

    @node_from(from_name='现场交互判断', status='现场有交互物')
    @operation_node(name='现场交互')
    def interact_nearby(self) -> OperationRoundResult:
        """首次按提示确定类型，之后检查本次交互结果，不要求场景提示仍可见。"""
        if self.current_kind is None:
            self.current_kind = self._nearby_kind()
        if self.current_kind is None:
            return self.round_retry('提示消失，再看一帧', wait=0.2)
        return self._interact_by_kind(self.current_kind)

    @node_from(from_name='现场交互判断', status='现场无交互物')
    @operation_node(name='路线结束')
    def finish_route(self) -> OperationRoundResult:
        """本局只执行一条路线，走完且现场没有交互物即结束。"""
        log.info('贝果路线执行：路线「%s」走完', self.routes[self.route_index].name)
        return self.round_success(self.STATUS_DONE)

    # ---------- 导航 ----------

    @node_from(from_name='出生点识别', status='已锚定')
    @node_from(from_name='点调度', status='下一格是途径点')
    @operation_node(name='导航移动', mute=True)
    def navigate(self) -> OperationRoundResult:
        """朝当前点移动；途径点 3px 到达，箱类见到提示就停。

        到点后必须回点调度重新分派，不能直接再进本节点：游标已经推进到
        下一个点，那个点可能是电梯或箱子，得由点调度按种类交给对应节点。
        直接自环会让电梯点落到箱类分支，被当成箱子慢步逼近。
        """
        return self._move_toward()

    @node_from(from_name='点调度', status='下一格是箱类')
    @node_from(from_name='按类型交互', status='刷卡提示消失')
    @node_from(from_name='按类型交互', status='机械保险箱提示消失')
    @operation_node(name='碎步移动', mute=True)
    def small_step(self) -> OperationRoundResult:
        """按距离逼近箱类点，统一检查到位、提示与单点超时。"""
        return self._move_toward()

    # ---------- 交互 ----------

    @node_from(from_name='点调度', status='下一格是电梯')
    @operation_node(name='进出电梯')
    def elevator(self) -> OperationRoundResult:
        """进出电梯：走向电梯，点楼层选项，再走出来。

        电梯点在进门前能正常定位，进电梯后小地图整块消失，所以不能按
        3px 判到达（用户定义：读不到朝向就是进了电梯）。进去之后分三段，
        每段只等它自己的信号，统一不设超时——卡在电梯里连游戏都退不出来。

        1. 进门：朝电梯点走，读不到朝向即进入电梯。
        2. 等选项：一直点屏幕中间推进对话，直到楼层选项出现。
        3. 点选项：一直点那个选项，直到它从画面上消失。
        4. 出梯：一直按住配置的方向，直到小地图回来。

        点上没有楼层文字（`elevator_option` 为空）的电梯没有第 2、3 段，
        进门后直接进出梯段。
        """
        point = self._current_point()
        if point is None:
            return self.round_success('路线走完')
        if self.elevator_phase is None:
            # 首次进入本节点必须从「进门」阶段开始。之前漏了这一步，
            # phase 一直是 None，直接落到出梯阶段，「朝电梯走」整个没执行过。
            self.elevator_phase = ELEVATOR_PHASE_ENTER
            self.elevator_since = None
        phase = self.elevator_phase
        if phase == ELEVATOR_PHASE_ENTER:
            return self._elevator_enter(point)
        if phase == ELEVATOR_PHASE_WAIT_OPTION:
            return self._elevator_wait_option(point)
        if phase == ELEVATOR_PHASE_CLICK_OPTION:
            return self._elevator_click_option(point)
        return self._elevator_exit(point)

    def _elevator_enter(self, point: BagelRoutePoint) -> OperationRoundResult:
        """朝电梯点走，途中读不到朝向就是进了电梯。

        电梯点没有「到达」这回事：小地图会整块消失，坐标和朝向都读不到，
        3px 这种判据在电梯口根本不成立。所以这里只朝点走，一旦
        `player_angle()` 读不到就当已进电梯（用户定义）。
        """
        if self.player_angle() is None:
            self._release_forward()
            self.elevator_since = None
            if point.elevator_option:
                self.elevator_phase = ELEVATOR_PHASE_WAIT_OPTION
                self.elevator_last_tap = 0.0
                return self.round_wait('小地图已消失，视为进入电梯', wait=0.2)
            self.elevator_phase = ELEVATOR_PHASE_EXIT
            return self.round_wait('小地图已消失，这电梯没有楼层选项', wait=0.2)
        if self.elevator_since is None:
            self.elevator_since = time.monotonic()
        if time.monotonic() - self.elevator_since >= ELEVATOR_ENTRY_TIMEOUT:
            self._release_forward()
            self.elevator_phase = None
            self.elevator_since = None
            return self.round_recoverable_fail('电梯门口走了 10 秒还没进入电梯')
        position = self.locate()
        if position is not None:
            angle = self.player_angle()
            if angle is not None:
                target_angle = self._target_angle(position, point)
                angle_diff = cal_utils.angle_delta(angle % 360, target_angle)
                self._submit_turn(target_angle, angle, deadband=TURN_DEADBAND_FAST)
                if abs(angle_diff) > WALK_TURN_ANGLE:
                    # 方向偏太多先停下等转，不然会走到旁边去。
                    self._release_forward()
                    return self.round_wait(f'朝电梯转向 {angle_diff:.0f}度', wait=0.05)
                self.input_worker.hold_forward()
                return self.round_wait('走向电梯', wait=0.15)
        # 定位暂时丢失但朝向还在（贴墙等）：先不松前进，等下一帧。
        self.input_worker.hold_forward()
        return self.round_wait('走向电梯，定位暂缺', wait=0.15)

    def _elevator_wait_option(self, point: BagelRoutePoint) -> OperationRoundResult:
        """一直点屏幕中间推进对话，直到楼层选项出现。不设超时。"""
        found = self._find_elevator_option(point.elevator_option)
        if found is not None:
            self.elevator_phase = ELEVATOR_PHASE_CLICK_OPTION
            return self.round_wait(f'楼层选项已出现：{found.data}', wait=0.1)
        if time.monotonic() - self.elevator_last_tap >= ELEVATOR_OPTION_TAP_SECONDS:
            self.elevator_last_tap = time.monotonic()
            self.input_worker.tap_center()
        return self.round_wait('等待电梯楼层选项', wait=0.2)

    def _elevator_click_option(self, point: BagelRoutePoint) -> OperationRoundResult:
        """一直点那个选项，直到它从画面上消失。不设超时。

        选项点中后会收起，收起即表示楼层已选、电梯开始移动。
        中途某一帧识别不到也算收起，所以这里用「识别到才点、识别不到就
        进下一段」，不需要连续几帧确认。
        """
        found = self._find_elevator_option(point.elevator_option)
        if found is None:
            self.elevator_phase = ELEVATOR_PHASE_EXIT
            return self.round_wait('楼层选项已消失，准备出梯', wait=0.1)
        self.ctx.controller.click(found.rect.center)
        return self.round_wait(f'点选楼层：{found.data}', wait=0.2)

    def _find_elevator_option(self, option: str) -> OcrMatchResult | None:
        """在楼层列表里找配置的文字。

        Args:
            option: 要选的楼层文字，如「负二层」。

        Returns:
            匹配到的 OCR 结果；没有则返回 None。
        """
        area = self.ctx.screen_loader.get_area('贝果-局内', '电梯选项')
        if area is None:
            return None
        results = self.ctx.ocr_service.get_ocr_result_list(
            image=self.last_screenshot, rect=area.pc_rect,
        )
        target = gt(option, 'game')
        for item in results:
            if target in str_utils.remove_whitespace(item.data):
                return item
        return None

    def _elevator_exit(self, point: BagelRoutePoint) -> OperationRoundResult:
        """一直按住配置的方向往前走，直到小地图回来。不设超时。"""
        if self.player_angle() is not None:
            self._release_forward()
            self._release_backward()
            self.elevator_phase = None
            self.elevator_since = None
            self._enter_point(self.route_index, self.cursor + 1)
            return self.round_success('已离开电梯')
        if self.elevator_since is None:
            self.elevator_since = time.monotonic()
        if point.elevator_exit == ELEVATOR_EXIT_BACKWARD:
            self.input_worker.hold_backward()
        else:
            self.input_worker.hold_forward()
        elapsed = time.monotonic() - self.elevator_since
        return self.round_wait(f'出电梯 {elapsed:.1f}s', wait=0.15)

    def _release_backward(self) -> None:
        """松开后退键。"""
        self.input_worker.stop_backward()

    @node_from(from_name='碎步移动', status='提示已出现')
    @node_from(from_name='导航移动', status='提示已出现')
    @operation_node(name='按类型交互')
    def interact(self) -> OperationRoundResult:
        """按当前点类型执行交互。"""
        if self.current_kind is None:
            return self.round_retry('未知交互类型', wait=0.2)
        return self._interact_by_kind(self.current_kind)

    def _interact_by_kind(self, kind: str) -> OperationRoundResult:
        """各动作先检查结果，只有即将按键时才重新选目标。"""
        recovered = self._recover_pickup_panel()
        if recovered is not None:
            return recovered
        if kind == 'electronic_safe':
            return self._run_unlock_safe()
        if kind == 'mechanical_safe':
            return self._hold_mechanical()
        if kind == 'card_reader':
            return self._hold_card_reader()
        return self._press_f(kind)

    def _run_unlock_safe(self) -> OperationRoundResult:
        """电子保险箱：光圈解锁子操作。"""
        self._release_forward()
        op = BagelUnlockSafe(self.ctx, prepare_open=self._prepare_safe_open)
        result = op.execute()
        self._adopt_screenshot(op)
        if result.success:
            return self.round_success('电子保险箱已解锁')
        return self.round_recoverable_fail(f'电子保险箱解锁失败：{result.status}')

    def _prepare_safe_open(self, op: BagelOperation) -> OperationRoundResult | None:
        """首次开箱与每次补按都用子操作本轮的新截图选行。"""
        self._adopt_screenshot(op)
        if not self._game_in_foreground():
            return self.round_fail(self.STATUS_NOT_FOREGROUND)
        waiting = self._prepare_interaction('electronic_safe')
        if waiting is None:
            self.selection.reset(self.last_screenshot_time)
        return waiting

    def _hold_mechanical(self) -> OperationRoundResult:
        """机械保险箱：提示出现时长按 F，直到搜查面板出现。

        长按发生在提示出现时，不是面板打开后；面板一出就松手，
        后面与普通箱走同一套收集入箱。面板用底部共用的安全箱标记确认，
        不依赖专有标题。
        """
        if self.mechanical_hold_until is None:
            if self._has('搜查安全箱'):
                self._reset_point_state()
                return self.round_success('机械保险箱已打开')
            waiting = self._prepare_interaction('mechanical_safe')
            if waiting is not None:
                return waiting
            self.mechanical_hold_until = time.monotonic() + MECHANICAL_HOLD_SECONDS
            self.input_worker.hold_interact()
            self.selection.reset(self.last_screenshot_time)
            return self.round_wait('长按F中')
        if self._has('搜查安全箱'):
            self.input_worker.release_interact()
            self._reset_point_state()
            return self.round_success('机械保险箱已打开')
        lost = self._mechanical_prompt_lost()
        if lost is not None:
            return lost
        if time.monotonic() >= self.mechanical_hold_until:
            self.input_worker.release_interact()
            if self.mechanical_retried:
                self._reset_point_state()
                return self.round_recoverable_fail('机械保险箱长按F超时')
            self.mechanical_retried = True
            self.mechanical_hold_until = None
            return self.round_wait('长按F超时，释放后重试', wait=0.3)
        return self.round_wait('长按F等待面板', wait=0.3)

    def _mechanical_prompt_lost(self) -> OperationRoundResult | None:
        """长按中 F 键连续消失 ``MECHANICAL_PROMPT_LOST_SECONDS``：松手回碎步移动重新靠近。

        只在路线上的机械保险箱点生效；路线末尾的现场交互没有当前点，
        回不到碎步移动，仍按原超时逻辑处理。同一点最多重新靠近
        ``MECHANICAL_PROMPT_LOST_LIMIT`` 次。

        Returns:
            转回碎步移动的结果；F 键仍在、还没确认消失或次数用完时返回 None。
        """
        point = self._current_point()
        if point is None or point.kind != 'mechanical_safe':
            return None
        if self.mechanical_lost_count >= MECHANICAL_PROMPT_LOST_LIMIT:
            return None
        now = time.monotonic()
        if self._interact_key_rows():
            self.mechanical_lost_since = None
            return None
        if self.mechanical_lost_since is None:
            self.mechanical_lost_since = now
        if now - self.mechanical_lost_since < MECHANICAL_PROMPT_LOST_SECONDS:
            return None
        self.input_worker.release_interact()
        self.mechanical_lost_count += 1
        self.mechanical_lost_since = None
        self.mechanical_hold_until = None
        self.current_kind = None
        self.target_wait_since = None
        self.small_step_since = None
        log.info(
            '贝果第 %s 点（机械保险箱）长按中提示消失，松手回碎步移动重新靠近（第 %s/%s 次），位置 %s',
            self.cursor + 1, self.mechanical_lost_count, MECHANICAL_PROMPT_LOST_LIMIT, self.position,
        )
        return self.round_success('机械保险箱提示消失')

    def _hold_card_reader(self) -> OperationRoundResult:
        """刷卡处：确认 F 键对准刷卡条目后长按 F，直到刷完。

        条目第二行是持卡计数。本点还没长按就读到 0，说明没带这个房间的门禁卡，
        重开也没用，直接停止任务；长按过后读到 0 见 ``_card_spent``。长按期间进度圈盖住钥匙图标，图标匹配不到，所以完成判据
        不看图标，而看 F 键：长按满 ``CARD_HOLD_MIN_SECONDS`` 后 F 键消失，
        或计数读成 0，就算刷完。

        判定刷完松手后确认：F 键在松手后 ``CARD_SWIPE_SETTLE_SECONDS`` 内仍对准刷卡条目，
        说明没刷上，重按一次。

        """
        now = time.monotonic()
        if self.card_hold_started is not None:
            return self._check_card_hold(now)
        if self.card_verify_since is not None:
            return self._verify_card_swipe(now)
        card = self._card_entry()
        if card is None:
            self.card_stable_since = None
            if self.card_held or self.card_spent_since is not None:
                self.card_swiped = True
                self._reset_point_state()
                return self.round_success('已完成')
            return self._card_prompt_lost(now)
        self.card_lost_since = None
        _, text = card
        count = self._card_count(text)
        if count is not None and count > 0:
            self.card_seen_loaded = True
        if count == 0:
            self._release_forward()
            if self.card_pressed:
                return self._card_spent(now, text)
            log.info('贝果刷卡处还没长按就显示持卡数为 0：%s', text)
            return self.round_fail(self.STATUS_KEYCARD_MISSING, data=text)
        waiting = self._prepare_interaction('card_reader')
        if waiting is not None:
            self.card_stable_since = None
            return waiting
        if self.card_stable_since is None:
            self.card_stable_since = now
        if now - self.card_stable_since < CARD_STABLE_SECONDS:
            return self.round_wait('刷卡提示已对准，等角色停稳', wait=0.1)
        self.card_stable_since = None
        self.card_hold_started = time.monotonic()
        self.card_held = True
        self.card_pressed = True
        self.input_worker.hold_interact()
        self.selection.reset(self.last_screenshot_time)
        return self.round_wait('长按F刷卡', wait=0.2)

    def _check_card_hold(self, now: float) -> OperationRoundResult:
        """长按中：判断刷完、超时重按或继续等。

        Args:
            now: 本轮的单调时钟读数。
        """
        elapsed = now - (self.card_hold_started or now)
        card = self._card_entry()
        count = self._card_count(card[1]) if card is not None else None
        if count is not None and count > 0:
            self.card_seen_loaded = True
        keys = self._interact_key_rows()
        if card is None and not keys:
            lost = self._card_hold_lost(now, elapsed)
            if lost is not None:
                return lost
        else:
            self.card_hold_lost_since = None
        done = count == 0 or (elapsed >= CARD_HOLD_MIN_SECONDS and not keys)
        if done:
            self.input_worker.release_interact()
            self.card_released_at = now
            log.info('贝果刷卡松手：长按 %.1f 秒，计数 %s', elapsed, '未读到' if count is None else count)
            self.card_hold_started = None
            self.card_verify_since = now
            return self.round_wait('松手确认刷卡结果', wait=CARD_VERIFY_WAIT_SECONDS)
        if elapsed < CARD_HOLD_SECONDS:
            return self.round_wait('长按F刷卡中', wait=0.2)
        self.input_worker.release_interact()
        self.card_released_at = now
        self.card_hold_started = None
        if self.card_retried:
            self._reset_point_state()
            return self.round_recoverable_fail('刷卡处长按F超时')
        self.card_retried = True
        return self.round_wait('刷卡长按超时，释放后重按', wait=0.3)

    def _card_spent(self, now: float, text: str) -> OperationRoundResult:
        """本点长按过后读到计数 0：不再重按，等提示消失。

        提示消失由 ``_hold_card_reader`` 判刷开；这里只处理提示还在的情况。从松手起满
        ``CARD_SPENT_WAIT_SECONDS`` 仍在时，读到过非 0 计数按本局失败，没读到过按没带卡停止。

        Args:
            now: 本轮的单调时钟读数。
            text: 刷卡条目文字。
        """
        if self.card_spent_since is None:
            self.card_spent_since = now
            log.info('贝果刷卡：长按过后计数为 0，不再重按，等提示消失：%s', text)
        released = self.card_released_at if self.card_released_at is not None else self.card_spent_since
        if now - released < CARD_SPENT_WAIT_SECONDS:
            return self.round_wait('长按过后计数为 0，等刷卡提示消失', wait=0.2)
        if self.card_seen_loaded:
            self._reset_point_state()
            return self.round_recoverable_fail(f'刷卡后计数为 0，但提示 {CARD_SPENT_WAIT_SECONDS:.0f} 秒没消失，门是否刷开不明')
        log.info('贝果刷卡：长按过后计数为 0，提示一直在，本点没读到过非 0 计数，按没带卡处理：%s', text)
        return self.round_fail(self.STATUS_KEYCARD_MISSING, data=text)

    def _card_hold_lost(self, now: float, elapsed: float) -> OperationRoundResult | None:
        """长按中刷卡条目与 F 键一起消失：在满 ``CARD_HOLD_MIN_SECONDS`` 前开始消失就没刷上。

        正常长按时进度圈会盖住钥匙图标，图标分数掉到 0.75–0.79、匹配不到，但 F 键一直在；
        滑出交互范围或转开时，条目和 F 键一起消失。所以只在两者都没有时才算消失。
        消失始于 ``CARD_HOLD_MIN_SECONDS`` 之后的，交回原判据（多半是刷完）；之前的，
        连续 ``CARD_PROMPT_LOST_SECONDS`` 后松手回碎步试探，与还没长按时提示消失走同一条路。
        同一个门最多 ``CARD_HOLD_LOST_LIMIT`` 次。

        Args:
            now: 本轮的单调时钟读数。
            elapsed: 本次长按已持续的秒数。

        Returns:
            等待或转回碎步移动的结果；交回原判据时返回 None。
        """
        if self.card_hold_lost_since is None:
            self.card_hold_lost_since = now
        started = self.card_hold_started if self.card_hold_started is not None else now
        if self.card_hold_lost_since - started >= CARD_HOLD_MIN_SECONDS:
            return None
        if now - self.card_hold_lost_since < CARD_PROMPT_LOST_SECONDS:
            return self.round_wait('长按F刷卡中：刷卡条目和 F 键消失', wait=0.1)
        self.input_worker.release_interact()
        self.card_hold_started = None
        self.card_hold_lost_since = None
        self.card_held = False
        self.card_hold_lost_count += 1
        if self.card_hold_lost_count > CARD_HOLD_LOST_LIMIT:
            self._reset_point_state()
            return self.round_recoverable_fail('刷卡长按中刷卡提示反复消失，门没刷开')
        self.current_kind = None
        if self.card_probe_step is None:
            self.card_probe_step = 0
        log.info(
            '贝果第 %s 点（房卡门）长按 %.1f 秒时刷卡提示消失，没刷上，松手回碎步移动左右试探'
            '（第 %s/%s 次，从第 %s 步起），位置 %s',
            self.cursor + 1, elapsed, self.card_hold_lost_count, CARD_HOLD_LOST_LIMIT,
            self.card_probe_step + 1, self.position,
        )
        return self.round_success('刷卡提示消失')

    def _verify_card_swipe(self, now: float) -> OperationRoundResult:
        """松手后确认刷卡结果。

        刷完后计数变成 0、F 键消失；F 键仍与刷卡条目同一行，说明这次长按没刷上
        （提前松手或长按被打断）。计数 OCR 常读不到，所以这里只看 F 键模板。
        F 键仍对准时先等提示消失，从松手起满 ``CARD_SWIPE_SETTLE_SECONDS`` 仍对准
        才判没刷上；没刷上时重按一次，重按后仍没刷上按失败局收尾。

        Args:
            now: 本轮的单调时钟读数。
        """
        released_at = self.card_verify_since if self.card_verify_since is not None else now
        card = self._card_entry()
        if card is not None:
            card_y, text = card
            keys = self._interact_key_rows()
            if any(abs(key_y - card_y) <= CARD_KEY_ROW_TOLERANCE for key_y in keys):
                if now - released_at < CARD_SWIPE_SETTLE_SECONDS:
                    return self.round_wait('松手后 F 键仍对准刷卡条目，等提示消失', wait=0.2)
                self.card_verify_since = None
                if self.card_retried:
                    self._reset_point_state()
                    return self.round_recoverable_fail('刷卡重按后 F 键仍对准刷卡条目，门没刷开')
                self.card_retried = True
                log.info('贝果刷卡松手 %.1f 秒后 F 键仍对准刷卡条目，判定没刷上，重按：%s',
                         now - released_at, text)
                return self.round_wait('刷卡没刷上，重按', wait=0.1)
        self.card_verify_since = None
        log.info('贝果刷卡完成：松手 %.1f 秒后%s', now - released_at,
                 '刷卡提示已消失' if card is None else f'F 键已消失，条目 {card[1]}')
        self.card_swiped = True
        self._reset_point_state()
        return self.round_success('已完成')

    def _start_card_probe(self, reason: str) -> OperationRoundResult:
        """房卡门到点没有刷卡提示，开始左右试探。

        Args:
            reason: 开始试探的原因，写进日志。
        """
        self._release_forward()
        log.info('贝果第 %s 点（房卡门）%s，开始左右试探刷卡提示', self.cursor + 1, reason)
        self.card_probe_step = 0
        return self._probe_card_reader()

    def _probe_card_reader(self) -> OperationRoundResult:
        """按 ``CARD_PROBE_SEQUENCE`` 侧移一步；提示由移动节点的提示检查接手。"""
        step = self.card_probe_step or 0
        if step >= len(CARD_PROBE_SEQUENCE):
            self._reset_point_state()
            return self.round_recoverable_fail('房卡门附近左右试探后仍没有刷卡提示')
        direction = CARD_PROBE_SEQUENCE[step]
        self.card_probe_step = step + 1
        self.input_worker.side_step(direction, CARD_PROBE_STEP_SECONDS)
        self._record_input()
        log.info(
            '贝果房卡门试探第 %s/%s 步：向%s侧移 %.1f 秒，位置 %s',
            step + 1, len(CARD_PROBE_SEQUENCE), '左' if direction < 0 else '右',
            CARD_PROBE_STEP_SECONDS, self.position,
        )
        return self.round_wait('房卡门左右试探刷卡提示', wait=0.3)

    def _card_prompt_lost(self, now: float) -> OperationRoundResult:
        """还没长按时刷卡条目消失：确认 ``CARD_PROMPT_LOST_SECONDS`` 后回碎步移动试探。

        试探进度沿用已有的，不从头开始；提示再出现时移动节点会转回交互，
        试探序列走完仍没有提示，由 ``_probe_card_reader`` 判失败。

        Args:
            now: 本轮的单调时钟读数。
        """
        self._release_forward()
        if self.card_lost_since is None:
            self.card_lost_since = now
        if now - self.card_lost_since < CARD_PROMPT_LOST_SECONDS:
            return self.round_wait('等待刷卡：刷卡提示消失', wait=0.1)
        self.card_lost_since = None
        self.select_since = None
        self.current_kind = None
        if self.card_probe_step is None:
            self.card_probe_step = 0
        log.info(
            '贝果第 %s 点（房卡门）刷卡提示出现后又消失，回到碎步移动左右试探（从第 %s 步起），位置 %s',
            self.cursor + 1, self.card_probe_step + 1, self.position,
        )
        return self.round_success('刷卡提示消失')

    def _card_entry(self) -> tuple[int, str] | None:
        """长按时外框可能被进度动画盖住，仍可单独读取钥匙图标与计数。"""
        frame = self._prompts()
        ys = frame.icons('card_key_icon')
        if not ys:
            return None
        return ys[0], frame.text_at(ys[0])

    @staticmethod
    def _card_count(text: str) -> int | None:
        """从刷卡条目文字里取持卡数；读不到时返回 None。

        Args:
            text: 条目文字，如「风蚀公路汽车旅馆二层1/1」。
        """
        found = CARD_COUNT_PATTERN.findall(text)
        if not found:
            return None
        return int(found[-1][0])

    def _interact_key_rows(self) -> list[int]:
        """提示区左侧所有 F 键的整屏纵坐标。F 键所在行就是按 F 生效的条目。"""
        area = self.ctx.screen_loader.get_area('贝果-局内', '交互F键')
        loaded = load_template('interact_key_f')
        if area is None or loaded is None:
            return []
        crop = cv2_utils.crop_image_only(self.last_screenshot, area.pc_rect)
        if crop is None or crop.size == 0:
            return []
        result = cv2_utils.match_template(
            crop, loaded[0], INTERACT_KEY_THRESHOLD,
            mask=loaded[1], only_best=False, ignore_inf=True,
        )
        return [area.pc_rect.y1 + item.center.y for item in result]

    def _prompts(self) -> PromptFrame:
        """同一轮的拾取、停步和选行共享截图与识别结果。"""
        if self.prompt_frame is None or self.prompt_frame.stamp != self.last_screenshot_time:
            self.prompt_frame = PromptFrame(
                self.ctx, self.last_screenshot, self.last_screenshot_time, on_cost=self.perf.add,
            )
        return self.prompt_frame

    def _scroll(self, down: bool) -> OperationRoundResult:
        """滚一格后废弃旧观测，只在新截图上决定下一次输入。"""
        self.input_worker.submit_turn(None)
        self.input_worker.scroll_prompt(down)
        self._record_input()
        self.selection.reset(self.last_input_frame or self.last_screenshot_time)
        return self.round_wait('已滚动一格，复核高亮条目', wait=0.2)

    def _prepare_interaction(self, kind: str) -> OperationRoundResult | None:
        """只有返回 None 才能按 F。超时后禁止继续滚动或补按。"""
        self._release_forward()
        self.input_worker.submit_turn(None)
        now = time.monotonic()
        if self.select_since is None:
            self.select_since = now
        if now - self.select_since >= SELECT_TIMEOUT:
            return self.round_recoverable_fail(f'无法稳定选中目标 {kind}')
        pickup = self._try_pickup()
        if pickup is not None:
            return pickup
        frame = self._prompts()
        if not self._ready(frame):
            return self.round_wait('等待选中行稳定', wait=0.1)
        step = frame.step_toward(kind)
        if step == 0:
            self.pickup_pending = False
            self.select_since = None
            return None
        if step is None:
            return self.round_wait('等待目标条目', wait=0.1)
        return self._scroll(step > 0)

    def _ready(self, frame: PromptFrame) -> bool:
        """在真实的新截图上确认选中行，输入通道空闲才接受本帧。"""
        return (self.selection.observe(frame) and self.input_worker.instant_idle
                and self._game_in_foreground())

    def _scan_more(self) -> OperationRoundResult | None:
        """三行视窗可能还有隐藏项。向下遍历，回到已见条目即结束。"""
        frame = self._prompts()
        row = frame.selected
        if len(frame.rows) < 3 or row is None:
            return None
        if not self._ready(frame):
            return self.round_wait('等待滚动后的列表稳定', wait=0.1)
        key = (row.kind, row.text, row.y // 10)
        if key in self.scanned_rows or len(self.scanned_rows) >= 16:
            return None
        self.scanned_rows.add(key)
        return self._scroll(True)

    def _should_tap(self, text: str) -> bool:
        """F 键行的拾取物这次能不能点按 F；捡不起来的记下并返回 False。

        Args:
            text: F 键行的条目文字。
        """
        if self.pickup_taps >= PICKUP_TAP_LIMIT:
            self._give_up_pickup(text, f'点按 F {PICKUP_TAP_LIMIT} 次仍在')
            self.pickup_text = None
            self.pickup_taps = 0
            return False
        return True

    def _tap_pickup(self, text: str) -> None:
        """点按 F 拾取 F 键行的拾取物，并记账。

        Args:
            text: F 键行的条目文字。
        """
        self.pickup_text = text
        self.pickup_pending = True
        self.pickup_taps += 1
        self.pickup_presses += 1
        self.pickup_tap_at = time.monotonic()
        self.input_worker.tap_interact(INTERACT_TAP_SECONDS)
        self._record_input()

    def _give_up_pickup(self, text: str, reason: str) -> None:
        """把一条拾取物记为捡不起来，本局之后不再捡它；累计够数就按背包已满处理。

        Args:
            text: 条目文字。
            reason: 写进日志的原因。
        """
        self.pickup_ignored.add(text)
        self.pickup_failures += 1
        log.warning('贝果拾取：「%s」%s，跳过（本局第 %s 条捡不起来）', text, reason, self.pickup_failures)
        if self.pickup_failures >= PICKUP_FAIL_LIMIT:
            log.warning('贝果拾取：累计 %s 条捡不起来，按背包已满处理，本局不再拾取', self.pickup_failures)

    def _recover_pickup_panel(self) -> OperationRoundResult | None:
        """拾取后若列表重排导致误开面板，原地关闭并回到当前节点。"""
        if not self.pickup_pending:
            return None
        if time.monotonic() - self.pickup_tap_at > 3:
            self.pickup_pending = False
            return None
        if self._prompts().center is not None:
            if time.monotonic() - self.pickup_tap_at >= PICKUP_TAP_WAIT:
                self.pickup_pending = False
            return None
        if self._has('搜查安全箱') and (self._has('搜查容器标题') or self._has('电子保险箱标题')):
            self._release_forward()
            self.input_worker.submit_turn(None)
            op = BagelCloseSearch(self.ctx)
            result = op.execute()
            self._adopt_screenshot(op)
            self.pickup_pending = False
            if not result.success:
                return self.round_recoverable_fail(f'关闭拾取时误开的面板失败：{result.status}')
            return self.round_wait('已关闭拾取时误开的面板', wait=0.1)
        if self._has('大保险解锁提示'):
            self._release_forward()
            self.input_worker.submit_turn(None)
            self.pickup_pending = False
            self.ctx.controller.btn_press('esc', press_time=0.1)
            self._record_input()
            return self.round_wait('已退出拾取时误开的解锁界面', wait=0.5)
        return None

    def _try_pickup(self, *, moving: bool = False, all_rows: bool = False) -> OperationRoundResult | None:
        """行进中只捡选中项；原地收尾可滚到其他拾取物。

        每张新截图都观察选中行，不设固定冷却：按过 F 后换成别的拾取物，两帧判稳就按。
        只有同名条目在 ``PICKUP_TAP_WAIT`` 内不重按。
        """
        frame = self._prompts()
        stable = self._ready(frame)
        row = frame.selected
        if self.pickup_text and stable and row is not None and (row.kind != 'pickup' or row.text != self.pickup_text):
            self.pickups += 1
            log.info('贝果拾取：「%s」已离开选中行，按已拾取计（本局第 %s 条）', self.pickup_text, self.pickups)
            if not moving and self.cleanup_since is not None:
                self.cleanup_since = time.monotonic()
            self.pickup_text = None
            self.pickup_taps = 0
        if self.pickup_failures >= PICKUP_FAIL_LIMIT:
            return None
        if row is not None and row.kind == 'pickup' and row.text not in self.pickup_ignored:
            if not stable:
                return None if moving else self.round_wait('等待拾取物高亮稳定', wait=0.1)
            remaining = PICKUP_TAP_WAIT - (time.monotonic() - self.pickup_tap_at)
            if row.text == self.pickup_text and remaining > 0:
                return None if moving else self.round_wait('等待拾取结果', wait=remaining)
            if not self._should_tap(row.text):
                return None if moving else self.round_wait('跳过无法拾取的条目', wait=0.1)
            since = time.monotonic() - self.pickup_tap_at
            self._tap_pickup(row.text)
            self.selection.reset(self.last_input_frame or self.last_screenshot_time)
            self.scanned_rows.clear()
            log.info(
                '贝果%s点按F拾取「%s」（第 %s 次按它%s）',
                '移动中' if moving else '原地', row.text, self.pickup_taps,
                f'，距上次拾取点按 {since:.2f}s' if since < 10 else '',
            )
            return None if moving else self.round_wait('已点按F拾取', wait=0.1)
        if all_rows and row is not None:
            target = next((r for r in frame.rows if r.kind == 'pickup' and r.text not in self.pickup_ignored), None)
            if target is not None:
                if not stable:
                    return self.round_wait('等待滚轮选中拾取物', wait=0.1)
                return self._scroll(target.y > row.y)
        return None

    def _log_leftover_prompts(self) -> None:
        """现场观察结束时记下提示列表；还剩拾取物就存一张截图，供排查漏捡。"""
        rows = self._prompts().rows
        if not rows:
            return
        log.info(
            '贝果拾取：现场观察结束时的提示列表：%s',
            '；'.join(f'{row.kind}「{row.text}」' if row.text else row.kind for row in rows),
        )
        left = [row.text for row in rows if row.kind == 'pickup']
        if left:
            path = save_screenshot(self.ctx, self.last_screenshot, 'BagelPickupLeft')
            log.info('贝果拾取：现场还剩 %s 条拾取物没捡，截图 %s', len(left), path)

    def _press_f(self, kind: str) -> OperationRoundResult:
        """普通容器和门点按 F。

        容器等搜查面板出现，门等提示持续消失。
        按 F 前先松前进键，避免按住前行漂移。
        """
        self._release_forward()
        if kind == 'box' and self._has('搜查容器标题'):
            # 容器面板已开：立刻交给收集入箱，不在这里判完成。
            # 用「提示消失」推断完成会漏掉收集和关闭面板，面板一直开着
            # 盖住小地图，后面整段导航都定位失败。
            self._reset_point_state()
            return self.round_success('面板已打开')
        if kind == 'door' and self.f_pressed and not any(row.kind == kind for row in self._prompts().rows):
            if self.door_missing_at is None:
                self.door_missing_at = self.last_screenshot_time
            if self.last_screenshot_time - self.door_missing_at < 0.4:
                return self.round_wait('确认门提示已消失', wait=0.1)
            self._reset_point_state()
            return self.round_success('已完成')
        self.door_missing_at = None
        if time.monotonic() - self.press_f_at < PRESS_F_RESULT_SECONDS:
            return self.round_wait('已点按F，等待结果', wait=PRESS_F_POLL_SECONDS)
        if self.press_f_taps >= PRESS_F_MAX_TAPS:
            log.info('贝果第 %s 点（%s）点按 F %s 次后提示仍在，判本局失败', self.cursor + 1, kind, self.press_f_taps)
            self._reset_point_state()
            return self.round_recoverable_fail('点按F后提示一直不消失')
        waiting = self._prepare_interaction(kind)
        if waiting is not None:
            return waiting
        self.input_worker.tap_interact(INTERACT_TAP_SECONDS)
        self.selection.reset(self.last_screenshot_time)
        self.f_pressed = True
        self.press_f_taps += 1
        self.press_f_at = time.monotonic()
        return self.round_wait('已点按F，等待结果', wait=PRESS_F_POLL_SECONDS)

    @property
    def needs_backpack_organize(self) -> bool:
        """是否存在尚未整理进安全箱的新拾取。"""
        return self.pickup_presses > self.organized_at_presses

    def _is_last_container(self) -> bool:
        """当前容器之后路线中是否不再有容器点（含路线末尾的现场交互容器）。"""
        point = self._current_point()
        if point is None:
            return True
        route = self.routes[self.route_index]
        return not any(item.kind in CONTAINER_KINDS for item in route.points[self.cursor + 1:])

    def _result_slots_empty(self) -> bool:
        """已定位的结果格是否全部为空；无坐标或非标准图时返回 False（不跳过）。"""
        screen = self.last_screenshot
        if not self._last_result_centers or screen is None or screen.shape != (1080, 1920, 3):
            return False
        return not any(slot_occupied(screen, center) for center in self._last_result_centers)

    @node_from(from_name='按类型交互', status='电子保险箱已解锁')
    @node_from(from_name='按类型交互', status='机械保险箱已打开')
    @node_from(from_name='按类型交互', status='面板已打开')
    @node_from(from_name='现场交互', status='电子保险箱已解锁')
    @node_from(from_name='现场交互', status='机械保险箱已打开')
    @node_from(from_name='现场交互', status='面板已打开')
    @operation_node(name='收集入箱')
    def collect(self) -> OperationRoundResult:
        """逐件装入安全箱。"""
        op = BagelStoreSafe(self.ctx, blueprint_priority=self.blueprint_priority, ledger=self.ledger)
        result = op.execute()
        self._adopt_screenshot(op)
        self._pick_all_sent_at = None
        if result.success:
            self._last_result_centers = op.result_centers
            return self.round_success('已入箱')
        self._last_result_centers = ()
        return self._fail_after_survey(f'入箱失败：{result.status}')

    @node_from(from_name='收集入箱', status='已入箱')
    @operation_node(name='拾取剩余')
    def pick_all_rest(self) -> OperationRoundResult:
        """用新截图确认搜查面板有剩余物品且可拾取，点按 F 并轮询结果格清空。"""
        now = time.monotonic()
        if self._pick_all_sent_at is not None:
            elapsed = now - self._pick_all_sent_at
            if elapsed >= PICK_ALL_MAX_WAIT:
                self._pick_all_sent_at = None
                return self.round_success('已点按全部拾取')
            if self.input_worker.instant_idle and self._result_slots_empty():
                self._pick_all_sent_at = None
                return self.round_success('全部拾取已清空')
            return self.round_wait('等待全部拾取生效', wait=PICK_ALL_POLL_SECONDS)
        if self._result_slots_empty():
            log.info('贝果拾取：结果区已清空，跳过「全部拾取」')
            return self.round_success('结果区已空，跳过')
        has_title = self._has('搜查容器标题') or self._has('电子保险箱标题')
        has_button = self._has('搜查完成')
        if not (has_title and has_button):
            return self.round_success('面板不可拾取，跳过', wait=0.2)
        self.input_worker.tap_interact(INTERACT_TAP_SECONDS)
        self.pickup_presses += 1
        self._pick_all_sent_at = now
        self._last_pick_all_at = now
        log.info('贝果拾取：入箱后点按 F「全部拾取」')
        return self.round_wait('等待全部拾取生效', wait=PICK_ALL_INITIAL_WAIT)

    @node_from(from_name='拾取剩余')
    @operation_node(name='容器内整理背包', screenshot_before_round=False)
    def organize_in_panel(self) -> OperationRoundResult:
        """最后一个容器内若有未整理拾取，直接在当前搜查面板左栏整理背包，省去关面板再按 B。"""
        if not self.organize_backpack or not self._is_last_container() or not self.needs_backpack_organize:
            return self.round_success('无需容器内整理')
        op = BagelOrganizeBackpack(
            self.ctx,
            blueprint_priority=self.blueprint_priority,
            ledger=self.ledger,
            in_search_panel=True,
            pickup_at=self._last_pick_all_at,
        )
        result = op.execute()
        self._adopt_screenshot(op)
        if result.status == self.STATUS_DEFEATED:
            return self.round_fail(self.STATUS_DEFEATED)
        if result.success:
            self.organized_at_presses = self.pickup_presses
            return self.round_success('容器内背包整理完成')
        log.warning('贝果容器内整理背包未完成：%s；留待退出前处理', result.status)
        return self.round_success('容器内整理跳过')

    @node_from(from_name='容器内整理背包')
    @operation_node(name='关闭面板')
    def close_panel(self) -> OperationRoundResult:
        """关闭搜查面板并确认回到局内。"""
        op = BagelCloseSearch(self.ctx)
        result = op.execute()
        self._adopt_screenshot(op)
        if result.success:
            self._read_survey_value()
            return self.round_success('已关闭面板')
        return self._fail_after_survey(f'关闭面板失败：{result.status}')

    def _fail_after_survey(self, reason: str) -> OperationRoundResult:
        """记下失败原因和失败画面，先补读调查收益，再按原因结束本局。

        Args:
            reason: 失败原因；补读结束后原样作为可结算重开的失败返回。
        """
        log.info('贝果%s；先补读调查收益，再结束本局', reason)
        self.fail_reason = reason
        if self.last_screenshot is not None:
            self.fail_screenshot = (self.last_screenshot_time, self.last_screenshot)
        return self.round_fail(self.STATUS_FAIL_SURVEY)

    @node_from(from_name='收集入箱', success=False, status=STATUS_FAIL_SURVEY)
    @node_from(from_name='关闭面板', success=False, status=STATUS_FAIL_SURVEY)
    @operation_node(name='失败前补读收益')
    def read_survey_before_fail(self) -> OperationRoundResult:
        """已回到局内时补读一次 HUD 调查收益，再按原失败原因结束本局。

        连续两帧读数相同才采用，避开涨幅浮字刚出现时的错读。
        以下情况不读，保留最后一次读数：
        - 已是 DEFEAT 画面；
        - 搜查面板仍开着：底栏数字左侧的金币图标会被 OCR 读成「9」（如「187K」读成「9187K」），
          错值每帧相同，连续两帧相同也挡不住；
        - 超过 ``FAIL_SURVEY_SECONDS`` 仍读不到稳定读数。
        """
        now = time.monotonic()
        if self.fail_survey_started is None:
            self.fail_survey_started = now
        if self.is_bagel_result():
            return self._finish_fail('已是失败结算画面')
        if self._has('搜查安全箱'):
            return self._finish_fail('搜查面板仍开着，不读底栏')
        value: int | None = None
        if self.round_by_find_area(self.last_screenshot, '战斗画面', '按键-普通攻击').is_success:
            value = self._read_hud_survey()
        if value is not None and value == self.fail_survey_last:
            log.info(
                '贝果失败前补读调查收益：HUD 读数 %s，上一次读数 %s',
                format_value(value), format_value(self.survey_value),
            )
            self.survey_value = value
            return self._finish_fail(None)
        self.fail_survey_last = value
        if now - self.fail_survey_started >= FAIL_SURVEY_SECONDS:
            return self._finish_fail(f'{FAIL_SURVEY_SECONDS:.0f} 秒内读数未停稳')
        return self.round_wait('等调查收益读数停稳', wait=FAIL_SURVEY_WAIT)

    def _finish_fail(self, skipped: str | None) -> OperationRoundResult:
        """恢复失败画面，按原失败原因结束本局。

        Args:
            skipped: 没有采用补读结果的原因；已采用时为 None。
        """
        if skipped is not None:
            log.info('贝果失败前补读调查收益：%s，沿用最后一次读数 %s', skipped, format_value(self.survey_value))
        if self.fail_screenshot is not None:
            self.last_screenshot_time, self.last_screenshot = self.fail_screenshot
        return self.round_recoverable_fail(self.fail_reason or self.STATUS_FAIL_SURVEY)

    @node_from(from_name='按类型交互', status='已完成')
    @node_from(from_name='现场交互', status='已完成')
    @node_from(from_name='关闭面板', status='已关闭面板')
    @operation_node(name='游标推进')
    def advance(self) -> OperationRoundResult:
        """原地捡完仍在范围内的物品，再进入下一点。"""
        recovered = self._recover_pickup_panel()
        if recovered is not None:
            return recovered
        self._release_forward()
        self.input_worker.submit_turn(None)
        now = time.monotonic()
        if self.cleanup_since is None:
            self.cleanup_since = now
        if now - self.cleanup_since < SELECT_TIMEOUT:
            pickup = self._try_pickup(all_rows=True) or self._scan_more()
            if pickup is not None:
                return pickup
            just_sent = self.last_input_frame is not None and self.last_screenshot_time - self.last_input_frame < 0.5
            if now - self.cleanup_since < 0.5 or just_sent:
                return self.round_wait('等待局内提示恢复', wait=0.1)
        self._enter_point(self.route_index, self.cursor + 1)
        return self.round_success('已推进')

    # ---------- 移动核心 ----------

    def _move_toward(self) -> OperationRoundResult:
        """朝当前游标点移动一轮。"""
        point = self._current_point()
        if point is None:
            return self.round_success('路线走完')
        timeout = self._check_timeout(point)
        if timeout is not None:
            return timeout
        fast = self._pickup_fast_round(point)
        if fast is not None:
            return fast
        hud = self._check_hud()
        if hud is not None:
            return hud
        recovered = self._recover_pickup_panel()
        if recovered is not None:
            return recovered
        position = self.locate()
        if position is None:
            # 定位丢失立刻停步：继续按着前进会走到不知道哪里去，
            # 恢复定位后再接着走。判失败要等满 10 秒，给降级搜索留时间。
            self._release_forward()
            if self.is_bagel_result():
                return self.round_fail(self.STATUS_DEFEATED)
            hud = self._check_hud(force=True)
            if hud is not None:
                return hud
            stray = self._handle_stray_unlock(point)
            if stray is not None:
                return stray
            if self.last_position is None:
                return self.round_retry('起步定位失败，重试', wait=0.3)
            if self.locate_miss_since is None:
                self.locate_miss_since = time.monotonic()
            if time.monotonic() - self.locate_miss_since < LOCATE_FAIL_SECONDS:
                self.locate_misses += 1
                return self.round_wait('定位丢失，停下等待', wait=0.1)
            return self.round_recoverable_fail(
                f'连续 {LOCATE_FAIL_SECONDS:.0f}s 定位失败，停止移动',
            )
        self.locate_misses = 0
        self.locate_miss_since = None
        self.last_position = position
        distance = hypot(point.x - position[0], point.y - position[1])
        if (
            point.kind == 'electronic_safe'
            and map_id_of(point.layer) == MAP_JANUS
            and distance <= ELECTRONIC_SMALL_STEP_RADIUS
            and not self.electronic_small_steps
        ):
            self.electronic_small_steps = True
            self._release_forward()
            log.info(
                '贝果电子保险箱进入短步阶段：距离 %.1fpx，半径 %.0fpx，按压 %.2fs',
                distance, ELECTRONIC_SMALL_STEP_RADIUS, SMALL_STEP_PRESS,
            )
        if self.last_input_frame is not None and self.last_screenshot_time <= self.last_input_frame:
            return self.round_wait('等待动作后的新截图', wait=0.05)
        # 边走边捡：只投递点按，不停车也不结束本轮，到点判定照常做。
        if self.opening_blocking_door:
            door = self._open_blocking_door()
            if door is not None:
                return door
        self._try_pickup(moving=True)
        if self.retreat_point is not None:
            return self._retreat(position)
        if distance <= SPRINT_DISTANCE and self.sprinting:
            self._exit_sprint()
        if point.kind == BEACON:
            # 途径点是纯路点：3px 内算到点，10px 内慢步逼近。
            if distance <= ARRIVE_RADIUS:
                self._release_forward()
                log.info(
                    '贝果到点：第 %s 点判定距离 %.2fpx（判据 %.0fpx，位置 %s）',
                    self.cursor + 1, distance, ARRIVE_RADIUS, position,
                )
                self._enter_point(self.route_index, self.cursor + 1)
                # 状态文字必须正好是「已到点」：节点边按状态精确匹配，
                # 掺了距离就匹配不上，导航链会在这里断掉、整条路线直接判走完。
                return self.round_success('已到点')
            if distance <= BEACON_SLOW_RADIUS:
                return self._slow_step(point, position)
            return self._cruise(point, position)
        # 雅努斯电子保险箱进入 15px 后保持短步；其他箱类仍在 3px 内使用碎步。
        # 可信提示出现就转入交互，不要求先走到原地等待半径。
        prompt = self._prompt_stop(distance)
        if prompt is not None:
            return prompt
        if point.kind == KEYCARD_DOOR and self.card_probe_step is not None:
            # 试探中不再按坐标逼近，否则会被拉回门板正中，试探白做。
            return self._probe_card_reader()
        if distance < BOX_WAIT_RADIUS:
            self.small_step_since = None
            return self._wait_here(point)
        self.target_wait_since = None
        if distance <= BOX_SMALL_STEP_RADIUS:
            timeout = self._check_small_step_timeout()
            if timeout is not None:
                return timeout
            return self._short_step()
        self.small_step_since = None
        return self._slow_step(point, position, short_press=self.electronic_small_steps)

    def _pickup_fast_round(self, point: BagelRoutePoint) -> OperationRoundResult | None:
        """拾取物在选中行时只走快轮：截图、找高亮、读选中行、判稳后点按 F。

        完整一轮还要做定位、朝向，40px 内还要读整个提示列表；两帧判稳要两轮，
        边走边捡时下一件常在判稳前就出了范围。快轮跳过这些步骤，前进键和转向线程
        按上一轮完整判断的意图继续，不停步。

        以下情况不走快轮，返回 None 交给完整一轮：
        - 正在退回、开挡路的门或试探刷卡；拾取已按背包满停用；
        - 按住前进走向途径点、且距离不超过 ``PICKUP_FAST_BEACON_DISTANCE``；
        - 选中行不是可捡的拾取物，且不在点按后的过渡期内；
        - 本段快轮已满 ``PICKUP_FAST_SECONDS``。

        Args:
            point: 当前游标点。

        Returns:
            快轮的等待结果；要做完整一轮时返回 None。
        """
        now = time.monotonic()
        if self.pickup_fast_since is not None and now - self.pickup_fast_since >= PICKUP_FAST_SECONDS:
            self._end_pickup_fast('满时长')
            return None
        if not self._pickup_fast_allowed(point):
            self._end_pickup_fast('条件不满足')
            return None
        presses = self.pickup_presses
        self._try_pickup(moving=True)
        row = self._prompts().selected
        if row is not None:
            in_play = row.kind == 'pickup' and row.text not in self.pickup_ignored
        else:
            in_play = now - self.pickup_tap_at < PICKUP_FAST_AFTER_TAP
        if not in_play:
            self._end_pickup_fast('选中行不是拾取物')
            return None
        if self.pickup_fast_since is None:
            self.pickup_fast_since = now
            self.pickup_fast_rounds = 0
            self.pickup_fast_presses = self.pickup_presses
        self.pickup_fast_rounds += 1
        if self.pickup_presses > presses:
            status = '快轮已点按F拾取'
        elif row is None:
            status = '快轮等列表重排'
        else:
            status = '快轮等拾取物判稳'
        return self.round_wait(status, wait=0)

    def _pickup_fast_allowed(self, point: BagelRoutePoint) -> bool:
        """当前状态能不能走拾取快轮，见 ``_pickup_fast_round``。"""
        if (self.retreat_point is not None or self.opening_blocking_door
                or self.card_probe_step is not None or self.last_position is None
                or self.pickup_failures >= PICKUP_FAIL_LIMIT):
            return False
        if point.kind != BEACON:
            return True
        distance = hypot(point.x - self.last_position[0], point.y - self.last_position[1])
        holding = self.input_worker.forward_held_seconds() is not None and not self.input_worker.forward_step_active()
        return not (holding and distance <= PICKUP_FAST_BEACON_DISTANCE)

    def _end_pickup_fast(self, reason: str) -> None:
        """结束本段快轮并打一条汇总日志；没在走快轮时什么也不做。

        Args:
            reason: 结束原因，写进日志。
        """
        if self.pickup_fast_since is None:
            return
        log.info(
            '贝果拾取快轮结束（%s）：%s 轮 %.2fs，点按 %s 次',
            reason, self.pickup_fast_rounds, time.monotonic() - self.pickup_fast_since,
            self.pickup_presses - self.pickup_fast_presses,
        )
        self.pickup_fast_since = None

    def _handle_stray_unlock(self, point: BagelRoutePoint) -> OperationRoundResult | None:
        """移动中出现电子保险箱解锁界面时接手；没有该界面返回 None。

        F 键在条目间换位时，按 F 可能误开旁边的电子保险箱。解锁界面盖住小地图，
        不处理就会连续定位失败。目标点就是电子保险箱时直接转入交互，解锁操作
        认得已打开的界面；否则按 esc 退出，同一个点最多 ``STRAY_UNLOCK_ESC_LIMIT`` 次。

        Args:
            point: 当前游标点。
        """
        if not self._has('大保险解锁提示'):
            return None
        self.locate_miss_since = None
        if point.kind == 'electronic_safe':
            log.info('贝果第 %s 点（电子保险箱）移动中已进入解锁界面，直接接着解锁', self.cursor + 1)
            self.current_kind = 'electronic_safe'
            self.target_wait_since = None
            return self.round_success('提示已出现')
        if self.stray_unlock_escapes >= STRAY_UNLOCK_ESC_LIMIT:
            return None
        self.stray_unlock_escapes += 1
        log.info(
            '贝果第 %s 点（%s）移动中误开了电子保险箱解锁界面，按 esc 退出（第 %s 次）',
            self.cursor + 1, point.kind, self.stray_unlock_escapes,
        )
        self.ctx.controller.btn_press('esc', press_time=0.1)
        self._record_input()
        return self.round_wait('误开解锁界面，按 esc 退出', wait=0.5)

    def _wait_here(self, point: BagelRoutePoint) -> OperationRoundResult:
        """已站在箱子上：原地不动，等提示出现或超时跳到下一个点（房卡门判失败）。

        箱子一定会刷出交互提示，所以到时间还没提示，只可能是这个点
        本身没有东西（或站位被挡住），直接跳过，不判失败、也不死等。

        Args:
            point: 当前游标点。
        """
        self._release_forward()
        self.input_worker.submit_turn(None)
        if self.target_wait_since is None:
            self.target_wait_since = time.monotonic()
            return self.round_wait('原地等待交互提示', wait=0.2)
        elapsed = time.monotonic() - self.target_wait_since
        if elapsed < SELECT_TIMEOUT:
            scan = self._scan_more()
            if scan is not None:
                return scan
        if elapsed >= BOX_PROMPT_SECONDS:
            self.target_wait_since = None
            if point.kind == KEYCARD_DOOR:
                # 房卡门不开，后面的点都在门里，不能跳过；改为左右试探找刷卡提示。
                return self._start_card_probe('到点等 1 秒没有刷卡提示')
            log.info(
                '贝果第 %s 点（%s）原地等 %ss 无提示，跳到下一个点',
                self.cursor + 1, KIND_LABELS.get(point.kind, point.kind), f'{BOX_PROMPT_SECONDS:.0f}',
            )
            self._enter_point(self.route_index, self.cursor + 1)
            return self.round_success('跳过此点')
        return self.round_wait('原地等待交互提示', wait=0.2)

    def _prompt_stop(self, distance: float) -> OperationRoundResult | None:
        """可信提示出现后松键并转入交互，不另做停稳复核。

        Args:
            distance: 当前可靠定位到目标点的距离。

        Returns:
            允许交互的结果；未识别到可信提示时返回 None。
        """
        if distance > INTERACTION_TRUST_DISTANCE:
            return None
        kind = self._nearby_kind()
        if kind is None:
            return None
        self._release_forward()
        self.input_worker.submit_turn(None)
        self.selection.reset(self.last_screenshot_time)
        self.current_kind = kind
        self.select_since = None
        self.target_wait_since = None
        return self.round_success('提示已出现')

    def _angle_miss_result(self, angle: float | None) -> OperationRoundResult | None:
        """读不到镜头朝向时停下等；连续读不到满 10 秒才判失败。

        单帧读不到是常事（转视角、贴墙、扇形被墙切掉），一帧失败就中断
        整条路线太脆。读不到时转向线程也转不了（没有当前角算偏差），
        这里只松键等画面回来，不下发任何转向。

        Args:
            angle: 本轮读到的镜头朝向；读到就不做处理。

        Returns:
            需要等待或失败时返回本轮结果；读到朝向后返回 None。
        """
        if angle is not None:
            self.angle_miss_since = None
            return None
        self._release_forward()
        if self.angle_miss_since is None:
            self.angle_miss_since = time.monotonic()
        missed = time.monotonic() - self.angle_miss_since
        if missed >= ANGLE_FAIL_SECONDS:
            return self.round_recoverable_fail(
                f'连续 {ANGLE_FAIL_SECONDS:.0f}s 识别不到镜头朝向，停止移动',
            )
        self.input_worker.submit_turn(None)
        return self.round_wait(f'未识别到朝向，停下等待（{missed:.1f}s）', wait=0.1)

    def _cruise(
        self, point: BagelRoutePoint, position: tuple[float, float],
    ) -> OperationRoundResult:
        """远距离移动：提交目标与本帧朝向，前进键由主线管理。

        转向全由转向线程负责（稳 3 帧、2 度以上才转）。这里按滞回束角管前进：
        行进中偏差超过 16 度松键停车等转，停车后转回 10 度以内恢复前进；
        同时走和大角度转会互相干扰，表现为原地乱转。超过 16 度要连续两张不同截图确认，
        第一张照常边走边修，单帧朝向跳变不会触发停车。

        近距离不再按角度停车转：这个距离上的目标方位由定位抖动决定，
        停下车来按它转只会原地打摆，改成边走边转（线程内限幅 CRUISE_TURN_CAP）。
        """
        angle = self.player_angle()
        miss = self._angle_miss_result(angle)
        if miss is not None:
            return miss
        target_angle = self._target_angle(position, point)
        controller_angle = angle % 360
        angle_diff = cal_utils.angle_delta(controller_angle, target_angle)
        distance = hypot(point.x - position[0], point.y - position[1])
        near = distance <= NEAR_STEP_TURN_RADIUS
        self._diag_cruise(point, position, angle_diff, distance)
        stop_limit = WALK_TURN_ANGLE if self.turning_in_place else WALK_STOP_TURN_ANGLE
        over = abs(angle_diff) > stop_limit and not near
        if not over:
            self.cruise_over_frame = None
            stop_turn = False
        elif self.turning_in_place:
            stop_turn = True
        else:
            # 行进中首次超阈值：记下这张截图，同一张截图的后续轮次不算确认。
            if self.cruise_over_frame is None:
                self.cruise_over_frame = self.last_screenshot_time
            stop_turn = self.last_screenshot_time > self.cruise_over_frame
        self._submit_turn(target_angle, angle, cruise_only=not stop_turn, deadband=TURN_DEADBAND_FAST)
        if stop_turn:
            # 大偏差：松键停车等转向线程转到位再走。
            self.turning_in_place = True
            self.cruise_over_frame = None
            self._release_forward()
            return self.round_wait(f'停车等转向 {angle_diff:.0f}度')
        self.turning_in_place = False
        stuck = self._check_stuck(position, point, controller_angle)
        if stuck is not None:
            return stuck
        if self.input_worker.hold_forward():
            self._record_input()
        self._set_sprint(distance)
        return self.round_wait('冲刺前进' if self.sprinting else '按住前进')

    def _diag_cruise(
        self, point: BagelRoutePoint, position: tuple[float, float],
        angle_diff: float, distance: float,
    ) -> None:
        """巡航诊断一行：点位、距离、镜头差与横向偏，供离线算走偏分布。

        字段顺序由 ``bagel/diag.py`` 的格式契约固定，新增字段只能往后追加。
        """
        off = self._off_track(point, position)
        diag({
            '点': self.cursor + 1,
            '距': f'{distance:.1f}',
            '差': f'{angle_diff:+.1f}',
            '偏': '?' if off is None else f'{off:.1f}',
            '位置': f'({position[0]:.0f},{position[1]:.0f})',
        })

    def _off_track(self, point: BagelRoutePoint, position: tuple[float, float]) -> float | None:
        """当前位置到「上一个点→当前点」这条腿的垂直距离；没有上一个点时为 None。"""
        origin = self._previous_point()
        if origin is None:
            return None
        foot = _closest_on_segment(origin, point, position)
        return hypot(position[0] - foot[0], position[1] - foot[1])

    def _submit_turn(
        self, target: float, angle: float, cruise_only: bool = False, deadband: float = TURN_DEADBAND,
    ) -> None:
        """提交目标方向给转向线程；只存不定夺，不记输入。

        转向线程使用本帧朝向判稳，主线没有新样本时才自行补采。
        这里不调 _record_input：提交只是写内存，不是输入，
        记了会逼主线每轮多等一张新截图，把主循环限速。

        Args:
            target: 目标方向（度）。
            cruise_only: 近距离只做边走边转，线程内限幅 15 度。
            deadband: 本次目标的对准死区；快速移动传 2 度，其余用默认 5 度。
        """
        if self._capture_start < self._turn_resume_at:
            return
        sample = None
        if self.last_screenshot is not None and self.last_screenshot is self._captured_frame:
            self.capture_stats.main_sample(self._capture_start)
            sample = HeadingSample(self._capture_start, angle)
        self.input_worker.submit_turn(target, cruise_only=cruise_only, deadband=deadband, sample=sample)

    def remember_turn_ratio(self) -> None:
        """把本局学到的转向校准值记进跨局记忆，作下一局的起步值；不写 turn_dx。"""
        if self.turn_memory is None:
            return
        ratio = self.input_worker.turn_ratio
        samples = self.input_worker.turn_samples
        log.info('贝果转向本局校准值 %.3f（%s 个有效样本）', ratio, samples)
        self.turn_memory.absorb(ratio, samples, self._turn_dx())

    def _turn_dx(self) -> float:
        """转向实际使用的 turn_dx：读控制器持有的游戏配置。

        后端运行前会重建 ctx.game_config，控制器仍持有启动时的对象，两者的值可能不同；
        转向按控制器那份换算，日志和提示也以它为准。
        """
        return float(self.ctx.controller.game_config.turn_dx)

    def _set_sprint(self, distance: float) -> None:
        """按距离决定是否冲刺。

        绝区零里冲刺是「按住方向键 + 点闪避」，冲刺时按 SPRINT_DODGE_INTERVAL
        的间隔补点闪避，不需要按住闪避键。

        Args:
            distance: 距当前目标点的距离。
        """
        want = distance > SPRINT_DISTANCE
        if want:
            self.sprinting = True
            self._dodge_tap()
        elif self.sprinting:
            self._exit_sprint()

    def _dodge_tap(self) -> None:
        """点一下闪避维持冲刺。

        W 实际按住不足 SPRINT_START_DELAY，或距上次点不足
        SPRINT_DODGE_INTERVAL 时跳过。
        """
        held = self.input_worker.forward_held_seconds()
        if held is None or held < SPRINT_START_DELAY:
            return
        now = time.monotonic()
        if self.last_dodge_at is not None and now - self.last_dodge_at < SPRINT_DODGE_INTERVAL:
            return
        self.last_dodge_at = now
        self.input_worker.dodge()

    def _exit_sprint(self) -> None:
        """退出冲刺：松开方向键。"""
        self.sprinting = False
        self.input_worker.stop_forward()

    def _stop_sprint(self) -> None:
        """立刻松键。"""
        self.sprinting = False
        self.input_worker.stop_forward()

    def _prepare_forward_step(self) -> OperationRoundResult | None:
        """决策帧到了才允许发起下一步：上一短按已结束，且本轮截图拍于松键加停顿之后。

        停顿下限 ``STEP_REST_SECONDS`` 是节拍的唯一来源，不随主线每轮耗时变化。
        停顿期间每轮照常截图、定位、读提示，到点和可信提示不受停顿约束；
        轮末最多睡 ``STEP_REST_POLL``，停顿一结束就拍决策帧。
        """
        if self.input_worker.forward_step_active():
            return self.round_wait('等待本次前进短按结束', wait=0.02)
        self._release_forward()
        released_at = self.input_worker.forward_released_at
        if released_at is None:
            return self.round_wait('等待松键', wait=0.02)
        ready_at = released_at + STEP_REST_SECONDS
        if self.last_screenshot_time <= ready_at:
            remaining = max(0.0, ready_at - time.time())
            return self.round_wait('前进停顿中', wait=min(remaining, STEP_REST_POLL))
        return None

    def _short_step(self) -> OperationRoundResult:
        """近距离移动：不停下来对准，直接短按前进。

        3px 内不再转向：这么近的距离，目标方向完全由 1px 级的坐标抖动
        决定，算出来的角度没有意义。一旦按角度判停，就会出现「每轮都在
        转向、一步也不走」的原地打摆。

        方向真的偏了也不怕：走一步后距离变大，自然回到慢步分支去对准。
        超时由调用方 _check_small_step_timeout 管，这里只负责迈步。
        """
        ready = self._prepare_forward_step()
        if ready is not None:
            return ready
        limited = self._fail_if_action_limit()
        if limited is not None:
            return limited
        if not self._step_forward(SMALL_STEP_PRESS):
            return self.round_wait('等待前进短按可用', wait=0.02)
        self._record_input()
        return self.round_wait('短步前进')

    def _check_small_step_timeout(self) -> OperationRoundResult | None:
        """碎步段超时检查：3px 内迈步超过 5 秒还没贴到就跳过这个点（房卡门判失败）。

        碎步不看角度硬迈，方向偏了会一直原地蹭，靠单点 75 秒兜底太慢，
        所以这一小段单独计时。超时说明这个点贴不上（站位被挡、坐标偏），
        直接推进到下一个点，不判失败、也不死等。

        Returns:
            跳过结果；未超时返回 None。
        """
        if self.small_step_since is None:
            self.small_step_since = time.monotonic()
            return None
        if time.monotonic() - self.small_step_since < SMALL_STEP_TIMEOUT:
            return None
        self.small_step_since = None
        self._release_forward()
        point = self._current_point()
        if point is not None and point.kind == KEYCARD_DOOR:
            return self._start_card_probe('碎步贴不上')
        log.info(
            '贝果第 %s 点（%s）碎步 %ss 没贴上，跳到下一个点',
            self.cursor + 1, self._current_kind_name(), f'{SMALL_STEP_TIMEOUT:.0f}',
        )
        self._enter_point(self.route_index, self.cursor + 1)
        return self.round_success('跳过此点')

    def _current_kind_name(self) -> str:
        """当前游标点的类型名，只用于日志。"""
        point = self._current_point()
        return KIND_LABELS.get(point.kind, point.kind) if point is not None else '无'

    def _slow_step(
        self, point: BagelRoutePoint, position: tuple[float, float],
        short_press: bool = False,
    ) -> OperationRoundResult:
        """按目标方向步进，电子箱短步阶段使用更短的按压时间。

        提交目标方向和本帧朝向；偏差超过 10 度时停车等转，
        其余情况短按前进。每步结束后使用新截图重新判断，
        避免多次短按相互覆盖；短按按住中仅在小偏差内微调，松键停顿期按正常逻辑提交转向。

        近距离不再按角度停车转：这个距离上的目标方位由定位抖动决定，
        改成边走边转（线程内限幅 CRUISE_TURN_CAP）；偏差超过 ``NEAR_STOP_TURN_ANGLE`` 时仍停车转到位。

        Args:
            point: 当前目标点。
            position: 当前可靠定位。
            short_press: 是否使用 ``SMALL_STEP_PRESS`` 短步；否则使用 0.5 秒慢步。
        """
        distance = hypot(point.x - position[0], point.y - position[1])
        target_angle = self._target_angle(position, point)
        ready = self._prepare_forward_step()
        if ready is not None:
            if distance > BOX_SMALL_STEP_RADIUS:
                self._submit_slow_step_turn(distance, target_angle)
            return ready
        action = '短步' if short_press else '慢步'
        angle = self.player_angle()
        miss = self._angle_miss_result(angle)
        if miss is not None:
            return miss
        angle_diff = cal_utils.angle_delta(angle % 360, target_angle)
        near = distance <= NEAR_STEP_TURN_RADIUS
        self.cruise_over_frame = None
        stop_turn = abs(angle_diff) > (NEAR_STOP_TURN_ANGLE if near else WALK_TURN_ANGLE)
        self._submit_turn(target_angle, angle, cruise_only=near and not stop_turn)
        if stop_turn:
            # 大偏差：松键停车等转向线程转到位，不短按前进。
            self._release_forward()
            return self.round_wait(f'{action}停车等转向 {angle_diff:.0f}度')
        self.turning_in_place = False
        blocked = self._check_slow_stuck(position)
        if blocked is not None:
            return blocked
        limited = self._fail_if_action_limit()
        if limited is not None:
            return limited
        duration = SMALL_STEP_PRESS if short_press else SLOW_STEP_PRESS
        if not self._step_forward(duration):
            return self.round_wait('等待前进短按可用', wait=0.02)
        self._record_input()
        wait = None if short_press else SLOW_STEP_WAIT
        return self.round_wait(f'{action}前进', wait=wait)

    def _submit_slow_step_turn(self, distance: float, target_angle: float) -> None:
        """慢步等待期按距离与前进键状态分流提交转向，避免临近到点抢发微调占坑。

        距目标 ``NEAR_STEP_TURN_RADIUS``（8px）内时，角色可能一步就到点换向且定位抖动会放大偏差：
        W 仍在短按中不发微调；已松键停顿期也只在偏差超过 ``NEAR_STOP_TURN_ANGLE`` 时提前全速转。
        8px 外时，短按中仅在 ``WALK_STOP_TURN_ANGLE`` 内微调，停顿期按正式慢步规则提交。

        Args:
            distance: 距当前目标点的距离（px）。
            target_angle: 目标方向（度）。
        """
        angle = self.player_angle()
        if angle is None:
            return
        diff = abs(cal_utils.angle_delta(angle % 360, target_angle))
        near = distance <= NEAR_STEP_TURN_RADIUS
        if (
            self.input_worker.forward_step_active()
            or self.input_worker.forward_released_at is None
        ):
            if not near and diff <= WALK_STOP_TURN_ANGLE:
                self._submit_turn(target_angle, angle, cruise_only=True)
            return
        stop_turn = diff > (NEAR_STOP_TURN_ANGLE if near else WALK_TURN_ANGLE)
        if near and not stop_turn:
            return
        self._submit_turn(target_angle, angle, cruise_only=False)

    @staticmethod
    def _target_angle(
        position: tuple[float, float], point: BagelRoutePoint,
    ) -> float:
        """当前位置指向目标点的角度。

        与朝向识别（analyze）同一套坐标系：屏幕上方为正、逆时针增加，
        所以纵向取负号；符号写反会让角色一直原地打转。
        """
        return degrees(atan2(-(point.y - position[1]), point.x - position[0])) % 360

    def _check_stuck(
        self, position: tuple[float, float], point: BagelRoutePoint, angle: float,
    ) -> OperationRoundResult | None:
        """按住前进却走不动时，往路径线那一侧侧移一小步。

        走不动一般不是被挡住，而是已经偏到路径线外，正顶着墙或箱子。
        这时继续判失败没意义。拿「上一个点 → 当前目标点」这条连线当路径线，
        朝线上的最近点挪一小步，让前进方向重新对上线。
        """
        if self.stuck_since is None:
            self.stuck_since = (position, time.monotonic())
            return None
        origin, started = self.stuck_since
        if hypot(position[0] - origin[0], position[1] - origin[1]) >= STUCK_DISTANCE:
            self.stuck_since = (position, time.monotonic())
            self.stuck_steps = 0
            return None
        if time.monotonic() - started < STUCK_SECONDS:
            return None
        door = self._open_blocking_door()
        if door is not None:
            self.stuck_since = None
            return door
        if self.stuck_steps >= STUCK_STEP_LIMIT:
            return self._blocked('持续前进但位置未变化，侧移多次仍无法脱离')
        direction = self._path_side(position, point, angle)
        on_line = direction is None and self._previous_point() is not None
        if on_line:
            direction = STUCK_ON_LINE_SEQUENCE[self.stuck_steps % len(STUCK_ON_LINE_SEQUENCE)]
        self.stuck_steps += 1
        self.stuck_since = None
        if direction is None:
            # 路线第一个点没有路径线，判断不出该往哪挪，不瞎动。
            return self.round_wait('前进受阻，等待重新判定路径线', wait=0.2)
        self.input_worker.side_step(direction, SIDE_STEP_SECONDS)
        self._record_input()
        side = '右' if direction > 0 else '左'
        what = f'已在路径线上，试着往{side}侧' if on_line else f'往路径线{side}侧'
        return self.round_wait(f'前进受阻，{what}移动一步', wait=SIDE_STEP_SECONDS + 0.1)

    def _check_slow_stuck(self, position: tuple[float, float]) -> OperationRoundResult | None:
        """慢步或短步逼近时，位置停滞满 ``SLOW_STUCK_SECONDS`` 就按受阻处理。

        慢步只在离点 10px 内用，按住前进的侧移判卡位管不到这里；
        不单独判的话，顶着扶手原地踏步要等单点超时才停。
        """
        now = time.monotonic()
        if self.slow_stuck_since is not None:
            origin, started = self.slow_stuck_since
            if hypot(position[0] - origin[0], position[1] - origin[1]) < STUCK_DISTANCE:
                if now - started < SLOW_STUCK_SECONDS:
                    return None
                door = self._open_blocking_door()
                if door is not None:
                    self.slow_stuck_since = None
                    return door
                return self._blocked('慢步前进但位置未变化')
        self.slow_stuck_since = (position, now)
        return None

    def _open_blocking_door(self) -> OperationRoundResult | None:
        """受阻时选中门再点按；门消失或用完次数后恢复导航。"""
        if self.blocked_door_taps >= BLOCKED_DOOR_TAP_LIMIT or not any(
            row.kind == 'door' for row in self._prompts().rows
        ):
            self.opening_blocking_door = False
            self.select_since = None
            return None
        self.opening_blocking_door = True
        waiting = self._prepare_interaction('door')
        if waiting is not None:
            return waiting
        self.opening_blocking_door = False
        self.blocked_door_taps += 1
        log.info(
            '贝果第 %s 点前进受阻，前方是「%s」，点按 F 开门（第 %s/%s 次），位置 %s',
            self.cursor + 1, self._prompts().selected.text, self.blocked_door_taps, BLOCKED_DOOR_TAP_LIMIT, self.position,
        )
        self.input_worker.tap_interact(INTERACT_TAP_SECONDS)
        self._record_input()
        self.selection.reset(self.last_input_frame or self.last_screenshot_time)
        return self.round_wait('前进受阻，点按 F 开门', wait=BLOCKED_DOOR_WAIT_SECONDS)

    def _blocked(self, reason: str) -> OperationRoundResult:
        """当前点走不通：先退回上一个点再重走，次数用完或退回途中受阻才判失败。

        受阻常见于拐角：途径点 3px 内就判到点，角色还没绕过扶手或家具就转向
        下一个点，直线顶上去。退回上一个点站准了再走，走的就是录制时的线。
        上一个点是电梯时不退：电梯跨层，走不回去。
        """
        self._release_forward()
        point = self._current_point()
        origin = self._previous_point()
        if (
            self.retreat_point is None and origin is not None and origin.kind != ELEVATOR
            and self.retreats < RETREAT_LIMIT
        ):
            self.retreats += 1
            self.retreat_point = origin
            self.retreat_since = time.monotonic()
            self.point_started_at = self.retreat_since
            self._clear_stuck()
            log.warning(
                '贝果第 %s 点（%s）%s，位置 %s，退回上一个点 (%.0f,%.0f) 重走（第 %s/%s 次）',
                self.cursor + 1, point.kind if point is not None else '无', reason,
                self.last_position, origin.x, origin.y, self.retreats, RETREAT_LIMIT,
            )
            return self.round_wait('受阻，退回上一个点', wait=0.1)
        if self.retreat_point is not None:
            reason = f'退回上一个点途中{reason}'
        elif self.retreats:
            reason = f'{reason}（已退回重走 {self.retreats} 次）'
        return self.round_recoverable_fail(reason)

    def _retreat(self, position: tuple[float, float]) -> OperationRoundResult:
        """朝上一个点的坐标走，站准（或到时限）后回到当前点重走。"""
        target = self.retreat_point
        distance = hypot(target.x - position[0], target.y - position[1])
        overdue = self.retreat_since is not None and time.monotonic() - self.retreat_since >= RETREAT_SECONDS
        if distance < RETREAT_ARRIVE_RADIUS or overdue:
            self._release_forward()
            log.info(
                '贝果第 %s 点：%s上一个点 (%.0f,%.0f)，距 %.1fpx，位置 %s，重走本点',
                self.cursor + 1, '已退回' if not overdue else f'{RETREAT_SECONDS:.0f}s 内没站准',
                target.x, target.y, distance, position,
            )
            self.retreat_point = None
            self.retreat_since = None
            self.point_started_at = time.monotonic()
            self.steps = 0
            self.electronic_small_steps = False
            self.small_step_since = None
            self.target_wait_since = None
            self._clear_stuck()
            return self.round_wait('已退回上一个点，重走本点', wait=0.1)
        if distance <= SPRINT_DISTANCE and self.sprinting:
            self._exit_sprint()
        if distance <= BEACON_SLOW_RADIUS:
            return self._slow_step(target, position, short_press=distance <= ARRIVE_RADIUS)
        return self._cruise(target, position)

    def _clear_stuck(self) -> None:
        self.stuck_since = None
        self.stuck_steps = 0
        self.slow_stuck_since = None

    def _path_side(
        self, position: tuple[float, float], point: BagelRoutePoint, angle: float,
    ) -> int | None:
        """该往哪一侧挪：1 按 D（角色右手边），-1 按 A（角色左手边）。

        A/D 是相对角色朝向的左右，不是地图上的左右，所以要先把「指向路径线」
        的方向换算到角色局部坐标：角色朝向角 angle 下，右手方向是
        `(sin angle, cos angle)`（地图 y 轴向下），与它点乘为正就是往右挪。

        路径线取上一个点到当前目标点的连线；角色在线的哪一侧都不必先判，
        直接朝线上最近点走，偏多偏少都成立。
        """
        origin = self._previous_point()
        if origin is None:
            return None
        foot = _closest_on_segment(origin, point, position)
        to_line = (foot[0] - position[0], foot[1] - position[1])
        if hypot(to_line[0], to_line[1]) < 0.5:
            return None
        right_x, right_y = sin(radians(angle)), cos(radians(angle))
        return 1 if to_line[0] * right_x + to_line[1] * right_y >= 0 else -1

    def _previous_point(self) -> BagelRoutePoint | None:
        """上一个点；路线起点没有上一个点时返回 None。"""
        if not self.routes:
            return None
        route = self.routes[self.route_index]
        if self.cursor <= 0:
            return None
        return route.points[self.cursor - 1]

    def _check_timeout(self, point: BagelRoutePoint) -> OperationRoundResult | None:
        """单点超时检查。"""
        if self.point_started_at is None:
            self.point_started_at = time.monotonic()
            return None
        limit = POINT_TIMEOUT.get(point.kind, POINT_TIMEOUT['default'])
        if time.monotonic() - self.point_started_at >= limit:
            self._release_forward()
            return self.round_recoverable_fail(
                f'第 {self.cursor + 1} 点（{KIND_LABELS.get(point.kind, point.kind)}）超时 {limit:.0f}s',
            )
        return None

    def _check_hud(self, force: bool = False) -> OperationRoundResult | None:
        """局内 HUD 检查：暂停菜单直接失败。

        菜单盖满整屏，小地图也被盖住，打开时定位必然失败，所以定位失败那一轮
        总是检查（``force``）。其余移动轮按 ``HUD_CHECK_INTERVAL`` 节流兜底，
        防菜单弹出动画的中间帧被误定位。这次 OCR 约 70ms，不再每轮都做；
        碎步节拍由 ``STEP_REST_SECONDS`` 管，不依赖轮次耗时。

        这里不能因为没识别到普通攻击按钮就松前进键。长距离行进时
        只要某一帧模板没命中，前进键就被松开，表现就是走两步停一下。
        只有确认打开了暂停菜单才停。

        Args:
            force: 为 True 时不看节流，本轮必查。
        """
        now = time.monotonic()
        if not force and now - self.hud_checked_at < HUD_CHECK_INTERVAL:
            return None
        self.hud_checked_at = now
        with self.perf.measure('HUD'):
            paused = self.round_by_find_area(
                self.last_screenshot, '战斗-菜单', '按钮-退出战斗',
            ).is_success
        if paused:
            self._release_forward()
            return self.round_recoverable_fail('移动中打开了暂停菜单')
        return None

    def _fail_if_action_limit(self) -> OperationRoundResult | None:
        """单点输入次数上限。"""
        if self.steps < ACTION_LIMIT:
            return None
        self._release_forward()
        return self.round_recoverable_fail('单点输入达到上限')

    def _current_point(self) -> BagelRoutePoint | None:
        """当前路线游标处的点；还没确定路线时返回 None。"""
        if not self.routes:
            return None
        route = self.routes[self.route_index]
        if self.cursor >= len(route.points):
            return None
        return route.points[self.cursor]

    def _enter_point(self, route_index: int, point_index: int) -> None:
        """进入某个点：切换游标并清空单点状态。"""
        self.route_index = route_index
        self.cursor = point_index
        self.point_started_at = time.monotonic()
        self._reset_point_state()

    def _reset_point_state(self) -> None:
        """清空与单个点绑定的交互状态。

        转向比例不重置：它是整局共享的校准结果，换点后继续沿用。
        只清掉待学习样本，避免拿跨点的旧样本反推。
        """
        self.selection.reset(self.last_screenshot_time)
        self.select_since = None
        self.cleanup_since = None
        self.scanned_rows.clear()
        self.opening_blocking_door = False
        self.current_kind = None
        self.f_pressed = False
        self.door_missing_at = None
        self.press_f_taps = 0
        self.press_f_at = float('-inf')
        self.mechanical_hold_until = None
        self.mechanical_retried = False
        self.mechanical_lost_since = None
        self.mechanical_lost_count = 0
        self.card_hold_started = None
        self.card_retried = False
        self.card_held = False
        self.card_pressed = False
        self.card_seen_loaded = False
        self.card_released_at = None
        self.card_spent_since = None
        self.card_stable_since = None
        self.card_lost_since = None
        self.card_hold_lost_since = None
        self.card_hold_lost_count = 0
        self.stray_unlock_escapes = 0
        self.blocked_door_taps = 0
        self.card_probe_step = None
        self.card_verify_since = None
        self.target_wait_since = None
        self.electronic_small_steps = False
        self.small_step_since = None
        self.steps = 0
        self.turning_in_place = False
        self.cruise_over_frame = None
        self.stuck_since = None
        self.stuck_steps = 0
        self.slow_stuck_since = None
        self.retreat_point = None
        self.retreat_since = None
        self.retreats = 0
        self.elevator_phase = None
        self.elevator_since = None

    def _step_forward(self, duration: float) -> bool:
        """请求一次前进短按，返回输入线程是否接受。

        Args:
            duration: 这一次按住的时长（秒）。
        """
        return self.input_worker.step_forward(duration)

    def _record_input(self) -> None:
        """记录一次输入并拒绝复用旧截图。"""
        self.steps += 1
        self.last_input_frame = max(time.time(), self.last_screenshot_time)

    def _release_forward(self) -> None:
        """松开前进键与冲刺键，并清卡位计时与步进计时。"""
        self.stuck_since = None
        self._stop_sprint()
        self.input_worker.stop_forward()
        self.input_worker.stop_backward()

    def _read_survey_value(self) -> None:
        """关闭面板回到局内后读 HUD 调查收益；读不到时保留上一次读数。"""
        value = self._read_hud_survey()
        if value is not None:
            self.survey_value = value

    def _read_hud_survey(self) -> int | None:
        """读 HUD 调查收益区域最下面一行的数字，只用于统计日志。

        物品入箱后，HUD 数字上方可能浮出该件涨幅（如「+10K」），会落进读数区域。
        OCR 读不出浮字前的「+」，整块拼起来会先读到涨幅，所以只取最下面一行。

        Returns:
            读数；读不出数字或读取出错时为 None。
        """
        try:
            area = self.ctx.screen_loader.get_area('贝果-局内', '调查收益')
            if area is None:
                raise ValueError('缺少贝果画面区域：贝果-局内/调查收益')
            matches = self.ctx.ocr_service.get_ocr_result_list(
                self.last_screenshot, rect=area.pc_rect, crop_first=True, color_range=area.color_range,
            )
        except Exception:
            log.error('读取调查收益失败，本次不计入统计', exc_info=True)
            return None
        if not matches:
            return None
        bottom = max(match.center.y for match in matches)
        line = sorted(
            (match for match in matches if bottom - match.center.y <= SURVEY_LINE_TOLERANCE),
            key=lambda match: match.x,
        )
        return parse_value(''.join(match.data for match in line))

    def _adopt_screenshot(self, op: BagelOperation) -> None:
        """子操作结束后同步其最后截图。"""
        if op.last_screenshot is not None:
            self.last_screenshot = op.last_screenshot
            self.last_screenshot_time = op.last_screenshot_time

    def handle_pause(self) -> None:
        """暂停时松键，并让转向线程立刻停手。

        暂停期间主线不再提交，靠提交新鲜度线程 1 秒后也会自己停；
        这里主动清一次，不等那 1 秒，避免暂停瞬间按旧目标转镜头。
        排队中的侧移、长按 F 等即时动作随 ``release_all`` 作废，暂停期间不会再按下。
        """
        self.input_worker.set_turn_paused(True)
        self._release_forward()
        self.turning_in_place = False
        self.cruise_over_frame = None
        self.selection.reset(self.last_screenshot_time)
        self.select_since = None
        self.card_stable_since = None
        if self.card_hold_started is not None:
            self.card_hold_started = None
            self.card_held = False
        self.mechanical_hold_until = None
        self.pickup_fast_since = None
        self._flush_decision_log()
        self.capture_stats.finish_round('其他', close=True)
        self.input_worker.release_all()
        self.input_worker.submit_turn(None)
        super().handle_pause()

    def handle_resume(self) -> None:
        """恢复后等主线拍到新图再接受转向目标。"""
        self._turn_resume_at = time.perf_counter()
        self.input_worker.set_turn_paused(False)
        super().handle_resume()

    def is_failure_scene(self, result: OperationResult) -> bool:
        """出生点没有可执行路线是正常的退出重开，不存图也不报错。"""
        return super().is_failure_scene(result) and result.status != self.STATUS_ANCHOR_MISS

    def execute(self) -> OperationResult:
        """初始化失败时框架不调用 ``after_operation_done``，这里兜底停掉输入线程。"""
        try:
            return super().execute()
        finally:
            self.input_worker.shutdown()
            self.capture_stats.finish_route()

    def after_operation_done(self, result: OperationResult) -> None:
        """退出路径统一松键，含冲刺键，并停掉输入线程。"""
        self._flush_decision_log()
        self._stop_sprint()
        self.input_worker.shutdown()
        super().after_operation_done(result)
