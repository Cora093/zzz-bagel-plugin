from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from one_dragon.utils import os_utils
from one_dragon.utils.log_utils import log
from one_dragon_qt.services.app_setting.app_setting_provider import GroupIdMixin
from one_dragon_qt.utils.config_utils import get_prop_adapter
from one_dragon_qt.widgets.setting_card.switch_setting_card import SwitchSettingCard
from one_dragon_qt.widgets.vertical_scroll_interface import VerticalScrollInterface
from PySide6 import QtCore
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    FluentIcon,
    FluentStyleSheet,
    IconWidget,
    MessageBox,
    MessageBoxBase,
    PushButton,
    ScrollArea,
    SimpleCardWidget,
    StrongBodyLabel,
    SubtitleLabel,
    TransparentPushButton,
    setCustomStyleSheet,
    setFont,
)
from qfluentwidgets.components.widgets.card_widget import CardSeparator

from .. import usage
from ..config import BagelDifficulty, BagelMap, BagelMode
from ..const import APP_ID
from ..defs import (
    CLEAN_MODE_CUSTOM,
    CLEAN_TYPES,
    DEFAULT_CLEAN_STEPS,
    DIFFICULTY_HARD,
    DIFFICULTY_NORMAL,
    MODE_DIFFICULTIES,
    MODE_KEYCARD,
    MODE_MAPS,
)
from ..route.route_store import BagelRouteConfigError, BagelRouteStore
from ..vision.template import template_exists
from .setting_widgets import (
    AutoSaleSwitch,
    ChoiceSegment,
    CleanStepCard,
    CountSpinBox,
    SaleModeSelector,
    SaleScopePreview,
    tag_row,
)

if TYPE_CHECKING:
    from zzz_od.context.zzz_context import ZContext

    from ..config import BagelConfig


class _UsageDialog(MessageBoxBase):
    """按主题折叠的使用说明，不增加启动或保存操作。"""

    def __init__(self, parent: QWidget) -> None:
        """先显示启动检查，其他规则按需展开。"""
        super().__init__(parent)
        self.widget.setFixedWidth(680)
        self.viewLayout.addWidget(SubtitleLabel('使用说明', self))
        intro = BodyLabel('先完成「开始前」的 4 项检查。其他规则可按需展开。', self)
        intro.setWordWrap(True)
        self.viewLayout.addWidget(intro)
        terms = CaptionLabel(
            '入箱：将物品装入局内安全箱。\n入仓：将携带物存入研究站仓库。\n空箱：结算开始时安全箱为空。', self
        )
        terms.setWordWrap(True)
        self.viewLayout.addWidget(terms)
        scroll = ScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(max(240, min(460, parent.height() - 220)))
        contents = QWidget(scroll)
        column = QVBoxLayout(contents)
        column.setContentsMargins(0, 0, 8, 0)
        column.setSpacing(10)
        for index, (title, entries) in enumerate(usage.HELP_SECTIONS):
            card = SimpleCardWidget(contents)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(12, 10, 12, 12)
            heading = TransparentPushButton(FluentIcon.ARROW_DOWN, title, card)
            heading.setCheckable(True)
            heading.setChecked(index == 0)
            list_tag = 'ol' if index == 0 else 'ul'
            items = ''.join(
                f'<li style="margin-bottom: 8px;"><b>{escape(label)}</b>：{escape(body)}</li>'
                for label, body in entries
            )
            text = BodyLabel(
                f'<{list_tag} style="margin-top: 0; margin-bottom: 0;">{items}</{list_tag}>',
                card,
            )
            text.setTextFormat(Qt.TextFormat.RichText)
            text.setWordWrap(True)
            text.setVisible(index == 0)
            heading.toggled.connect(text.setVisible)
            layout.addWidget(heading, 0, Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(text)
            column.addWidget(card)
        column.addStretch(1)
        scroll.setWidget(contents)
        # 滚动区及内容容器透出弹窗底色，避免 Qt 默认浅色背景覆盖黑夜主题。
        scroll.enableTransparentBackground()
        self.viewLayout.addWidget(scroll)
        self.yesButton.setText('关闭')
        self.hideCancelButton()


class NonHazardNoticeDialog(MessageBoxBase):
    """非高危难度路线切换提示弹窗，带 10 秒强制阅读倒计时。"""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.widget.setFixedWidth(520)
        self.viewLayout.addWidget(SubtitleLabel('困难和常规需要换路线', self))
        content = BodyLabel(
            '1. 困难和常规会刷出「助战物资」，沿用高危路线时可能卡在这些物资上。\n'
            '2. 天赋未满时，卸下装备后的血量也不够跑完较长路线。\n'
            '\n'
            '请在路线编辑器中改开「录像店-非高危」和「白鸽工地站-非高危」，\n'
            '并关闭其他出生点。\n'
            '点确定不会自动切换路线。',
            self,
        )
        content.setWordWrap(True)
        self.viewLayout.addWidget(content)
        self.hideCancelButton()
        self.setClosableOnMaskClicked(False)
        self.remaining_seconds: int = 10
        self.yesButton.setEnabled(False)
        self.yesButton.setText(f'确定 ({self.remaining_seconds}s)')

        self.timer: QTimer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    def _on_tick(self) -> None:
        self.remaining_seconds -= 1
        if self.remaining_seconds > 0:
            self.yesButton.setText(f'确定 ({self.remaining_seconds}s)')
        else:
            self.timer.stop()
            self.yesButton.setEnabled(True)
            self.yesButton.setText('确定')

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.remaining_seconds > 0 and event.key() == Qt.Key.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.remaining_seconds > 0:
            event.ignore()
            return
        self.timer.stop()
        super().closeEvent(event)


class BagelSettingInterface(VerticalScrollInterface, GroupIdMixin):
    """贝果计划精简设置页，配置仍按账号和应用组即时保存。"""

    def __init__(self, ctx: ZContext, parent: QWidget | None = None) -> None:
        """复用主窗口导航与滚动容器。"""
        super().__init__(
            content_widget=None,
            object_name='setting_interface',
            nav_text_cn='贝果计划设置',
            parent=parent,
        )
        self.ctx: ZContext = ctx
        self.config: BagelConfig | None = None
        self.clean_step_cards: list[CleanStepCard] = []
        self._config_loaded: bool = False

    def get_content_widget(self) -> QWidget:
        """显示结束条件、入箱优先级与自动出售，首次清空提示放入使用说明。"""
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        header = QHBoxLayout()
        title = QVBoxLayout()
        title.addWidget(SubtitleLabel('贝果计划', widget))
        title.addWidget(CaptionLabel('设置仅用于当前账号和应用组', widget))
        header.addLayout(title)
        header.addStretch(1)
        help_button = TransparentPushButton('使用说明', widget)
        help_button.clicked.connect(self._show_usage)
        header.addWidget(help_button, 0, Qt.AlignmentFlag.AlignTop)
        # 路线编辑器依赖源码目录与网页资源，打包运行时没有这些文件，只在源码模式提供入口。
        if not os_utils.run_in_exe():
            editor_button = TransparentPushButton('打开路线编辑器', widget)
            editor_button.clicked.connect(self._open_route_editor)
            header.addWidget(editor_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)

        layout.addWidget(self._make_target_card(widget))
        self.keycard_card: QWidget = self._make_keycard_card(widget)
        layout.addWidget(self.keycard_card)
        layout.addWidget(self._make_end_conditions(widget))
        self.blueprint_priority: SwitchSettingCard = SwitchSettingCard(
            icon=FluentIcon.DOCUMENT,
            title='蓝图优先',
            content='局内优先拾取蓝图',
            parent=widget,
        )
        self.blueprint_priority.setFixedHeight(88)
        layout.addWidget(self.blueprint_priority)
        layout.addWidget(self._make_sale_group(widget))
        layout.addStretch(1)
        self._refresh_clean_options()
        return widget

    def _make_target_card(self, parent: QWidget) -> QWidget:
        """模式、地图、难度三行；当前模式用不到的选项置灰，原因写在右侧。"""
        card = SimpleCardWidget(parent)
        grid = QGridLayout(card)
        grid.setContentsMargins(20, 16, 20, 16)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(2, 1)
        self.mode_segment: ChoiceSegment = ChoiceSegment(
            [(item.value.value, item.value.ui_text) for item in BagelMode], '运行模式', card,
        )
        self.map_segment: ChoiceSegment = ChoiceSegment(
            [(item.value.value, item.value.ui_text) for item in BagelMap], '地图', card,
        )
        self.difficulty_segment: ChoiceSegment = ChoiceSegment(
            [(item.value.value, item.value.ui_text) for item in BagelDifficulty], '挑战难度', card,
        )
        self.map_hint: CaptionLabel = CaptionLabel(card)
        self.difficulty_hint: CaptionLabel = CaptionLabel(card)
        for row, (label, segment, hint) in enumerate((
            ('模式', self.mode_segment, None),
            ('地图', self.map_segment, self.map_hint),
            ('难度', self.difficulty_segment, self.difficulty_hint),
        )):
            grid.addWidget(BodyLabel(label, card), row, 0)
            grid.addWidget(segment, row, 1)
            if hint is not None:
                grid.addWidget(hint, row, 2)
        self.mode_segment.currentItemChanged.connect(self._on_mode_changed)
        self.map_segment.currentItemChanged.connect(self._on_map_changed)
        self.difficulty_segment.currentItemChanged.connect(self._on_difficulty_changed)
        return card

    def _make_keycard_card(self, parent: QWidget) -> QWidget:
        """开房卡专属：房间、路线就绪情况、保留张数。只在开房卡模式显示。"""
        card = SimpleCardWidget(parent)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        room_row = QHBoxLayout()
        room_row.setSpacing(12)
        room_row.addWidget(BodyLabel('房间', card))
        self.room_box: ComboBox = ComboBox(card)
        self.room_box.setMinimumWidth(ChoiceSegment.WIDTH)
        self.room_box.setAccessibleName('开房卡房间')
        self.room_box.currentIndexChanged.connect(self._on_room_changed)
        room_row.addWidget(self.room_box)
        room_row.addStretch(1)
        layout.addLayout(room_row)
        self.keycard_status: CaptionLabel = CaptionLabel(card)
        self.keycard_status.setWordWrap(True)
        layout.addWidget(self.keycard_status)
        reserve_row = QHBoxLayout()
        reserve_row.setSpacing(8)
        self.keycard_reserve: CountSpinBox = CountSpinBox(0, 999, card)
        self.keycard_reserve.setAccessibleName('保留张数')
        self.keycard_reserve.setToolTip('仓库里这个房间的卡不多于这个数就结束任务；0 表示用到没卡为止')
        self.keycard_reserve.valueChanged.connect(lambda _value: self._refresh_keycard_status())
        reserve_row.addWidget(BodyLabel('仓库保留', card))
        reserve_row.addWidget(self.keycard_reserve)
        reserve_row.addWidget(BodyLabel('张门禁卡', card))
        reserve_row.addStretch(1)
        layout.addLayout(reserve_row)
        return card

    def _refresh_target(self) -> None:
        """按配置刷新三组按钮的选中、置灰和开房卡卡片的显示。"""
        if self.config is None:
            return
        mode = self.config.mode
        mode_name = BagelMode.KEYCARD.value.ui_text if mode == MODE_KEYCARD else BagelMode.RUN.value.ui_text
        maps = MODE_MAPS[mode]
        difficulties = MODE_DIFFICULTIES[mode]
        self.mode_segment.set_value(mode)
        self.map_segment.set_value(self.config.map_name)
        self.difficulty_segment.set_value(self.config.run_difficulty)
        map_reason = f'{mode_name}仅支持{"、".join(maps)}'
        self.map_segment.set_allowed(maps, map_reason)
        self.map_hint.setText(map_reason if len(maps) < len(BagelMap) else '')
        difficulty_reason = f'{mode_name}固定{"、".join(difficulties)}'
        self.difficulty_segment.set_allowed(difficulties, difficulty_reason)
        self.difficulty_hint.setText(difficulty_reason if len(difficulties) < len(BagelDifficulty) else '')
        self.keycard_card.setVisible(self.config.is_keycard)
        if self.config.is_keycard:
            self._refresh_keycard_status()

    def _load_rooms(self) -> None:
        """读房间清单填下拉框；未选房间时选第一个并保存。"""
        if self.config is None:
            return
        try:
            rooms = BagelRouteStore().load_catalog().rooms
        except (OSError, ValueError, BagelRouteConfigError):
            log.error('贝果设置页读取房间清单失败', exc_info=True)
            rooms = []
        with QtCore.QSignalBlocker(self.room_box):
            self.room_box.clear()
            for room in rooms:
                self.room_box.addItem(room.name, userData=room.room_id)
            ids = [room.room_id for room in rooms]
            if self.config.keycard_room not in ids and ids and self.config.is_keycard:
                self.config.keycard_room = ids[0]
            if self.config.keycard_room in ids:
                self.room_box.setCurrentIndex(ids.index(self.config.keycard_room))

    def _refresh_keycard_status(self) -> None:
        """一行写清房间能不能跑：可用的出生点，或缺什么。"""
        if self.config is None:
            return
        room_id = self.config.keycard_room
        problems: list[str] = []
        spawns: list[str] = []
        try:
            catalog = BagelRouteStore().load_catalog()
        except (OSError, ValueError, BagelRouteConfigError) as error:
            catalog = None
            problems.append(str(error))
        if catalog is not None:
            if catalog.room(room_id) is None:
                problems.append('未选择房间')
            else:
                problem = catalog.keycard_problem(room_id)
                if problem is not None:
                    problems.append(problem)
                selection = catalog.keycard_selection(room_id)
                if selection.room is None or catalog.route(selection.room) is None:
                    problems.append('未开启房间段')
                spawns = [spawn.name for spawn in catalog.spawns if selection.entries.get(spawn.spawn_id) is not None]
                if not spawns:
                    problems.append('没有出生点开启入口段')
        if self.config.keycard_reserve > 0 and not template_exists(self.config.keycard_template):
            problems.append('没有卡图标，保留张数只能填 0')
        if problems:
            self.keycard_status.setText('；'.join(problems))
            self.keycard_status.setTextColor('#a4262c', '#f0b6bb')
        else:
            self.keycard_status.setText(f'可用出生点：{"、".join(spawns)}')
            self.keycard_status.setTextColor('#606060', '#a0a0a0')

    def _on_mode_changed(self, mode: str) -> None:
        """切换模式；地图不被新模式支持时跳到它的默认地图。"""
        if not self._config_loaded or self.config is None:
            return
        self.config.mode = mode
        if not self.config.map_supported:
            self.config.map_name = MODE_MAPS[mode][0]
        if self.config.is_keycard and not self.config.keycard_room:
            self._load_rooms()
        self._refresh_target()

    def _on_map_changed(self, map_name: str) -> None:
        """保存地图。"""
        if not self._config_loaded or self.config is None:
            return
        self.config.map_name = map_name
        self._refresh_target()

    def _on_room_changed(self, index: int) -> None:
        """保存房间并刷新就绪情况。"""
        if not self._config_loaded or self.config is None or index < 0:
            return
        room_id = self.room_box.itemData(index)
        if isinstance(room_id, str):
            self.config.keycard_room = room_id
        self._refresh_keycard_status()

    def _make_end_conditions(self, parent: QWidget) -> QWidget:
        """两个计数输入组成一句话。"""
        card = SimpleCardWidget(parent)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)
        sentence = QHBoxLayout()
        sentence.setSpacing(8)
        self.success_count: CountSpinBox = CountSpinBox(0, 1000, card)
        self.failure_count: CountSpinBox = CountSpinBox(1, 10, card)
        self.success_count.setAccessibleName('累计成功次数')
        self.failure_count.setAccessibleName('连续失败次数')
        sentence.addWidget(BodyLabel('累计成功', card))
        sentence.addWidget(self.success_count)
        sentence.addWidget(BodyLabel('次 或 连续失败', card))
        sentence.addWidget(self.failure_count)
        sentence.addWidget(BodyLabel('次后结束任务', card))
        sentence.addStretch(1)
        layout.addLayout(sentence)
        hint = CaptionLabel('成功次数为 0 表示不限；全局或同一路线连续失败达到上限时结束。', card)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return card

    def _make_sale_group(self, parent: QWidget) -> QWidget:
        """自动出售父卡片包含间隔、范围和步骤；关闭时收起。"""
        card = SimpleCardWidget(parent)
        card.setBorderRadius(8)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(20, 17, 22, 17)
        header.setSpacing(12)
        icon = IconWidget(FluentIcon.TAG, card)
        icon.setFixedSize(24, 24)
        header.addWidget(icon)
        header.addWidget(StrongBodyLabel('自动出售', card), 1)
        self.auto_clean_switch: AutoSaleSwitch = AutoSaleSwitch(card)
        self.auto_clean_switch.state_changed.connect(self._refresh_clean_options)
        header.addWidget(self.auto_clean_switch)
        layout.addLayout(header)
        layout.addWidget(CardSeparator(card))
        self.sale_children: QFrame = QFrame(card)
        self.sale_children.setObjectName('saleChildren')
        self.sale_children.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        FluentStyleSheet.CARD_WIDGET.apply(self.sale_children)
        setCustomStyleSheet(
            self.sale_children,
            'QFrame#saleChildren { background: #fafbfc; border: none; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; }',
            'QFrame#saleChildren { background: #25262a; border: none; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; }',
        )
        children = QVBoxLayout(self.sale_children)
        children.setContentsMargins(60, 20, 22, 22)
        children.setSpacing(14)
        interval = QHBoxLayout()
        interval.setSpacing(8)
        self.sell_interval: CountSpinBox = CountSpinBox(1, 999, self.sale_children)
        self.sell_interval.setAccessibleName('出售间隔')
        interval.addWidget(BodyLabel('每累计入仓', self.sale_children))
        interval.addWidget(self.sell_interval)
        interval.addWidget(BodyLabel('次后出售', self.sale_children))
        interval.addStretch(1)
        children.addLayout(interval)
        warning = QVBoxLayout()
        warning.setSpacing(2)
        warning_title = StrongBodyLabel('已有库存也会出售', self.sale_children)
        warning_title.setTextColor('#805000', '#e9c986')
        warning.addWidget(warning_title)
        safe_hint = CaptionLabel(
            '安全箱内符合出售条件的物品也可能被出售。', self.sale_children
        )
        safe_hint.setWordWrap(True)
        warning.addWidget(safe_hint)
        children.addLayout(warning)
        children.addWidget(CardSeparator(self.sale_children))
        scope_header = QHBoxLayout()
        scope_header.setSpacing(20)
        scope_header.addWidget(StrongBodyLabel('出售范围', self.sale_children))
        self.clean_mode: SaleModeSelector = SaleModeSelector(self.sale_children)
        self.clean_mode.state_changed.connect(self._refresh_clean_options)
        scope_header.addWidget(self.clean_mode)
        scope_header.addStretch(1)
        children.addLayout(scope_header)
        self.default_scope: QWidget = self._make_default_scope(self.sale_children)
        children.addWidget(self.default_scope)
        self.custom_scope: QWidget = self._make_custom_scope(self.sale_children)
        children.addWidget(self.custom_scope)
        layout.addWidget(self.sale_children)
        self.sale_disabled: CaptionLabel = CaptionLabel('只入仓，不出售。', card)
        self.sale_disabled.setContentsMargins(60, 13, 22, 15)
        layout.addWidget(self.sale_disabled)
        return card

    def _make_default_scope(self, parent: QWidget) -> QWidget:
        """每一步同排显示类型和品质，默认步骤与实际出售共用业务常量。"""
        widget = QWidget(parent)
        column = QVBoxLayout(widget)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)
        hint = CaptionLabel('按以下顺序出售，每一步都需同时符合物品类型和品质。', widget)
        hint.setWordWrap(True)
        column.addWidget(hint)
        table = QGridLayout()
        table.setHorizontalSpacing(16)
        table.setVerticalSpacing(12)
        table.setColumnStretch(1, 1)
        table.setColumnStretch(2, 1)
        for index, title in enumerate(('执行顺序', '物品类型', '品质')):
            table.addWidget(CaptionLabel(title, widget), 0, index)
        for row, (types, qualities) in enumerate(DEFAULT_CLEAN_STEPS, start=1):
            table.addWidget(CaptionLabel(f'第 {row} 步', widget), row, 0)
            table.addWidget(tag_row(types, widget), row, 1)
            table.addWidget(tag_row(qualities, widget), row, 2)
        column.addLayout(table)
        keep_hint = QLabel('金色蓝图默认保留', widget)
        keep_hint.setObjectName('blueprintKeepHint')
        FluentStyleSheet.LABEL.apply(keep_hint)
        setFont(keep_hint, 13)
        setCustomStyleSheet(
            keep_hint,
            'QLabel#blueprintKeepHint { color: #236344; background: #edf7f0; border: 1px solid #b6d8c2; border-radius: 5px; padding: 10px 12px; }',
            'QLabel#blueprintKeepHint { color: #a6dfbe; background: #29322e; border: 1px solid #455a4d; border-radius: 5px; padding: 10px 12px; }',
        )
        column.addWidget(keep_hint)
        label = CaptionLabel('同时保留全部 Z 品质，以及装备、战术道具和门禁卡。', widget)
        label.setWordWrap(True)
        column.addWidget(label)
        return widget

    def _make_custom_scope(self, parent: QWidget) -> QWidget:
        """自定义编辑区下方只有一份总预览。"""
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        hint = CaptionLabel(
            '每步都要选择类型和品质，同时符合才出售。按从上到下的顺序执行。', widget
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.clean_step_layout: QVBoxLayout = QVBoxLayout()
        self.clean_step_layout.setSpacing(10)
        layout.addLayout(self.clean_step_layout)
        self.empty_steps: BodyLabel = BodyLabel(
            '先添加一步，再选择要出售的类型和品质。', widget
        )
        layout.addWidget(self.empty_steps)
        self.add_clean_step_btn: PushButton = PushButton(
            FluentIcon.ADD, '新增出售步骤', widget
        )
        self.add_clean_step_btn.clicked.connect(self._on_add_clean_step)
        layout.addWidget(self.add_clean_step_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.sale_validation: BodyLabel = BodyLabel(widget)
        self.sale_validation.setWordWrap(True)
        self.sale_validation.setTextColor('#a4262c', '#f0b6bb')
        layout.addWidget(self.sale_validation)
        # 文案随所选类型变化，由 _refresh_clean_options 设置。
        self.z_warning: QLabel = QLabel(widget)
        self.z_warning.setWordWrap(True)
        self.z_warning.setObjectName('zWarning')
        FluentStyleSheet.LABEL.apply(self.z_warning)
        setFont(self.z_warning, 13)
        setCustomStyleSheet(
            self.z_warning,
            'QLabel#zWarning { color: #805000; background: #fff8eb; border: 1px solid #e8d3a7; border-radius: 5px; padding: 10px 12px; }',
            'QLabel#zWarning { color: #e9c986; background: #342b23; border: 1px solid #67503b; border-radius: 5px; padding: 10px 12px; }',
        )
        layout.addWidget(self.z_warning)
        # 文案随所选品质变化，由 _refresh_clean_options 设置。
        self.blueprint_sale_warning: QLabel = QLabel(widget)
        self.blueprint_sale_warning.setWordWrap(True)
        self.blueprint_sale_warning.setObjectName('blueprintSaleWarning')
        FluentStyleSheet.LABEL.apply(self.blueprint_sale_warning)
        setFont(self.blueprint_sale_warning, 13)
        setCustomStyleSheet(
            self.blueprint_sale_warning,
            'QLabel#blueprintSaleWarning { color: #805000; background: #fff8eb; border: 1px solid #e8d3a7; border-radius: 5px; padding: 10px 12px; }',
            'QLabel#blueprintSaleWarning { color: #e9c986; background: #342b23; border: 1px solid #67503b; border-radius: 5px; padding: 10px 12px; }',
        )
        layout.addWidget(self.blueprint_sale_warning)
        self.total_preview: SaleScopePreview = SaleScopePreview(widget)
        layout.addWidget(self.total_preview)
        return widget

    def _refresh_clean_options(self) -> None:
        """只更新可见区域与总预览，不重建正在操作的多选菜单。"""
        enabled = self.auto_clean_switch.isChecked()
        custom = self.clean_mode.currentRouteKey() == CLEAN_MODE_CUSTOM
        self.sale_children.setVisible(enabled)
        self.sale_disabled.setVisible(not enabled)
        self.default_scope.setVisible(not custom)
        self.custom_scope.setVisible(custom)
        steps = [] if self.config is None else self.config.clean_steps
        incomplete = [
            str(index + 1)
            for index, step in enumerate(steps)
            if not step.types or not step.qualities
        ]
        self.empty_steps.setVisible(not steps)
        if not steps:
            validation = '尚未添加出售步骤，任务无法启动。请新增一步，或改用默认方案。'
        elif incomplete:
            validation = f'第 {"、".join(incomplete)} 步未选全类型和品质，任务无法启动。请补全选项，或删除未选全的步骤。'
        else:
            validation = ''
        self.sale_validation.setText(validation)
        self.sale_validation.setVisible(bool(validation))
        z_steps = [step for step in steps if 'Z' in step.qualities]
        z_types = '、'.join(
            name for name in CLEAN_TYPES if any(name in step.types for step in z_steps)
        )
        self.z_warning.setText(f'红色「{z_types or "物品"}」会被出售，请注意确认！')
        self.z_warning.setVisible(bool(z_steps))
        # 「其他」类型包含蓝图：S 品质为金色蓝图，Z 品质为红色蓝图。
        blueprint_colors = '和'.join(
            color for quality, color in (('S', '金色'), ('Z', '红色'))
            if any('其他' in step.types and quality in step.qualities for step in steps)
        )
        self.blueprint_sale_warning.setText(f'{blueprint_colors}蓝图会被出售，请注意确认！')
        self.blueprint_sale_warning.setVisible(bool(blueprint_colors))
        self.total_preview.set_steps(steps)

    def _rebuild_clean_steps(self) -> None:
        """仅加载、增删或排序时重建；编辑选项时保留菜单与滚动位置。"""
        while self.clean_step_layout.count():
            item = self.clean_step_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self.clean_step_cards.clear()
        steps = [] if self.config is None else self.config.clean_steps
        for index, step in enumerate(steps):
            card = CleanStepCard(index, len(steps), step, self.custom_scope)
            card.changed.connect(lambda current=index: self._save_clean_step(current))
            card.move_requested.connect(
                lambda offset, current=index: self._move_clean_step(current, offset)
            )
            card.delete_requested.connect(
                lambda current=index: self._delete_clean_step(current)
            )
            self.clean_step_layout.addWidget(card)
            self.clean_step_cards.append(card)
        self._refresh_clean_options()

    def _save_clean_step(self, index: int) -> None:
        """改完这一步立刻写入，不关闭菜单或重建步骤。"""
        if self.config is None or not 0 <= index < len(self.clean_step_cards):
            return
        self.config.update_clean_step(index, self.clean_step_cards[index].value())
        self._refresh_clean_options()

    def _on_add_clean_step(self) -> None:
        """追加空步骤，保留不完整配置提示。"""
        if self.config is not None:
            self.config.add_clean_step()
            self._rebuild_clean_steps()

    def _delete_clean_step(self, index: int) -> None:
        """删除指定步骤并刷新总预览。"""
        if self.config is not None:
            self.config.delete_clean_step(index)
            self._rebuild_clean_steps()

    def _move_clean_step(self, index: int, offset: int) -> None:
        """与相邻步骤交换，只改变原有执行顺序。"""
        if self.config is None:
            return
        target = index + offset
        if 0 <= target < len(self.config.clean_steps):
            self.config.move_clean_step(index, target)
            self._rebuild_clean_steps()

    def _show_usage(self) -> None:
        """在主窗口内打开说明，不启动任何游戏操作。"""
        dialog = _UsageDialog(self.window())
        dialog.exec()
        dialog.deleteLater()

    def _open_route_editor(self) -> None:
        """打开网页版路线编辑器；录制与写盘由 Python 负责。"""
        from .route_web_editor import BagelRouteEditor

        try:
            editor = BagelRouteEditor(self.ctx, self.window())
        except (OSError, ValueError) as error:
            log.error('打开贝果路线编辑器失败', exc_info=True)
            MessageBox('无法打开路线编辑器', str(error), self.window()).exec()
            return
        editor.exec()
        editor.deleteLater()
        if self.config is not None and self.config.is_keycard:
            self._refresh_keycard_status()

    def on_interface_shown(self) -> None:
        """进入页面后绑定当前账号和应用组。"""
        super().on_interface_shown()
        self.init_config()

    def init_config(self) -> None:
        """加载时阻断保存信号，旧的异步加载由适配器代次机制丢弃。"""
        self._config_loaded = False
        self.config = self.ctx.run_context.get_config(
            app_id=APP_ID,
            instance_idx=self.ctx.current_instance_idx,
            group_id=self.group_id,
        )
        self.blueprint_priority.init_with_adapter(
            get_prop_adapter(self.config, 'blueprint_priority'),
        )
        problem = self.config.stored_mode_map_problem()
        if problem is not None:
            # 手改配置写进了不认识的值：把界面按默认显示的值写回，启动校验才放行。
            log.warning('贝果设置：%s，已改回 %s · %s', problem, self.config.mode, self.config.map_name)
            self.config.mode = self.config.mode
            self.config.map_name = self.config.map_name
        if not self.config.map_supported:
            # 手改配置写进了不支持的组合：改回该模式的默认地图。
            log.warning('贝果设置：%s 不支持 %s，已改回 %s',
                        self.config.mode, self.config.map_name, MODE_MAPS[self.config.mode][0])
            self.config.map_name = MODE_MAPS[self.config.mode][0]
        self._load_rooms()
        self._refresh_target()
        for control, field in (
            (self.success_count, 'max_success_rounds'),
            (self.failure_count, 'max_consecutive_failures'),
            (self.sell_interval, 'sell_interval'),
            (self.auto_clean_switch, 'auto_clean_warehouse'),
            (self.clean_mode, 'clean_mode'),
            (self.keycard_reserve, 'keycard_reserve'),
        ):
            control.setEnabled(False)
            control.init_with_adapter(get_prop_adapter(self.config, field))
        self._rebuild_clean_steps()
        self._config_loaded = True

    def _on_difficulty_changed(self, difficulty: str) -> None:
        """保存跑刀难度；选困难或常规时弹出 10 秒倒计时路线切换提醒。开房卡固定高危，不写入。"""
        if not self._config_loaded or self.config is None or self.config.is_keycard:
            return
        self.config.difficulty = difficulty
        if difficulty in (DIFFICULTY_HARD, DIFFICULTY_NORMAL):
            dialog = NonHazardNoticeDialog(self.window())
            dialog.exec()
            dialog.deleteLater()
