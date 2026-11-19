"""路线决策测试的输入：已识别的条目与截图时间，方向和判稳仍由插件计算。"""

from bagel.vision.prompt_list import PromptFrame, PromptRow


class ObservedFrame(PromptFrame):
    def __init__(self, stamp: float, selected: PromptRow | None, *others: PromptRow) -> None:
        self.stamp = stamp
        self.center = selected.y if selected else None
        self.selected = selected
        visible = [selected, *others] if selected else list(others)
        self.rows = sorted(visible, key=lambda row: row.y)

    def icons(self, name: str) -> list[int]:
        """刷卡时可单独读钥匙图标。"""
        assert name == 'card_key_icon'
        return [row.y for row in self.rows if row.kind == 'card_reader']

    def text_at(self, y: int, right: int = 1575) -> str:
        return next(row.text for row in self.rows if row.y == y)
