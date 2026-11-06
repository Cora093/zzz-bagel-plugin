# 流程画面参考帧

本目录为 `bagel/screen_info/` 中的每个画面保存至少一张整屏参考帧，用来核对区域坐标和画面状态。专题素材（角标、朝向、安全箱、出生点）见上级目录。

## 来源与限制

- 本目录的帧全部采自同一次实机运行：插件 0.3.3 跑一局高危雅努斯幻境，采集时用 MCP 截图。
- 这一次运行先在两个非支持出生点退出重开，第三次入场的出生点受支持，于是完成采集、主动退出、入仓，并按自定义方案分两步出售。
- 所有图都是 1920×1080 原生分辨率，WebP q90 有损，只能作布局参考和区域坐标核对，不能作为模板裁剪源。
- 右下角 UID 已遮挡。

## 画面

| 画面（screen_info） | 文件 | 画面状态 | 相关文档 |
| --- | --- | --- | --- |
| 贝果-研究站 | [hub_main.webp](hub_main.webp) | 研究站主界面 | [快捷手册入口与局间返回](../../entry.md#快捷手册入口与局间返回) |
| 贝果-选图 | [map_janus_high_risk.webp](map_janus_high_risk.webp) | 已选雅努斯幻境和「高危」页签，推荐装备价值 258000，右下「前往备战」 | [选图与难度](../../entry.md#选图与难度) |
| 贝果-备战 | [prepare_carried.webp](prepare_carried.webp) | 清空前：代理人武备 63000、装备 138000、道具 63000；背包 0/50，安全箱 0/5 | [备战卸装](../../entry.md#备战卸装) |
| 贝果-备战 | [prepare_item_detail.webp](prepare_item_detail.webp) | 道具详情；右侧有放入背包、放入安全箱、放入仓储枢纽、拆分、出售、前往获取 | [备战卸装](../../entry.md#备战卸装) |
| 贝果-备战 | [prepare_cleared.webp](prepare_cleared.webp) | 清空后：三项价值都是 0，背包 0/20，安全箱 0/5；中部提示「已放入仓储枢纽」 | [入场与携带核验](../../entry.md#入场与携带核验) |
| 贝果-入场确认 | [enter_confirm_low_value.webp](enter_confirm_low_value.webp) | 「当前装备价值为0，未达到推荐价值258000……是否继续前往？」 | [入场提示](../../entry.md#入场提示) |
| 贝果-入场确认 | [enter_confirm_no_team_equip.webp](enter_confirm_no_team_equip.webp) | 「未穿戴队伍装备（异体刃、抗蚀器），是否直接出战？」 | [入场提示](../../entry.md#入场提示) |
| 贝果-入场确认 | [enter_investment_zero.webp](enter_investment_zero.webp) | 高危投资页，投资金额 0 | [投资金额归零](../../entry.md#投资金额归零) |
| 贝果-局内 | [hud_spawn.webp](hud_spawn.webp) | 刚入场：计时 25:00，调查收益 0 | [出生识别与导航](../../navigation.md#出生识别与导航) |
| 贝果-局内 | [hud_big_map.webp](hud_big_map.webp) | 局内大地图，左上「可提前进入探索」，开局倒计时 00:40 | [出生识别与导航](../../navigation.md#出生识别与导航) |
| 贝果-局内 | [hud_interact_weapon_box.webp](hud_interact_weapon_box.webp) | 靠近武备箱，右侧交互条目「武备箱」；调查收益 33K | [容器交互与解锁](../../container.md#容器交互与解锁) |
| 贝果-局内 | [search_weapon_box.webp](search_weapon_box.webp) | 武备箱搜查面板，产出价值 31500，右下「全部拾取」 | [搜查面板可操作性](../../container.md#搜查面板可操作性) |
| 贝果-局内 | [safe_unlock_calibration.webp](safe_unlock_calibration.webp) | 电子保险箱信号校准，进度 1/4，中央显示 GOOD | [容器交互与解锁](../../container.md#容器交互与解锁) |
| 贝果-局内 | [search_electronic_safe.webp](search_electronic_safe.webp) | 电子保险箱搜查面板，左侧安全箱已有 2 件 | [收集与对换](../../container.md#收集与对换) |
| 贝果-局内 | [pause_menu.webp](pause_menu.webp) | 暂停菜单「战况」页，底栏脱离卡死、暂离、退出 | [退出与结算提示](../../settle.md#退出与结算提示) |
| 贝果-退出确认 | [exit_confirm.webp](exit_confirm.webp) | 「是否确认退出战斗？」，副文说明退出后仅保留安全箱内物资 | [退出与结算提示](../../settle.md#退出与结算提示) |
| 贝果-结算 | [result_exit_empty.webp](result_exit_empty.webp) | 非支持出生点主动退出：标题 DEFEAT，收益 0，时长 00:00:02 | [退出与结算提示](../../settle.md#退出与结算提示) |
| 贝果-结算 | [result_exit_with_safe.webp](result_exit_with_safe.webp) | 采集后主动退出：标题仍是 DEFEAT，收益 245000，撤离点「无」，时长 00:00:56 | [退出与结算提示](../../settle.md#退出与结算提示) |
| 贝果-仓库 | [warehouse_after_settle.webp](warehouse_after_settle.webp) | 结算后仓库：撤离奖励 1 件，安全箱 5 件，仓库 221/280 | [入仓核验](../../settle.md#入仓核验) |
| 贝果-仓库 | [sell_quick_select.webp](sell_quick_select.webp) | 批量出售的「快速选择」弹窗，未选条件，符合数量 0 | [筛选与出售等待](../../settle.md#筛选与出售等待) |
| 贝果-仓库 | [sell_confirm.webp](sell_confirm.webp) | 「确认出售以下道具」，2 件，预估价值 230000 | [出售与结算核验](../../settle.md#出售与结算核验) |
| 贝果-仓库 | [sell_gained.webp](sell_gained.webp) | 出售后「获得」230000 齿轮硬币 | [出售与结算核验](../../settle.md#出售与结算核验) |
| 贝果-仓库 | [warehouse_after_sell.webp](warehouse_after_sell.webp) | 入仓和出售后：背包和安全箱为空，仓库 218/280，货币旁显示 +452000 | [最终核验](../../settle.md#最终核验) |

「贝果-拟态单质提示」本次没有出现，参考帧见 [safe_and_notice](../safe_and_notice/README.md)。

## 尚未覆盖

这些画面本次运行没有经过，目前没有参考帧：

- 研究站达塔前与达塔对话
- 选图「常规」「困难」页签
- 非零投资的投资页
- 局内电梯选项
- 满箱对换失败
- 战斗失败或超时的结算
- 满仓与撤离奖励提示弹窗
- 启动恢复时的仓库转存
