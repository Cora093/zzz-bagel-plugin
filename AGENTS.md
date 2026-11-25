# AGENTS.md（贝果计划插件）

本文件只管插件仓库这一层。主仓根 `AGENTS.md` 是给主项目代码用的，
其中「测试统一放 `zzz-od-test`」「同步更新 `docs/develop/`」「改 framework `src/`」
这些约定不适用于本插件。

## 仓库结构

本仓库同时满足插件商店的收录规范，插件代码在 `bagel/` 子目录里，目录名必须等于 `plugin.json` 的 `id`：

```
（仓库根）
├── bagel/                 # 插件目录，打包后就是 zip 的顶层目录
│   ├── plugin.json        # 展示元数据：名称、版本、作者、主页、简介、封面
│   ├── const.py           # 身份与运行行为，手写维护
│   ├── defs.py            # 业务常量与 PLUGIN_DIR
│   ├── factory.py         # 必须在插件根第一层
│   ├── app.py             # 应用实现
│   ├── app_setting.py / config.py / stats.py / usage.py 等插件级模块
│   ├── operation/         # 局内操作节点
│   ├── route/             # 路线模型、录制、存储、执行
│   ├── vision/            # 画面识别与地图定位
│   ├── gui/               # 界面、浮层与线程包装
│   └── assets/ screen_info/ web/
├── docs/                  # 开发文档与画面素材（docs/materials/），不进 zip
├── tests/
└── README.md / CHANGELOG.md / LICENSE
```

## 开发文档

改动或排查游戏流程（入场、导航、转向、开箱、入箱、结算、出售、统计、路线）前，先按 [docs/README.md](docs/README.md) 的索引读对应阶段文档；改了行为就同步改该文档。

新增或替换画面素材前，先读 [docs/materials/README.md](docs/materials/README.md) 的素材规范。

## 边界

- 只改本仓库内的文件。改到仓库外（主仓 `src/`、`docs/`、`config/`、`pyproject.toml`、
  `plugins/README.md`、根 `.gitignore`）都算越界，先问用户。
- 插件必须自包含：模板、地图底图、出生点、screen 定义全在 `bagel/` 内。测试在仓库根 `tests/`，不进 zip。
  资源路径一律从 `bagel_defs.PLUGIN_DIR` 取，不读工作目录里的同名副本。
- 不要为了迁就框架去改框架。框架缺的能力，先在插件内解决，解决不了再问用户。

## 应用定位

- 贝果是一条龙**默认组应用**：`const.py` 的 `DEFAULT_GROUP` 为 `True`；`internal_app` 保持
  `False`，所以仍可从应用列表单独运行。
- 首次进组时 `update_full_app_list` 给新应用建未持久化的 `enabled=False` 项并放在 `app_list`
  **头部**，所以贝果会以**未勾选**状态出现在一条龙列表**最上面**，要用户手动打开才会执行。
  `set_app_enable` 勾选时会持久化，勾完固定为首项，其余应用相对顺序不变。
- 一条龙按 `run_record.is_done`（当日 `STATUS_SUCCESS`）跳过本应用：当天跑成功一次后，
  再跑一条龙会显示「应用已完成 贝果计划」。独立应用运行不受此限。
- 默认组注册顺序按 `PRIORITY` 升序。贝果的 `const.py` 不声明 `PRIORITY`，
  所以在注册顺序里排在声明了 `PRIORITY` 的应用之后；这不影响上面的界面列表位置。
- `INTERNAL_APP` 不要开：那会让应用只能由代码固定入口调用。

## 插件机制约束

- 扫描器只认**插件根第一层**的 `*.py`，不递归子目录：
  - 工厂文件只能有一个，固定叫 `bagel/factory.py`（旧的后缀写法 `*_factory.py` 只为兼容保留，不要再用）
  - 常量文件只能有一个，固定叫 `bagel/const.py`（同上，`*_const.py` 不再使用）
  - 插件级模块（`app_setting.py`、`config.py` 等）同样只在第一层被扫到
- 其余模块可以放子包，插件内一律用相对导入：
  - 同包：`from .xxx import`
  - 跨包：`from ..route.xxx import`
  - 不要出现 `from bagel.xxx import` 或 `zzz_od.application.bagel`，测试会拦下
- `bagel/const.py` 手写维护，只放身份与运行行为：`APP_ID`、`APP_NAME`、`DEFAULT_GROUP`、`NEED_NOTIFY`。
  校验工具用 `ast` 读字面量，所以这些值必须直接写字面量，不要用表达式或从别处导入。
- 版本、作者、主页、简介、封面只写在 `bagel/plugin.json`，一条龙直接读它；不要在 const 里再加 `PLUGIN_*`。
- 改应用名要同时改 `const.py` 的 `APP_NAME` 和 `plugin.json` 的 `name`，两者不一致时校验判 `NAME_MISMATCH`。
- `plugin.json` 不再写 `defaultGroup` / `needNotify` / `apps`，写了校验会报 `DEPRECATED_FIELD` 警告。

## 测试

- 命令：`PYTHONPATH=<主仓>/src uv run pytest <本仓库>/tests/`
- 测试随插件走，不放主项目测试仓 `zzz-od-test`。

## 校验与打包

校验和打包脚本由官方 [plugin-tools](https://github.com/OneDragon-Anyone/plugin-tools) 提供，本仓库不放拷贝。
CI 的 `release.yml` 调它的可复用工作流，`build-pr.yml` 检出它的 `v1`。本地校验（可选）在仓库根执行：

```bash
git clone --depth 1 --branch v1 https://github.com/OneDragon-Anyone/plugin-tools.git .plugin-tools
python .plugin-tools/scripts/validate_plugin.py --repo . --strict
python .plugin-tools/scripts/build_zip.py --repo .
```

`.plugin-tools/` 已加进 `.gitignore`。已有克隆时先 `git -C .plugin-tools pull`，保证与 CI 的 `@v1` 一致。

## 本地运行

仓库住在主仓 `plugins/<仓库名>/`，插件代码在它下面的 `bagel/` 里。一条龙的扫描
既认 `plugins/<插件>/`，也认 `plugins/<仓库>/<插件>/`——仓库布局下的判据是子目录里
同时有工厂文件和常量文件，`plugin.json` 不参与判定。所以改完代码直接运行，不需要同步或复制。
运行期数据写在主程序工作目录的 `plugin_data/bagel/`，由主程序按 APP_ID 分配。

## PR 合入

- 默认用 `squash` 合入并始终删除分支：`gh pr merge <PR> --squash --delete-branch --subject "<type>(bagel):<中文描述>（#<issue>）" --body ""`。
- 合入后从最新 `main` 切新分支继续开发，不复用已合入分支。

## 发版

1. 改 `bagel/plugin.json`：`version` 与 `versionCode` 一起递增，`versionCode` 必须是递增整数
2. 整理 `CHANGELOG.md`：把 `## 未发布` 改为 `## <version> — YYYY-MM-DD`，用面向用户的单行短句按「新增 / 变更 / 修复」分组（会直接用于商店更新日志；仅在需要额外保留技术细节时才在前面补 `### 概要`）
3. 按「校验与打包」一节跑 `validate_plugin.py --repo . --strict`
4. 提交并推 tag `v<version>`，tag 必须与 `version` 对应
5. GitHub Actions 调 plugin-tools 的 `build_zip.py` 打包 `<id>.zip` 并上传 Release，随后从 `CHANGELOG.md` 提取对应版本说明覆写 Release 正文

## 代码约定

- 函数签名与类成员变量都要类型注解；注释与 docstring 用中文。
- 1080p 坐标是既有前提，可以硬编码。
- ruff 配置在仓库根 `ruff.toml`，关闭了 TID252：插件必须用相对导入。
