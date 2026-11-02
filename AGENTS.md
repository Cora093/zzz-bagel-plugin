# AGENTS.md（贝果计划插件）

本文件只管插件仓库这一层。主仓根 `AGENTS.md` 是给主项目代码用的，
其中「测试统一放 `zzz-od-test`」「同步更新 `docs/develop/`」「改 framework `src/`」
这些约定不适用于本插件。

## 仓库结构

本仓库同时满足插件商店的收录规范，插件代码在 `bagel/` 子目录里，目录名必须等于 `plugin.json` 的 `id`：

```
（仓库根）
├── bagel/                 # 插件目录，打包后就是 zip 的顶层目录
│   ├── plugin.json        # 商店元数据，改这里再生成 const
│   ├── const.py           # 生成物，不要手改
│   ├── defs.py            # 业务常量与 PLUGIN_DIR
│   ├── factory.py         # 必须在插件根第一层
│   ├── app.py             # 应用实现
│   ├── app_setting.py / config.py / stats.py / usage.py 等插件级模块
│   ├── operation/         # 局内操作节点
│   ├── route/             # 路线模型、录制、存储、执行
│   ├── vision/            # 画面识别与地图定位
│   ├── gui/               # 界面、浮层与线程包装
│   └── assets/ screen_info/ web/
├── docs/                  # 开发文档，不进 zip
├── scripts/               # 官方打包与校验脚本，与 plugin-template 保持一致
├── tests/
└── README.md / CHANGELOG.md / LICENSE
```

## 开发文档

改动或排查游戏流程（入场、导航、转向、开箱、入箱、结算、出售、统计、路线）前，先按 [docs/README.md](docs/README.md) 的索引读对应阶段文档；改了行为就同步改该文档。

## 边界

- 只改本仓库内的文件。改到仓库外（主仓 `src/`、`docs/`、`config/`、`pyproject.toml`、
  `plugins/README.md`、根 `.gitignore`）都算越界，先问用户。
- 插件必须自包含：模板、地图底图、出生点、screen 定义、测试全在 `bagel/` 内。
  资源路径一律从 `bagel_defs.PLUGIN_DIR` 取，不读工作目录里的同名副本。
- 不要为了迁就框架去改框架。框架缺的能力，先在插件内解决，解决不了再问用户。

## 应用定位

- 贝果是一条龙**默认组应用**：`plugin.json` 的 `defaultGroup` 为 `true`；`internal_app` 保持
  `False`，所以仍可从应用列表单独运行。
- 首次进组时 `update_full_app_list` 给新应用建未持久化的 `enabled=False` 项并放在 `app_list`
  **头部**，所以贝果会以**未勾选**状态出现在一条龙列表**最上面**，要用户手动打开才会执行。
  `set_app_enable` 勾选时会持久化，勾完固定为首项，其余应用相对顺序不变。
- 一条龙按 `run_record.is_done`（当日 `STATUS_SUCCESS`）跳过本应用：当天跑成功一次后，
  再跑一条龙会显示「应用已完成 贝果计划」。独立应用运行不受此限。
- 默认组注册顺序按 `PRIORITY` 升序。插件 manifest 没有该字段的出口，贝果不声明，
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
- `bagel/const.py` 是生成物。改应用名、版本、作者、简介要改 `bagel/plugin.json`，
  然后跑 `python scripts/generate_const.py --repo .`。
  手改 const 而不改 manifest，商店校验会判 `CONST_OUTDATED`。
- `PLUGIN_AUTHOR` / `PLUGIN_HOMEPAGE` / `PLUGIN_VERSION` / `PLUGIN_DESCRIPTION`
  不要加类型注解：导入服务用正则 `名字 = '值'` 读这几个字段。

## 测试

- 命令：`PYTHONPATH=<主仓>/src uv run pytest <本仓库>/tests/`
- 测试随插件走，不放主项目测试仓 `zzz-od-test`。

## 本地运行

仓库住在主仓 `plugins/<仓库名>/`，插件代码在它下面的 `bagel/` 里。一条龙的扫描
既认 `plugins/<插件>/`，也认 `plugins/<仓库>/<插件>/`——仓库布局下的判据是子目录里
同时有工厂文件和常量文件，`plugin.json` 不参与判定。所以改完代码直接运行，不需要同步或复制。
运行期数据写在主程序工作目录的 `plugin_data/zzz-bagel-plugin/bagel/`，仓库名取自
`plugin.json` 的 homepage 末段。

## 发版

1. 改 `bagel/plugin.json`：`version` 与 `versionCode` 一起递增，`versionCode` 必须是递增整数
2. `python scripts/generate_const.py --repo .`
3. `python scripts/validate_plugin.py --repo . --strict`
4. 提交并推 tag `v<version>`，tag 必须与 `version` 对应
5. GitHub Actions 调 `scripts/build_zip.py` 打包 `<id>.zip` 并上传 Release

## 代码约定

- 函数签名与类成员变量都要类型注解；注释与 docstring 用中文。
- 1080p 坐标是既有前提，可以硬编码。
- ruff 配置在仓库根 `ruff.toml`，关闭了 TID252：插件必须用相对导入。
