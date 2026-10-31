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
│   ├── bagel_const.py     # 生成物，不要手改
│   ├── bagel_defs.py      # 业务常量与 PLUGIN_DIR
│   ├── bagel_factory.py   # 必须在插件根第一层
│   ├── bagel_app_setting.py
│   ├── bagel_app.py / bagel_config.py / bagel_stats.py 等插件级模块
│   ├── operation/         # 局内操作节点
│   ├── route/             # 路线模型、录制、存储、执行
│   ├── vision/            # 画面识别与地图定位
│   ├── gui/               # 界面、浮层与线程包装
│   └── assets/ screen_info/ web/
├── scripts/               # 官方打包与校验脚本，与 plugin-template 保持一致
├── tests/
└── README.md / CHANGELOG.md / LICENSE
```

## 边界

- 只改本仓库内的文件。改到仓库外（主仓 `src/`、`docs/`、`config/`、`pyproject.toml`、
  `plugins/README.md`、根 `.gitignore`）都算越界，先问用户。
- 插件必须自包含：模板、地图底图、出生点、screen 定义、测试全在 `bagel/` 内。
  资源路径一律从 `bagel_defs.PLUGIN_DIR` 取，不读工作目录里的同名副本。
- 不要为了迁就框架去改框架。框架缺的能力，先在插件内解决，解决不了再问用户。

## 应用定位

- 贝果是**独立应用**，不进一条龙默认应用组：`plugin.json` 的 `defaultGroup` 必须保持 `false`，
  也不要加 `PRIORITY`（那只在默认组里排序时有用）。它由用户在应用列表里单独运行。
- `INTERNAL_APP` 也不要开：那会让应用只能由代码固定入口调用。

## 插件机制约束

- 扫描器只认**插件根第一层**的 `*.py`，不递归子目录：
  - `*_factory.py` 只能有一个，放 `bagel/bagel_factory.py`
  - `*_const.py` 只能有一个，放 `bagel/bagel_const.py`
  - `*_app_setting.py` 同样只在第一层被扫到
- 其余模块可以放子包，插件内一律用相对导入：
  - 同包：`from .bagel_xxx import`
  - 跨包：`from ..route.bagel_xxx import`
  - 不要出现 `from bagel.xxx import` 或 `zzz_od.application.bagel`，测试会拦下
- `bagel/bagel_const.py` 是生成物。改应用名、版本、作者、简介要改 `bagel/plugin.json`，
  然后跑 `python scripts/generate_const.py --repo .`。
  手改 const 而不改 manifest，商店校验会判 `CONST_OUTDATED`。
- `PLUGIN_AUTHOR` / `PLUGIN_HOMEPAGE` / `PLUGIN_VERSION` / `PLUGIN_DESCRIPTION`
  不要加类型注解：导入服务用正则 `名字 = '值'` 读这几个字段。

## 测试

- 命令：`PYTHONPATH=<主仓>/src uv run pytest <本仓库>/tests/`
- 测试随插件走，不放主项目测试仓 `zzz-od-test`。

## 本地运行

仓库就住在主仓 `plugins/bagel`，插件代码在它下面的 `bagel/` 里。一条龙的扫描
既认 `plugins/<插件>/`，也认 `plugins/<仓库>/<插件>/`——判据是子目录里有
`plugin.json`。所以改完代码直接运行，不需要同步或复制。

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
