# 画面素材

本目录保存开发文档引用的画面截图、裁剪图和离线对比基准。`docs/` 不进 `bagel.zip`，素材不影响用户安装。

## 目录

| 目录 | 内容 |
| --- | --- |
| [screens](screens/README.md) | 每个 screen_info 画面的整屏参考帧 |
| [item_badges](item_badges/README.md) | 物品角标源图、局内参考帧、品质取样与模板重建参数 |
| [minimap_heading](minimap_heading/README.md) | 小地图镜头朝向识别的离线对比基准 |
| [safe_and_notice](safe_and_notice/README.md) | 安全箱锁定格、时钟空格与拟态单质提示 |
| [spawn_points](spawn_points/README.md) | 雅努斯幻境出生位置代表帧与小地图定位对照 |
| [misc](misc/README.md) | 不成专题的单张画面 |

## 素材规范

新增或替换素材时按以下规则处理。

### 格式

| 用途 | 格式 |
| --- | --- |
| 模板裁剪源、识别基准、离线对比素材 | 无损 WebP |
| 裁剪图、拼图、标注图 | 无损 WebP |
| 只做文档插图、不做识别依据的整屏图 | 可以用 WebP q90 |
| 原样引用的外部图片，例如 issue 附件 | 保留原文件和原字节 |
| 视频 | 不收；需要时序证据时存关键帧，并写明帧时间 |

- 不确定用途时用无损。有损文件无法恢复精度。
- 已有的有损文件不重新编码，在目录 README 中标明“有损”，并说明不能作为模板裁剪源。
- 写文件用 `cv2.imencode('.webp', img, [cv2.IMWRITE_WEBP_QUALITY, 101])` 加 `ndarray.tofile(path)`，质量值 101 表示无损；q90 写 90。不用 `cv2.imwrite`，它在 Windows 中文路径下会失败。

主仓测试仓的截图规范以 WebP q90 为默认，精度敏感的图另存 PNG。本仓不另存 PNG 原图，所以默认无损；判断哪些图可以有损的标准与主仓一致。

### 尺寸

- 整屏图保存 1920×1080 原生分辨率，不缩放，坐标与 `screen_info` 的 `pc_rect` 一致。
- 缩放过的图只做插图，并在目录 README 中注明缩放来源。
- 裁剪图在目录 README 或 JSON 中写明裁剪矩形。

### 目录与命名

- 素材放在 `docs/materials/<专题>/`，不放进 `bagel/`。
- 文件名用小写英文和下划线，写清画面状态，例如 `warehouse_after_notice.webp`。
- 每个专题目录有 `README.md`，写用途和整体来源。以下三类图在 README 中逐个注明：有损、缩放、外部原样引用。

### 隐私与体量

- 遮挡账号标识（UID）后再入库，遮挡区域不能覆盖识别要用的像素。
- 单个文件不超过 2 MB，本目录总量不超过 30 MB。大批量回归素材不进本仓。
