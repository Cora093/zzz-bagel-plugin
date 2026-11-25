"""贝果计划插件包。"""

from .gui.local_web_server import disable_webengine_sandbox_on_network_runtime

# 插件被主程序加载时就设置，早于任何网页视图的创建；等到打开编辑器再设置，
# 网页引擎可能已被主程序或别的插件初始化，变量就不再生效。
disable_webengine_sandbox_on_network_runtime()
