"""给路线编辑器网页提供只读的本机 HTTP 服务。

插件放在 SMB 网络路径（``\\\\主机\\共享``）时，网页引擎对 ``file://主机/...``
的访问会被拦截或在载入阶段出错，路线编辑器打不开。这时改由本进程在
``127.0.0.1`` 上把网页与底图读出来再交给网页，不让网页引擎直接碰网络路径。
"""

from __future__ import annotations

import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

INDEX_URL_PATH = '/'
"""首页路径，对应 ``web/index.html``。"""

SERVED_PREFIXES: tuple[str, ...] = ('assets/game_data/map_locator/',)
"""除首页外允许读取的相对路径前缀，其余一律 404。"""


def is_network_path(path: Path) -> bool:
    """判断路径是否为 UNC 网络路径（含映射盘解析后的结果）。

    Args:
        path: 待判断的路径。

    Returns:
        以两个反斜杠或两个正斜杠开头时为 True。
    """
    text = str(path)
    return text.startswith('\\\\') or text.startswith('//')


def resolve_request(root: Path, request_path: str) -> Path | None:
    """把请求路径映射为白名单内的文件。

    Args:
        root: 插件根目录。
        request_path: 请求行里的路径，可能带查询串。

    Returns:
        可读取的文件；不在白名单、含 ``..`` 或不是文件时为 None。
    """
    path = unquote(urlsplit(request_path).path)
    if path == INDEX_URL_PATH:
        relative = 'web/index.html'
    else:
        relative = path.lstrip('/')
        if not relative.startswith(SERVED_PREFIXES):
            return None
    if '..' in relative.split('/') or '\\' in relative:
        return None
    target = root / relative
    return target if target.is_file() else None


class LocalWebServer:
    """绑定在回环地址、随机端口上的只读静态服务。"""

    def __init__(self, plugin_dir: Path) -> None:
        """
        Args:
            plugin_dir: 插件根目录；首页取其下 ``web/index.html``。
        """
        root = plugin_dir

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                target = resolve_request(root, self.path)
                if target is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    data = target.read_bytes()
                except OSError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
                if content_type.startswith('text/') or content_type == 'application/json':
                    content_type += '; charset=utf-8'
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                """不往控制台刷访问日志。"""

        self._server: ThreadingHTTPServer = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread = threading.Thread(
            target=self._server.serve_forever, name='bagel-web-server', daemon=True,
        )
        self._thread.start()

    @property
    def index_url(self) -> str:
        """首页地址。"""
        return f'http://127.0.0.1:{self._server.server_address[1]}{INDEX_URL_PATH}'

    def close(self) -> None:
        """停止服务并释放端口。"""
        self._server.shutdown()
        self._server.server_close()
