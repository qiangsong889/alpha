"""控制台的 HTTP 服务：只在本机，读取的接口也要令牌。

安全边界（只监听 127.0.0.1 还不够，浏览器里的其他网页也能向本机发请求）：
- 启动令牌：终端打印 /login?token=...，换成 HttpOnly + SameSite=Strict 的 cookie 后跳转，地址栏里不留令牌；
  没有 cookie 的请求一律拒绝，包括页面本身和所有读取接口。
- Host 检查：只接受 127.0.0.1 / localhost，防止 DNS rebinding 让别的域名指向本机。
- 改变状态的请求（D4 起才有）必须带同源 Origin 和自定义请求头，跨站表单和简单请求都过不了。
- 安全响应头：CSP 只允许自己的脚本和样式，禁止被嵌入，不发送 Referer。
"""

from __future__ import annotations

import hmac
from pathlib import Path
import secrets
from typing import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from console.data import ConsoleData
from core.runtime import ProcessStatus

COOKIE = 'alpha_console'
CSRF_HEADER = 'x-alpha-console'
ALLOWED_HOSTS = frozenset({'127.0.0.1', 'localhost'})
SAFE_METHODS = frozenset({'GET', 'HEAD'})
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
SECURITY_HEADERS = {
    'Content-Security-Policy': CSP,
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Resource-Policy': 'same-origin',
}

BrainStatus = Callable[[str], Awaitable[ProcessStatus]]


def _page(title: str, body: str, status: int) -> HTMLResponse:
    return HTMLResponse(f'<!doctype html><meta charset="utf-8"><title>{title}</title>'
                        f'<body style="font-family:system-ui;padding:2rem"><h1>{title}</h1><p>{body}</p>', status)


def _hostname(host: str) -> str:
    # Host 头可能带端口；IPv6 字面量本来就不在白名单里，按原样返回即可被拒绝。
    if host.startswith('['):
        return host
    return host.rsplit(':', 1)[0] if ':' in host else host


def create_app(*, token: str, data: ConsoleData, brain_status: BrainStatus, web_dir: Path) -> FastAPI:
    session = secrets.token_urlsafe(32)  # cookie 的值和登录令牌分开：令牌只用来换 cookie
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def authorized(request: Request) -> bool:
        return hmac.compare_digest(request.cookies.get(COOKIE, ''), session)

    @app.middleware('http')
    async def guard(request: Request, call_next):
        host = request.headers.get('host', '')
        if _hostname(host) not in ALLOWED_HOSTS:
            return _secure(Response('Host 不被允许', 400, media_type='text/plain; charset=utf-8'), request)
        if request.method not in SAFE_METHODS:
            same_origin = request.headers.get('origin') == f'http://{host}'
            if not same_origin or request.headers.get(CSRF_HEADER) != '1':
                return _secure(JSONResponse({'detail': '请求来源不被允许'}, 403), request)
        if request.url.path != '/login' and not authorized(request):
            if request.url.path.startswith('/api/'):
                response = JSONResponse({'detail': '未登录：请用控制台启动时终端里打印的链接打开'}, 401)
            else:
                response = _page('需要登录', '请用控制台启动时终端里打印的链接打开（链接里带一次性令牌）。', 401)
            return _secure(response, request)
        return _secure(await call_next(request), request)

    @app.get('/login')
    async def login(request: Request):
        given = request.query_params.get('token', '')
        if not hmac.compare_digest(given.encode(), token.encode()):
            return _page('令牌不对', '令牌每次启动控制台都会重新生成，请用终端里最新打印的链接。', 403)
        response = RedirectResponse('/', 303)
        response.set_cookie(COOKIE, session, httponly=True, samesite='strict', path='/')
        return response

    async def brain() -> ProcessStatus:
        return await brain_status(data.brain_url())

    @app.get('/api/overview')
    async def overview():
        status = await brain()
        return await run_in_threadpool(data.overview, status)

    @app.get('/api/settings')
    async def settings():
        status = await brain()
        return await run_in_threadpool(data.settings, status)

    @app.get('/api/{path:path}')
    async def api_not_found(path: str):
        return JSONResponse({'detail': '没有这个接口'}, 404)

    dist = (web_dir / 'dist').resolve()

    @app.get('/{path:path}')
    async def frontend(path: str):
        """前端是构建好的静态文件；不存在的路径交给前端路由（返回 index.html）。"""
        index = dist / 'index.html'
        if not index.exists():
            return _page('前端还没有构建', '在 console/web 目录运行 npm install && npm run build。', 503)
        target = (dist / path).resolve()
        if path and target.is_file() and target.is_relative_to(dist):
            return FileResponse(target)
        return FileResponse(index)

    return app


def _secure(response: Response, request: Request) -> Response:
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    # 带哈希的静态资源可以缓存；页面和数据一律不缓存（里面有对话内容）。
    if not request.url.path.startswith('/assets/'):
        response.headers['Cache-Control'] = 'no-store'
    return response
