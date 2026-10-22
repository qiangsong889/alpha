"""获取 Google OAuth 凭据；main() 可验证 Tasks 和 Calendar API。

在 alpha 目录运行：.venv/bin/python -m servers.google_auth
将桌面应用的 OAuth 配置放在项目根目录 credentials.json 中。
每次运行都分组打开浏览器重新授权：Workspace 用 token.json，YouTube 用 youtube_token.json。
Alpha 运行时只读取和刷新令牌，失败就报错，不会打开浏览器。
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from core.paths import GOOGLE_CREDENTIALS, GOOGLE_TOKEN, GOOGLE_YOUTUBE_TOKEN

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from mcp.server.mcpserver.exceptions import ToolError


# scopes 是应用向用户请求的权限范围；最终是否授权，由登录的 Google 用户确认。
# 这里获取的是登录用户的数据权限，不是项目拥有者的所有数据。
SCOPES = [
    # 日历、Tasks 完整管理；Drive 全盘只读，写入限于应用创建/明确授权的文件。
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
    # Gmail：读取邮件、发送/回复；不授予 modify、compose 或 mail.google.com。
    # 不支持管理草稿、标记已读、归档或移动到垃圾箱。
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

# Google 不允许 drive.file 与这些 YouTube scope 出现在同一次授权请求。
# 同一个桌面 OAuth 应用分两次授权；令牌文件、缓存和 API 连接均分开。
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
]

# 不能只看“所需权限是否齐全”：旧 token 可能同时保留已经移除的高权限。
FORBIDDEN_SCOPES = frozenset({
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://mail.google.com/",
})

TOKEN_PATH = GOOGLE_TOKEN
YOUTUBE_TOKEN_PATH = GOOGLE_YOUTUBE_TOKEN
CREDENTIALS_PATH = GOOGLE_CREDENTIALS


def _actual_scopes(creds: Credentials) -> set[str]:
    # 新授权/刷新以 Google 返回的 granted_scopes 为准；缓存以保存的 scopes 为准。
    scopes = creds.granted_scopes if creds.granted_scopes is not None else creds.scopes
    return set(scopes.split() if isinstance(scopes, str) else scopes or [])


def _compatible_scopes(creds: Credentials, scopes: list[str] | None = None,
                       forbidden: frozenset[str] | None = None) -> bool:
    if scopes is None:
        scopes = SCOPES
    if forbidden is None:
        forbidden = FORBIDDEN_SCOPES | frozenset(YOUTUBE_SCOPES)
    actual = _actual_scopes(creds)
    return set(scopes) <= actual and not (forbidden & actual)


AUTHORIZE_COMMAND = ".venv/bin/python -m servers.google_auth"


class GoogleAuthRequired(ToolError):
    """授权缺失、失效或权限不符：请求没有发给 Google，本次操作没有执行。

    继承 ToolError：MCP 只把 ToolError 的文字交给模型，模型才能告诉用户该怎么做。
    """

    def __init__(self, reason: str):
        super().__init__(f"Google 授权{reason}，本次没有执行。请在电脑上运行 {AUTHORIZE_COMMAND} 重新授权。")


def get_credentials() -> Credentials:
    """Workspace：Drive/Docs、Calendar、Tasks、Gmail，使用 token.json。"""
    return _get_credentials(SCOPES, TOKEN_PATH, FORBIDDEN_SCOPES | frozenset(YOUTUBE_SCOPES))


def get_youtube_credentials() -> Credentials:
    """仅 YouTube：使用 youtube_token.json；授权时选择实际要查询的频道。"""
    return _get_credentials(YOUTUBE_SCOPES, YOUTUBE_TOKEN_PATH, FORBIDDEN_SCOPES | frozenset(SCOPES))


def _get_credentials(scopes: list[str], token_path: Path, forbidden: frozenset[str]) -> Credentials:
    """运行时读取凭据：必要时刷新，失败就抛出 GoogleAuthRequired。

    绝不打开浏览器：Alpha 常驻时没有人去点登录页，交互授权只能由 authorize() 发起。
    每次都从磁盘读取，重新授权后不用重启 Alpha。
    """
    try:
        # token.json 保存之前同意授权的用户凭据。
        creds = Credentials.from_authorized_user_file(str(token_path))
    except FileNotFoundError:
        raise GoogleAuthRequired("还没有完成") from None
    except ValueError:
        raise GoogleAuthRequired("文件内容无效") from None

    # 权限增加或收窄都要重新授权；不把修改本地声明当作已经撤回 Google 端授权。
    if not _compatible_scopes(creds, scopes, forbidden):
        raise GoogleAuthRequired("的权限范围和当前要求不一致")
    if creds.valid:
        return creds
    if not creds.refresh_token:
        raise GoogleAuthRequired("已过期，且没有刷新令牌")
    try:
        # access token 过期不一定要重新登录；refresh token 可换取新的 access token。
        creds.refresh(Request())
    except RefreshError as exc:
        # refresh token 过期（测试模式 7 天）、被撤销，或改了 Google 密码。
        raise GoogleAuthRequired("已失效") from exc
    return _checked_and_saved(creds, scopes, token_path, forbidden)


def _authorize(scopes: list[str], token_path: Path, forbidden: frozenset[str]) -> Credentials:
    """交互授权：总是打开浏览器拿一个新的 refresh token，不复用旧令牌。"""
    # credentials.json 标识“哪个应用在申请权限”，与用户授权后保存的 token.json 不同。
    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), scopes)
    # 临时启动本地服务接收授权回调；0 表示使用空闲端口。
    # 桌面应用不支持增量授权；不要传 include_granted_scopes（即使值是 false）。
    # 收窄权限仍靠移除旧授权，并在下面检查实际返回的 scope。
    creds = flow.run_local_server(port=0, prompt="consent")
    # SDK 声明可能返回多种凭据；本脚本只接受普通用户 OAuth 凭据。
    if not isinstance(creds, Credentials):
        raise TypeError("授权返回的不是 Google 用户 OAuth 凭据")
    return _checked_and_saved(creds, scopes, token_path, forbidden)


def _checked_and_saved(creds: Credentials, scopes: list[str], token_path: Path,
                       forbidden: frozenset[str]) -> Credentials:
    actual = _actual_scopes(creds)
    if forbidden & actual:
        raise RuntimeError(
            "Google 仍返回已移除的高权限或混入另一组的权限。"
            "请先停止 Alpha，在 https://myaccount.google.com/connections 移除此应用的旧授权，"
            "再重新运行授权。本次不会保存或使用这份高权限凭据。"
        )
    if not set(scopes) <= actual:
        raise RuntimeError(f"授权未包含 {token_path.name} 本组请求的全部权限，请重新授权并勾选所需权限")

    # 刷新或授权后保存；只允许当前系统用户读写。先写临时文件再改名：
    # 大脑和个人服务可能同时刷新，读取方不会读到写了一半的文件。
    saved = json.loads(creds.to_json())
    saved["scopes"] = sorted(actual)  # 保留实际授予范围，不能把超额/部分授权藏在请求范围里。
    temporary = token_path.with_name(f".{token_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(saved), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, token_path)
    return creds


def authorize(service: str = "all") -> None:
    if service not in {"all", "workspace", "youtube"}:
        raise ValueError("service 必须是 all、workspace 或 youtube")
    if service in {"all", "workspace"}:
        print("授权 Workspace（Drive、Docs、Calendar、Tasks、Gmail）")
        _authorize(SCOPES, TOKEN_PATH, FORBIDDEN_SCOPES | frozenset(YOUTUBE_SCOPES))
        print("Workspace 凭据已保存到 token.json")
    if service in {"all", "youtube"}:
        print("单独授权 YouTube，请选择要查询的频道")
        _authorize(YOUTUBE_SCOPES, YOUTUBE_TOKEN_PATH, FORBIDDEN_SCOPES | frozenset(SCOPES))
        print("YouTube 凭据已保存到 youtube_token.json")


def main() -> None:
    creds = get_credentials()

    print("任务列表：")
    with build("tasks", "v1", credentials=creds) as tasks:
        # build 创建同步 API 客户端；list 构造请求，execute 才真的向 Google 发请求并等结果。
        page_token = None
        while True:
            result = tasks.tasklists().list(pageToken=page_token).execute()
            for task_list in result.get("items", []):
                print(f"- {task_list['title']}（{task_list['id']}）")
            page_token = result.get("nextPageToken")
            if not page_token:
                break

    print("\n最近即将发生的 3 条日程：")
    with build("calendar", "v3", credentials=creds) as calendar:
        # calendar 完整权限包括读取主日历中的日程。
        result = calendar.events().list(
            calendarId="primary",
            maxResults=3,
            timeMin=datetime.now(timezone.utc).isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
        for event in events:
            print(f"- {event.get('summary', '（无标题）')}")
        if not events:
            print("暂无日程")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分组授权 Google Workspace 和 YouTube")
    parser.add_argument("--service", choices=("all", "workspace", "youtube"), default="all")
    authorize(parser.parse_args().service)
