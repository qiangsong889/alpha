"""获取 Google OAuth 凭据，并验证 Tasks 和 Calendar API。

运行：alpha/.venv/bin/python alpha/google_auth.py
将桌面应用的 OAuth 配置放在同目录 credentials.json 中。
首次运行会打开浏览器授权，之后复用同目录 token.json。
"""

from datetime import datetime, timezone
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build  # type: ignore[import-untyped]


SCOPES = [
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/calendar.events",
]

TOKEN_PATH = Path(__file__).with_name("token.json")
CREDENTIALS_PATH = Path(__file__).with_name("credentials.json")


def get_credentials() -> Credentials:
    """复用或刷新已有凭据；没有可用凭据时，通过浏览器授权。"""
    creds = None
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
        except ValueError:
            # 文件内容无效时，重新走授权流程。
            creds = None

    # 已保存的 token 若缺少本次需要的权限，需要重新授权。
    if creds is not None and not creds.has_scopes(SCOPES):
        creds = None

    if creds is not None and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            # refresh token 已失效或被撤销时，重新授权。
            creds = None

    if creds is None or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_PATH), SCOPES
        )
        # 临时启动本地服务接收授权回调；0 表示使用空闲端口。
        creds = flow.run_local_server(port=0)

    # SDK 声明可能返回多种凭据；本脚本只接受普通用户 OAuth 凭据。
    if not isinstance(creds, Credentials):
        raise TypeError("授权返回的不是 Google 用户 OAuth 凭据")

    if not creds.has_scopes(SCOPES):
        raise RuntimeError("授权未包含 Tasks 和 Calendar events 所需的全部权限")

    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    TOKEN_PATH.chmod(0o600)
    return creds


def main() -> None:
    creds = get_credentials()

    print("任务列表：")
    with build("tasks", "v1", credentials=creds) as tasks:
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
        # calendar.events 权限支持读取日程；这里不调用 calendarList。
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
    main()
