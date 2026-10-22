"""授权与运行时读取：权限收窄、token 复用、刷新失败不打开浏览器；全部使用临时文件和假的浏览器授权，不访问 Google。"""

import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import Mock, patch

from google.oauth2.credentials import Credentials
from mcp.server.mcpserver.exceptions import ToolError
from servers import google_auth as auth


def credential(*, scopes=None, granted=None):
    return Credentials(token="fake-access", refresh_token="fake-refresh", client_id="fake-client",
                       client_secret="fake-secret", token_uri="https://oauth2.googleapis.com/token",
                       scopes=list(auth.SCOPES) if scopes is None else scopes, granted_scopes=granted)


class GoogleAuthTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "token.json"
        self.youtube_path = Path(directory.name) / "youtube_token.json"
        self.flow = Mock()
        self.flow.run_local_server.return_value = credential(granted=list(auth.SCOPES))
        self.patches = [patch.object(auth, "TOKEN_PATH", self.path),
                        patch.object(auth, "YOUTUBE_TOKEN_PATH", self.youtube_path),
                        patch.object(auth.InstalledAppFlow, "from_client_secrets_file", return_value=self.flow)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_two_authorizations_use_disjoint_scopes_and_separate_token_files(self):
        def flow_for_group(config, scopes):
            flow = Mock()
            flow.run_local_server.return_value = credential(scopes=scopes, granted=scopes)
            return flow
        auth.InstalledAppFlow.from_client_secrets_file.side_effect = flow_for_group
        auth.authorize("all")
        calls = auth.InstalledAppFlow.from_client_secrets_file.call_args_list
        self.assertEqual([call.args[1] for call in calls], [auth.SCOPES, auth.YOUTUBE_SCOPES])
        self.assertFalse(set(auth.SCOPES) & set(auth.YOUTUBE_SCOPES))
        self.assertIn("https://www.googleapis.com/auth/drive.file", auth.SCOPES)
        self.assertNotIn("https://www.googleapis.com/auth/drive.file", auth.YOUTUBE_SCOPES)
        self.assertEqual(set(json.loads(self.path.read_text())["scopes"]), set(auth.SCOPES))
        self.assertEqual(set(json.loads(self.youtube_path.read_text())["scopes"]), set(auth.YOUTUBE_SCOPES))
        self.assertEqual(self.youtube_path.stat().st_mode & 0o777, 0o600)

    def test_youtube_authorization_does_not_touch_workspace_token(self):
        self.path.write_text("workspace-untouched")
        self.flow.run_local_server.return_value = credential(scopes=auth.YOUTUBE_SCOPES, granted=auth.YOUTUBE_SCOPES)
        auth.authorize("youtube")
        auth.InstalledAppFlow.from_client_secrets_file.assert_called_once_with(str(auth.CREDENTIALS_PATH), auth.YOUTUBE_SCOPES)
        self.assertEqual(self.path.read_text(), "workspace-untouched")
        self.assertEqual(set(json.loads(self.youtube_path.read_text())["scopes"]), set(auth.YOUTUBE_SCOPES))
        with patch.object(auth.Credentials, "from_authorized_user_file",
                          return_value=credential(scopes=auth.YOUTUBE_SCOPES)) as load:
            auth.get_youtube_credentials()
        load.assert_called_once_with(str(self.youtube_path))

    def test_mixed_actual_grants_are_rejected_without_writing_either_token(self):
        self.flow.run_local_server.return_value = credential(granted=[*auth.SCOPES, *auth.YOUTUBE_SCOPES])
        with self.assertRaisesRegex(RuntimeError, "混入另一组"):
            auth.authorize("workspace")
        with self.assertRaisesRegex(RuntimeError, "混入另一组"):
            auth.authorize("youtube")
        self.assertFalse(self.path.exists())
        self.assertFalse(self.youtube_path.exists())

    def test_youtube_only_command_does_not_start_workspace_authorization(self):
        with patch.object(auth, "_authorize") as authorize:
            auth.authorize("youtube")
        authorize.assert_called_once()
        self.assertEqual(authorize.call_args.args[0], auth.YOUTUBE_SCOPES)

    def test_drive_is_readonly_plus_file_without_documents_full_scope(self):
        self.assertIn("https://www.googleapis.com/auth/drive.readonly", auth.SCOPES)
        self.assertIn("https://www.googleapis.com/auth/drive.file", auth.SCOPES)
        self.assertNotIn("https://www.googleapis.com/auth/drive", auth.SCOPES)
        self.assertNotIn("https://www.googleapis.com/auth/documents", auth.SCOPES)

    def test_old_extra_privileges_need_new_consent(self):
        self.path.write_text("old-token")
        old = credential(scopes=[*auth.SCOPES, "https://www.googleapis.com/auth/drive"])
        with patch.object(auth.Credentials, "from_authorized_user_file", return_value=old):
            with self.assertRaisesRegex(auth.GoogleAuthRequired, "权限范围"):
                auth.get_credentials()
        self.flow.run_local_server.assert_not_called()  # 运行时只报错，不打开浏览器
        auth.authorize("workspace")
        self.flow.run_local_server.assert_called_once_with(port=0, prompt="consent")
        saved = json.loads(self.path.read_text())
        self.assertEqual(set(saved["scopes"]), set(auth.SCOPES))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_new_token_with_broad_actual_grants_is_rejected_without_overwriting_old_file(self):
        self.path.write_text("old-token-untouched")
        self.flow.run_local_server.return_value = credential(
            granted=[*auth.SCOPES, "https://www.googleapis.com/auth/documents"])
        with self.assertRaisesRegex(RuntimeError, "移除此应用的旧授权"):
            auth.authorize("workspace")
        self.assertEqual(self.path.read_text(), "old-token-untouched")

    def test_partial_actual_grant_is_not_mistaken_for_requested_scopes(self):
        self.flow.run_local_server.return_value = credential(granted=auth.SCOPES[:-1])
        with self.assertRaisesRegex(RuntimeError, "未包含"):
            auth.authorize("workspace")
        self.assertFalse(self.path.exists())

    def test_compatible_cached_credentials_reuse_without_consent(self):
        self.path.write_text("cached-token")
        cached = credential()
        with patch.object(auth.Credentials, "from_authorized_user_file", return_value=cached):
            self.assertIs(auth.get_credentials(), cached)
        self.flow.run_local_server.assert_not_called()

    def test_authorize_always_asks_for_new_consent_even_with_a_working_token(self):
        # 否则旧的（例如测试模式下 7 天过期的）refresh token 会被原样保留。
        self.path.write_text("working-token")
        with patch.object(auth.Credentials, "from_authorized_user_file", return_value=credential()) as load:
            auth.authorize("workspace")
        load.assert_not_called()
        self.flow.run_local_server.assert_called_once_with(port=0, prompt="consent")

    def test_runtime_never_opens_a_browser_and_says_how_to_fix(self):
        expired = credential()
        expired.token = None  # 没有 access token 就需要刷新
        cases = {
            "missing": patch.object(auth.Credentials, "from_authorized_user_file", side_effect=FileNotFoundError()),
            "corrupt": patch.object(auth.Credentials, "from_authorized_user_file", side_effect=ValueError()),
            "refresh failed": patch.object(auth.Credentials, "from_authorized_user_file", return_value=expired),
        }
        for name, loader in cases.items():
            with self.subTest(name), loader, \
                    patch.object(Credentials, "refresh", side_effect=auth.RefreshError("invalid_grant")):
                with self.assertRaises(auth.GoogleAuthRequired) as error:
                    auth.get_credentials()
                self.assertIn(auth.AUTHORIZE_COMMAND, str(error.exception))
                self.assertIn("没有执行", str(error.exception))
        self.flow.run_local_server.assert_not_called()
        auth.InstalledAppFlow.from_client_secrets_file.assert_not_called()
        self.assertTrue(issubclass(auth.GoogleAuthRequired, ToolError))  # MCP 只把 ToolError 的文字交给模型

    def test_refreshed_token_is_saved_atomically_and_privately(self):
        expired = credential()
        expired.token = None

        def refreshed(self, request):
            self.token = "new-access"
        with patch.object(auth.Credentials, "from_authorized_user_file", return_value=expired), \
                patch.object(Credentials, "refresh", refreshed):
            self.assertEqual(auth.get_credentials().token, "new-access")
        self.assertEqual(json.loads(self.path.read_text())["token"], "new-access")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["token.json"])  # 没有遗留临时文件

    def test_returned_scope_string_and_empty_grants_are_handled(self):
        creds = credential(granted=" ".join(auth.SCOPES))
        self.assertTrue(auth._compatible_scopes(creds))
        self.assertFalse(auth._compatible_scopes(credential(granted=[])))

    def test_desktop_authorization_url_omits_incremental_authorization_parameter(self):
        from google_auth_oauthlib.flow import InstalledAppFlow
        flow = InstalledAppFlow.from_client_config({"installed": {
            "client_id": "fake-client.apps.googleusercontent.com", "client_secret": "fake-secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": ["http://localhost"],
        }}, auth.SCOPES)
        flow.redirect_uri = "http://localhost:12345/"
        url, _ = flow.authorization_url(prompt="consent")
        query = parse_qs(urlparse(url).query)
        self.assertNotIn("include_granted_scopes", query)
        self.assertEqual(query["prompt"], ["consent"])
        self.assertEqual(query["access_type"], ["offline"])
