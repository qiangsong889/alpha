"""项目里所有文件路径的唯一来源。代码移动位置时，只需要改这里。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# Google OAuth：客户端凭据和授权后保存的令牌（都不进 git）。
GOOGLE_CREDENTIALS = PROJECT_ROOT / "credentials.json"
GOOGLE_TOKEN = PROJECT_ROOT / "token.json"
