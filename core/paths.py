"""项目里所有文件路径的唯一来源。代码移动位置时，只需要改这里。"""

from pathlib import Path

# __file__ 指向 core/paths.py；resolve 得到绝对路径，两次 parent 得到 alpha 目录。
# 所有模块复用这里的路径，终端从哪里启动不会改变 .env、凭据和 Python 解释器的位置。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
OPERATIONS_DB = PROJECT_ROOT / "data" / "operations.db"
# 对话存档：只有正式入口（brain.server、CLI）显式开启，测试和 eval 不会写入。
ARCHIVE_DB = PROJECT_ROOT / "data" / "archive.db"

# Google OAuth：客户端凭据和授权后保存的令牌（都不进 git）。
GOOGLE_CREDENTIALS = PROJECT_ROOT / "credentials.json"
GOOGLE_TOKEN = PROJECT_ROOT / "token.json"
GOOGLE_YOUTUBE_TOKEN = PROJECT_ROOT / "youtube_token.json"
