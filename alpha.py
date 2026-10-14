"""兼容旧命令：python alpha.py [--voice]。实际入口在 cli.py。"""

from apps.cli import main, run


if __name__ == "__main__":
    run()
