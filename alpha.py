"""兼容旧命令：python alpha.py [--voice]。实际入口在 cli.py。"""

from apps.cli import main, run


if __name__ == "__main__":
    # 直接运行本文件时进入 CLI；被其他模块 import 时不会开始等待输入。
    # run 负责解析参数和启动事件循环，具体异步对话流程在 apps.cli.main。
    run()
