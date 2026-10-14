"""Alpha 终端输入输出：文字模式或录音、转写和朗读模式。"""

import argparse
import asyncio
import logging
import os
import time

from brain.agent import AlphaBrain
from integrations.mcp_client import format_error

logger = logging.getLogger("alpha.cli")

async def speak(text: str) -> None:
    """通过 macOS say 的标准输入朗读，避免把回答当命令参数。"""
    process = await asyncio.create_subprocess_exec(
        "say", "-v", os.environ.get("ALPHA_VOICE", "Meijia"),
        stdin=asyncio.subprocess.PIPE,
    )
    try:
        await process.communicate(text.encode("utf-8"))
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"朗读程序退出码：{process.returncode}")


async def main(voice: bool = False) -> None:
    if voice:
        from voice.listen import _get_model
        _get_model()
    brain = AlphaBrain()
    try:
        await brain.start()
        print("Alpha 已启动。输入 exit / quit / 退出 结束对话。")
        while True:
            timings: list[tuple[str, float]] = []
            try:
                if voice:
                    from voice.listen import record, transcribe
                    audio = await asyncio.to_thread(record)
                    turn_started = time.perf_counter()
                    text = await asyncio.to_thread(transcribe, audio)
                    timings.append(("转文字", time.perf_counter() - turn_started))
                    user_text = text.strip()
                    print(f"识别文字：{user_text or '（未识别到文字）'}")
                    print(f"录音时长：{audio.size / 16000:.2f} 秒")
                else:
                    user_text = (await asyncio.to_thread(input, "你：")).strip()
                    turn_started = time.perf_counter()
            except EOFError:
                break
            except Exception as exc:
                print(f"输入失败：{format_error(exc)}")
                continue
            if user_text.lower() in {"exit", "quit", "退出"}:
                break
            if not user_text:
                continue
            try:
                answer = await brain.ask(user_text)
                print(f"Alpha：{answer}")
            except Exception as exc:
                print(f"本轮失败：{format_error(exc)}")
                print("本轮未自动重试；已执行的操作可能仍然生效。")
                continue
            timings.extend(brain.last_timings)
            total_label = "朗读开始前总计" if voice else "总计"
            timings.append((total_label, time.perf_counter() - turn_started))
            logger.info("[耗时] %s", " ｜ ".join(
                f"{label} {seconds:.2f}s" for label, seconds in timings
            ))
            if voice and answer:
                try:
                    await speak(answer)
                except Exception as exc:
                    print(f"朗读失败：{format_error(exc)}")
    finally:
        await brain.close()


def run() -> None:
    parser = argparse.ArgumentParser(description="Alpha 家居与个人任务助手")
    parser.add_argument("--voice", action="store_true", help="录音输入并朗读回答（macOS）")
    options = parser.parse_args()
    # 仅入口配置输出；brain 被其他语音层导入时不修改全局 logging。
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s %(message)s"))
    logging.getLogger("alpha").addHandler(handler)
    logging.getLogger("alpha").setLevel(logging.INFO)
    logging.getLogger("alpha").propagate = False
    try:
        asyncio.run(main(voice=options.voice))
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    run()
