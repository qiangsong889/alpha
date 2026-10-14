"""第 4/5 步：流式语音播放、打断和首段音频延迟测量。

运行：alpha/.venv/bin/python alpha/realtime.py
.env：OPENAI_API_KEY；可选 OPENAI_REALTIME_MODEL、OPENAI_REALTIME_VOICE。
收到 speech_started 时清空播放缓冲区，并截断未听完的回答。
"""

import asyncio
import base64
import json
import os
import time
import sounddevice as sd  # type: ignore[import-untyped]
import httpx2
from mcp.types import TextContent
from integrations.mcp_client import load_ha_catalog, call_ha_tool, format_error

from voice.audio import (
    StreamingPlayer, audio_device, play_audio, MicrophoneUnavailable,
    SpeechActivity, SpeechCandidate, is_meaningful,
)

from dotenv import load_dotenv

from core.paths import ENV_FILE
from openai import AsyncOpenAI
from openai.types.realtime.session_update_event_param import Session
from openai.resources.realtime.realtime import AsyncRealtimeConnection


HA_DEVICE_TOOLS = {
    "intent__HassTurnOn", "intent__HassTurnOff",
    "light__HassLightSet", "homeassistant__GetLiveContext",
}


def session_config(voice: str, ha_tools: list | None = None, assist_prompt: str = "") -> Session:
    # 本函数只返回配置；等 main 调用 session.update 后才发给服务端。
    # 自动回答和自动打断关闭后，由客户端根据 VAD、转写和工具状态决定下一步。
    return {
        "type": "realtime",
        "instructions": (
            "你是一个中文语音助手。无论用户说中文还是英文，始终只用普通话回答，"
            "用一两句简短口语回答，不使用 Markdown、列表或表情符号。"
            "用户打断你时，停止原来的回答，听取并回应用户的新问题。"
            "查询设备状态或控制设备时必须使用提供的工具，不猜测状态，不虚构成功。"
            "工具失败时简短说明失败。设备目标不明确时先询问。"
            + ("\nHome Assistant Assist prompt：\n" + assist_prompt if assist_prompt
               else "\n家居设备暂时不可用。")
        ),
        "tools": ha_tools or [],
        "tool_choice": "auto",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "gpt-4o-mini-transcribe", "language": "zh"},
                "turn_detection": {
                    "type": "semantic_vad",
                    "eagerness": "auto",
                    "create_response": False,
                    "interrupt_response": False,
                },
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": voice,
            },
        },
    }


async def text_to_audio(connection: AsyncRealtimeConnection, text: str) -> bytes:
    # 文字测试路径：先添加一条用户消息，再请求模型回答，最后拼成完整音频返回。
    # 实时麦克风路径则边收到音频边交给 StreamingPlayer，不需要等完整回答。
    print(f"你：{text}")
    started = time.perf_counter()
    await connection.conversation.item.create(item={
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    })
    await connection.response.create()
    chunks: list[bytes] = []
    transcript: list[str] = []
    async with asyncio.timeout(60):
        async for event in connection:
            if event.type == "error":
                raise RuntimeError(f"{event.error.code}: {event.error.message}")
            if event.type == "response.output_audio.delta":
                chunk = base64.b64decode(event.delta)
                if chunk:
                    if not chunks:
                        print(f"[延迟] 文字发送到首个音频包：{time.perf_counter() - started:.2f}s")
                    chunks.append(chunk)
            elif event.type == "response.output_audio_transcript.delta":
                transcript.append(event.delta)
            elif event.type == "response.done":
                usage = event.response.usage
                print(f"[usage] {usage.model_dump_json(exclude_none=True) if usage is not None else '未返回'}")
                if event.response.status != "completed":
                    raise RuntimeError(f"回答未完成：{event.response.status}，{event.response.status_details}")
                print(f"模型：{''.join(transcript) or '（未收到文字转录）'}")
                print(f"[耗时] 完整回答接收：{time.perf_counter() - started:.2f}s")
                if not chunks:
                    raise RuntimeError("回答完成但没有收到音频")
                return b"".join(chunks)
    raise RuntimeError("连接关闭，未收到完整回答")


async def send_microphone(connection: AsyncRealtimeConnection, activity: SpeechActivity) -> None:
    # 外层只负责设备故障重试；一次真实收音由 send_microphone_once 管理。
    while True:
        try:
            await send_microphone_once(connection, activity)
        except (sd.PortAudioError, MicrophoneUnavailable) as exc:
            print(f"[麦克风断开] {exc}。会话保留，1 秒后尝试重新打开。")
            # 丢弃断开前未完成的输入，避免与恢复后的句子拼接。
            await connection.input_audio_buffer.clear()
            activity.reset()
            await asyncio.sleep(1)


async def send_microphone_once(connection: AsyncRealtimeConnection, activity: SpeechActivity) -> None:
    loop = asyncio.get_running_loop()
    # 麦克风线程产生音频，异步发送任务消费；容量 100 × 20ms，防止网络慢时无限积压。
    audio_queue: asyncio.Queue[bytes | Exception] = asyncio.Queue(maxsize=100)
    accepting = True

    def enqueue(chunk: bytes, status: str) -> None:
        nonlocal accepting
        if not accepting:
            return
        if status:
            print(f"[麦克风] {status}")
        activity.observe(chunk)
        if audio_queue.full():
            accepting = False
            audio_queue.get_nowait()
            audio_queue.put_nowait(RuntimeError("音频发送积压超过 2 秒，请检查网络"))
        else:
            audio_queue.put_nowait(chunk)

    def callback(indata, frames, time_info, status):
        # sounddevice 回调在音频线程执行，不直接操作 asyncio.Queue。
        chunk = indata.astype("<i2", copy=False).tobytes()
        loop.call_soon_threadsafe(enqueue, chunk, str(status) if status else "")

    try:
        try:
            stream = sd.InputStream(
                samplerate=24000, channels=1, dtype="int16",
                blocksize=480, callback=callback,  # 每块 20 ms
                device=audio_device("REALTIME_INPUT_DEVICE"),
            )
        except ValueError as exc:
            raise MicrophoneUnavailable(str(exc)) from exc
        with stream:
            # 普通 with 打开/关闭本地设备；它内部可以 await，不需要因此改成 async with。
            print("麦克风已开启，请用中文说话。Ctrl+C 退出，建议戴耳机。")
            while True:
                try:
                    # 队列空时当前发送任务暂停；模型接收任务和播放仍然可以工作。
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=1)
                except TimeoutError as exc:
                    raise MicrophoneUnavailable("1 秒内没有收到麦克风音频") from exc
                if not stream.active:
                    raise MicrophoneUnavailable("麦克风流已停止")
                if isinstance(chunk, Exception):
                    raise chunk
                # base64 让原始音频能放进 JSON 消息，这一步没有做语音识别。
                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(chunk).decode("ascii")
                )
    finally:
        accepting = False


async def execute_ha_calls(
    connection: AsyncRealtimeConnection, calls: list, *,
    available_tools: set[str], url: str, http_client: httpx2.AsyncClient, log,
) -> None:
    for call in calls:
        # 模型只是提出调用请求；客户端检查白名单、解析参数，再真正请求 HA 的 MCP server。
        log(f"[工具] {call.name} 参数：{call.arguments}")
        started = time.perf_counter()
        try:
            if call.name not in HA_DEVICE_TOOLS or call.name not in available_tools:
                raise ValueError(f"工具不在可用白名单中：{call.name}")
            arguments = json.loads(call.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是对象")
            result = await call_ha_tool(call.name, arguments, url=url, http_client=http_client)
            text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
            is_error = result.is_error
        except Exception as exc:
            text = f"工具执行失败：{format_error(exc)}。操作可能已生效，写入操作不要盲目重试。"
            is_error = True
        log(f"[工具结果] is_error={is_error} {text[:100]} ｜ {time.perf_counter() - started:.2f}s")
        # call_id 让模型知道这个输出属于哪一次调用；错误也回传，交给模型说明结果。
        await connection.conversation.item.create(item={
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps({"is_error": is_error, "content": text}, ensure_ascii=False),
        })


async def receive_microphone_answers(
    connection: AsyncRealtimeConnection, player: StreamingPlayer, activity: SpeechActivity,
    *, ha_tools: set[str] | None = None, ha_url: str = "",
    http_client: httpx2.AsyncClient | None = None,
) -> None:
    # 一个长期任务负责接收各种事件，局部变量跨事件保存“当前说话和回答”的状态。
    # 发声开始、音频提交、转写完成是不同事件，不能当成同一件事。
    threshold_ms = int(os.environ.get("INTERRUPT_MS", "400"))
    if not 300 <= threshold_ms <= 500:
        raise ValueError("INTERRUPT_MS 必须在 300 到 500 之间")
    transcript: list[str] = []
    speech_stopped_at: float | None = None
    first_audio = True
    response_id: str | None = None
    interrupted_responses: set[str] = set()
    candidate: SpeechCandidate | None = None
    timer: asyncio.Task | None = None
    response_pending = False
    tool_busy = False
    tool_continuation: SpeechCandidate | None = None
    continuation_ready = False
    handled_calls: set[str] = set()
    deleted_items: set[str] = set()
    receive_started = time.perf_counter()

    def log(message: str) -> None:
        print(f"[{time.perf_counter() - receive_started:.3f}] {message}", flush=True)

    async def delete_noise(item_id: str) -> None:
        if item_id in deleted_items:
            return
        await connection.conversation.item.delete(item_id=item_id)
        deleted_items.add(item_id)
        log(f"客户端发送 conversation.item.delete（无意义条目 {item_id}）")

    async def maybe_respond() -> None:
        # 只有有效文字和已提交音频都到齐，而且没有旧回答/工具正在处理，才请求新回答。
        nonlocal response_pending, continuation_ready
        if (candidate is not None and candidate.committed and candidate.meaningful is True
                and not candidate.responded and response_id is None and not response_pending
                and not tool_busy):
            candidate.responded = True
            response_pending = True
            log(f"客户端发送 response.create（用户条目 {candidate.item_id}）")
            await connection.response.create()
            continuation_ready = False
        elif (continuation_ready and not tool_busy and response_id is None and not response_pending
              and (candidate is tool_continuation or candidate is None
                   or candidate.meaningful is False)):
            continuation_ready = False
            response_pending = True
            log("客户端发送 response.create（工具结果已回传）")
            await connection.response.create()

    async def confirm(target: SpeechCandidate, reason: str) -> None:
        # 有意义的转写确认用户要说话后，才清空本地缓冲并取消旧回答。
        if candidate is not target or target.confirmed:
            return
        target.confirmed = True
        item = player.clear()
        log(f"确认结果：是打断 → 清空播放缓冲区（{reason}）")
        if response_id is not None:
            interrupted_responses.add(response_id)
            log(f"客户端发送 response.cancel（{response_id}）")
            await connection.response.cancel(response_id=response_id)
        if item is not None:
            # 截断服务端历史到估计已播放位置，避免模型把用户没听到的部分当作已说完。
            item_id, content_index, played_ms = item
            await connection.conversation.item.truncate(
                item_id=item_id, content_index=content_index, audio_end_ms=played_ms,
            )
        await maybe_respond()

    def pause_candidate(target: SpeechCandidate) -> None:
        # 本地持续人声只触发暂停，保留旧回答；若后续识别成噪音，还能继续播放。
        if (candidate is not target or target.paused or target.confirmed
                or target.meaningful is not None):
            return
        target.paused = True
        player.pause()
        log(f"[暂停] 本地 VAD 人声 {activity.voiced_ms}ms，保留缓冲区，等待转录")

    async def confirm_after_delay(target: SpeechCandidate) -> None:
        while candidate is target and not target.stopped and not target.confirmed:
            if activity.voiced_ms >= threshold_ms:
                pause_candidate(target)
                return
            # 服务端尚未报告 speech_stopped，也可以恢复短噪音后的音量。
            recent_voice = time.perf_counter() - activity.last_voice_at < 0.15
            if player.gain != (0.2 if recent_voice else 1.0):
                log("继续确认 → 降低音量" if recent_voice else
                    "确认结果：暂未确认打断 → 恢复音量，仍等待文字")
            player.duck(recent_voice)
            await asyncio.sleep(0.02)

    async def run_tools(calls: list, origin: SpeechCandidate | None) -> None:
        # 此函数会作为后台任务运行，因此等待 HA 时接收循环仍可处理用户打断。
        nonlocal tool_busy, continuation_ready, tool_continuation
        try:
            if http_client is None:
                raise RuntimeError("HA HTTP 客户端未配置")
            await execute_ha_calls(
                connection, calls, available_tools=ha_tools or set(),
                url=ha_url, http_client=http_client, log=log,
            )
        finally:
            tool_busy = False
        # 执行工具期间仍接收语音；有新的有效输入时优先处理新输入。
        tool_continuation = origin
        continuation_ready = True
        await maybe_respond()

    try:
        async with asyncio.TaskGroup() as timers:
            # timers 统一管理人声确认和工具调用任务；异常退出时清理同组任务。
            async for event in connection:
                # 每次收到一个服务端事件，按 type 更新状态或处理一小段音频。
                if event.type == "error":
                    # 取消请求可能与 response.done 交错，此错误无需结束会话。
                    if event.error.code == "response_cancel_not_active":
                        log(f"error：{event.error.code}（回答已结束，忽略取消竞态）")
                        continue
                    raise RuntimeError(f"{event.error.code}: {event.error.message}")
                if event.type == "input_audio_buffer.speech_started":
                    log(f"speech_started（item_id={event.item_id}）")
                    if timer is not None:
                        timer.cancel()
                    candidate = SpeechCandidate(event.item_id, event.audio_start_ms)
                    player.duck(True)
                    log(f"降低音量到 20%，开始确认（阈值 {threshold_ms}ms）")
                    timer = timers.create_task(confirm_after_delay(candidate))
                    speech_stopped_at = None
                elif event.type == "input_audio_buffer.speech_stopped":
                    speech_stopped_at = time.perf_counter()
                    duration = (
                        f"持续了 {(event.audio_end_ms - candidate.audio_start_ms) / 1000:.2f}s"
                        if candidate is not None and candidate.item_id == event.item_id
                        else "未匹配到开始事件"
                    )
                    log(f"speech_stopped（{duration}，item_id={event.item_id}）")
                    if candidate is not None and candidate.item_id == event.item_id:
                        candidate.stopped = True
                        if timer is not None:
                            timer.cancel()
                        if activity.voiced_ms >= threshold_ms:
                            pause_candidate(candidate)
                        elif not candidate.confirmed and not candidate.paused:
                            player.duck(False)
                            log("确认结果：暂不是打断 → 恢复音量（后续非空文字仍可确认）")
                elif event.type == "input_audio_buffer.committed":
                    log(f"input_audio_buffer.committed（item_id={event.item_id}）")
                    if candidate is not None and candidate.item_id == event.item_id:
                        candidate.committed = True
                        await maybe_respond()
                elif event.type == "conversation.item.input_audio_transcription.completed":
                    text = event.transcript.strip()
                    log(f"识别出的文字：{text!r}（input_audio_transcription.completed，item_id={event.item_id}）")
                    if candidate is not None and candidate.item_id == event.item_id:
                        candidate.meaningful = is_meaningful(text)
                        if timer is not None:
                            timer.cancel()
                        if candidate.meaningful is False:
                            log(f'[忽略] 识别出的文字："{text}"，不回应')
                            player.resume()
                            log("[恢复] 只是噪音，继续播放")
                            await delete_noise(event.item_id)
                        else:
                            await confirm(candidate, "识别到有意义的文字")
                        await maybe_respond()
                    elif not is_meaningful(text):
                        # 迟到的噪音转录也应删除，但不能改变当前新一轮的播放状态。
                        await delete_noise(event.item_id)
                elif event.type == "conversation.item.deleted":
                    log(f"conversation.item.deleted（item_id={event.item_id}）")
                elif event.type == "conversation.item.input_audio_transcription.failed":
                    log(f"input_audio_transcription.failed：{event.error.message}")
                    if candidate is not None and candidate.item_id == event.item_id:
                        candidate.meaningful = False
                        player.resume()
                        log("[恢复] 转录失败，保留旧回答继续播放，不生成新回答")
                elif event.type == "response.created":
                    source = "客户端已请求创建" if response_pending else "未记录到客户端创建请求"
                    log(f"response.created（{event.response.id}；{source}）")
                    response_id = event.response.id
                    response_pending = False
                    first_audio = True
                    transcript = []
                    # 新回答建立时，尚在说话的已确认用户需要取消它。
                    if candidate is not None and candidate.confirmed and not candidate.stopped:
                        interrupted_responses.add(response_id)
                        await connection.response.cancel(response_id=response_id)
                elif event.type == "response.output_audio.delta":
                    # 一小块音频解码后入队；player.callback 在音频线程真正把它送到扬声器。
                    if event.response_id in interrupted_responses:
                        continue
                    chunk = base64.b64decode(event.delta)
                    if chunk:
                        if first_audio:
                            if speech_stopped_at is not None:
                                log(f"speech_stopped 到首段音频：{time.perf_counter() - speech_stopped_at:.2f}s")
                            first_audio = False
                        player.append(chunk, event.item_id, event.content_index)
                elif event.type == "response.output_audio_transcript.delta":
                    if event.response_id not in interrupted_responses:
                        transcript.append(event.delta)
                elif event.type == "response.done":
                    log(f"response.done（{event.response.id}，{event.response.status}）")
                    usage = event.response.usage
                    log(f"[usage] {usage.model_dump_json(exclude_none=True) if usage is not None else '未返回'}")
                    calls = [item for item in event.response.output if item.type == "function_call"]
                    eligible_calls = (
                        calls if event.response.status == "completed"
                        and event.response.id not in interrupted_responses else []
                    )
                    if event.response.id not in interrupted_responses:
                        if event.response.status == "completed":
                            if not calls:
                                log(f"模型：{''.join(transcript) or '（未收到文字转录）'}")
                        else:
                            log(f"回答状态：{event.response.status}")
                    if response_id == event.response.id:
                        response_id = None
                    interrupted_responses.discard(event.response.id)
                    new_calls = [call for call in eligible_calls if call.call_id not in handled_calls]
                    if new_calls:
                        handled_calls.update(call.call_id for call in new_calls)
                        tool_busy = True
                        # 不在接收循环里 await 整次工具执行，否则期间收不到后续打断消息。
                        timers.create_task(run_tools(new_calls, candidate))
                    await maybe_respond()
            raise RuntimeError("Realtime 连接已关闭")
    finally:
        if timer is not None:
            timer.cancel()
        player.resume()


async def main() -> None:
    load_dotenv(ENV_FILE)
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise SystemExit("请在 alpha/.env 中设置 OPENAI_API_KEY")
    model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")
    voice = os.environ.get("OPENAI_REALTIME_VOICE", "marin")
    ha_root = os.environ.get("HA_URL", "").strip()
    ha_token = os.environ.get("HA_TOKEN", "").strip()
    ha_url = ha_root.rstrip("/") + "/api/mcp"

    async with AsyncOpenAI() as client, httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {ha_token}"},
        timeout=httpx2.Timeout(30.0, read=300.0),
    ) as http_client:
        # HA 目录只在启动时读取；模型看到描述后选择工具，实际执行通过 execute_ha_calls。
        ha_catalog = await load_ha_catalog(ha_url, http_client) if ha_root and ha_token else None
        ha_tools = []
        assist_prompt = ""
        if ha_catalog is not None:
            names: set[str] = set()
            for tool in ha_catalog[0]:
                if tool["name"] in HA_DEVICE_TOOLS:
                    if tool["name"] in names:
                        raise RuntimeError(f"HA 工具重名：{tool['name']}")
                    names.add(tool["name"])
                    ha_tools.append({
                        "type": "function", "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    })
            # load_ha_catalog 的提示含 Alpha 规则；仅复用 HA 的 Assist 内容。
            assist_prompt = ha_catalog[1][0]["text"].partition("Home Assistant Assist prompt：\n")[2]
        print("[HA 工具] " + ("、".join(tool["name"] for tool in ha_tools) or "不可用"))
        started = time.perf_counter()
        async with client.realtime.connect(model=model) as connection:
            print(f"[连接] {model}，WebSocket 建立耗时 {time.perf_counter() - started:.2f}s")
            configured_at = time.perf_counter()
            await connection.session.update(session=session_config(voice, ha_tools, assist_prompt))
            # update() 只发送配置；session.updated 才表示服务端已确认。
            async with asyncio.timeout(30):
                async for event in connection:
                    if event.type == "error":
                        raise RuntimeError(f"{event.error.code}: {event.error.message}")
                    if event.type == "session.created":
                        print(f"[会话] {event.session.id}")
                    elif event.type == "session.updated":
                        print(f"[配置] 服务端已确认，耗时 {time.perf_counter() - configured_at:.2f}s")
                        print(f"中文指令 ｜ PCM16 单声道 24kHz ｜ voice={voice} ｜ semantic_vad")
                        break
                else:
                    raise RuntimeError("连接关闭，未收到 session.updated")
            player = StreamingPlayer()
            activity = SpeechActivity()
            async with asyncio.TaskGroup() as tasks:
                # 三个协程并发协作：持续发送录音、持续接收事件、维持播放器输出流。
                # create_task 安排执行并马上返回，不会逐个等这些长期任务结束。
                tasks.create_task(send_microphone(connection, activity))
                tasks.create_task(receive_microphone_answers(
                    connection, player, activity,
                    ha_tools={tool["name"] for tool in ha_tools},
                    ha_url=ha_url, http_client=http_client,
                ))
                # run 打开并维持音频设备；实际播放由音频库调用 player.callback 完成。
                tasks.create_task(player.run())


if __name__ == "__main__":
    try:
        # 从同步脚本进入异步程序：创建事件循环并执行 main，退出时再关闭循环。
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已退出")
