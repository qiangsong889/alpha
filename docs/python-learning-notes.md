# 从 Alpha 项目理解 Python：个人学习笔记

这份笔记整理了你在开发 Alpha 时逐步理解的结论，以及容易混淆的知识点。例子优先采用项目里的录音、播放、WebSocket、MCP 和 Google 工具。

标注“你的理解成立”的地方，是可以保留的结论；标注“需要补充”的地方，是让结论更准确的边界。

建议阅读顺序：先看第 1～8 节理解异步，再看第 9～12 节理解音频和服务，最后按需要查阅其他章节。

## 1. 同步、异步、并发、线程

### 你的理解成立：等待异步操作时，其他已安排的任务可以继续运行

但“同步”不表示同时起步，“异步”也不表示自动开线程。

| 概念 | 在这个项目里怎么理解 |
|---|---|
| 同步调用 | 当前调用要等这段操作完成，才能继续往下执行 |
| 异步等待 | 当前任务可以暂停等待，把执行机会交给其他任务 |
| 并发 | 多项工作在同一段时间里交错推进，不要求同时执行 Python 指令 |
| 并行 | 多项工作在同一时刻实际执行，例如不同 CPU 核心上的进程 |
| 线程 | 操作系统调度的执行单元；一个进程可以有多个线程 |
| asyncio 任务 | 事件循环调度的协程；创建任务本身不等于创建线程 |

你的 Alpha 同时涉及几种机制：

```text
主进程的事件循环线程：录音发送任务、Qwen 接收任务、PersonalWorker 等
sounddevice 的音频线程：录音和播放回调
to_thread 的工作线程：运行 input、Whisper 转写等同步函数
个人 MCP server 子进程：由 stdio_client 启动，有自己的执行环境
```

“后台运行”只是说它与当前流程共同推进，并不能据此判断用了线程还是进程。

## 2. async def、协程对象、await

### 你的理解成立：直接调用普通的 async 函数，得到的是协程对象

```python
async def get_answer():
    return "你好"

coroutine = get_answer()  # 创建协程对象，此时还没有执行函数体
```

这里说的是通常的 `async def` 协程函数。包含 `yield` 的异步生成器是另一种类型，现阶段可以先分开记。

`async` 表示这是可以使用异步等待的函数，不是“让函数变成同步”，也不保证其中所有代码都不会阻塞。

### 你的理解成立：await 拿到结果以后，当前函数才执行下一行

```python
response = await client.get(url)
print("hello")
```

这段代码的顺序是：

```text
开始请求 → 当前任务等待 → 请求成功并取得结果 → print("hello")
```

等待期间，其他已经安排的任务可以运行。请求失败时，`await` 会抛异常，不会正常赋值后继续打印。

需要补充：`await` 不一定每次都会真的暂停。如果结果已经准备好，它可能直接继续执行。

### 不写 await 会怎样？

```python
coroutine = get_answer()
```

只保存了协程对象。你还需要等待它，或把它安排成任务：

```python
answer = await coroutine
```

或者：

```python
task = asyncio.create_task(coroutine)
answer = await task
```

不要把 Python 协程完全等同于 JavaScript Promise：普通 Python 协程调用一般还没有开始执行函数体；JavaScript 的 async 函数调用会开始运行，并返回 Promise。

## 3. asyncio.run：从普通脚本进入异步程序

### 你的理解成立：main() 得到协程对象，可以交给 asyncio.run 执行

```python
import asyncio

async def main():
    answer = await get_answer()
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
```

`asyncio.run` 负责创建事件循环、运行入口协程，并在结束时清理和关闭循环。

不是每个 async 函数都分别调用一次 `asyncio.run`。通常只在程序最外层调用一次；里面使用 `await`、`create_task` 或 `TaskGroup`。

判断依据也不是“有没有写 await”：

- 调用协程需要某种执行它的方式。
- `asyncio.create_task` 需要已有正在运行的事件循环。
- `asyncio.to_thread(...)` 返回协程，也需要被等待或安排执行。
- 在已经有事件循环的环境里，例如某些 Notebook、Web 框架，应使用该环境提供的异步入口，不再嵌套调用 `asyncio.run`。

## 4. create_task：安排运行，不等待完成

### 你的理解成立：运行到这一行，不会等待 worker.run 结束

```python
worker_task = asyncio.create_task(
    worker.run(),
    name="qwen-personal-worker",
)
print("继续往下执行")
```

分开看：

```text
worker.run()       → 创建协程对象
create_task(...)   → 把它安排给事件循环，返回 Task 对象
name=...          → 便于调试的任务名字，不影响它调用哪个函数
```

默认调度方式下，不保证新任务在下一行之前已经开始执行。当前任务让出控制权时，新任务才有机会运行。

可以以后再等结果：

```python
result = await task
```

但 `worker.run()` 是长期任务，一般等到退出时再取消，而不是创建后立刻等待它结束。

### 需要补充：create_task 不是创建线程

普通 asyncio 任务在同一个事件循环线程中协作运行。如果某个任务执行阻塞操作，其他任务仍可能被卡住。

```python
async def bad_example():
    time.sleep(5)  # 同步阻塞，事件循环线程也停在这里

async def better_example():
    await asyncio.sleep(5)  # 暂停当前任务，允许其他任务推进
```

实际代码里要先 `import time`、`import asyncio`。

## 5. to_thread：把同步函数交给工作线程

### 你的理解成立：await asyncio.to_thread(record) 会在另一个线程调用 record

```python
audio = await asyncio.to_thread(record)
text = await asyncio.to_thread(transcribe, audio)
```

注意传的是函数本身：

```python
await asyncio.to_thread(record)    # 在线程里调用 record
await asyncio.to_thread(record())  # 先在当前线程执行 record()，通常不是你想要的
```

项目里这样读取输入：

```python
user_text = (await asyncio.to_thread(input, "你：")).strip()
```

执行顺序：

```text
工作线程调用 input("你：")
    ↓
当前异步任务等待，事件循环仍可处理 MCP 和网络消息
    ↓
用户输入结束，取回字符串
    ↓
strip() 去掉两端空白
```

直接使用 `input()` 会阻塞当前事件循环线程，但不表示操作系统里所有线程、其他进程以及远程 server 都停止。主进程里需要该事件循环推进的网络读写会受到影响。

需要补充：取消等待 `to_thread` 的任务，不等于强制终止已经运行的同步线程函数。

## 6. with 和 async with：进入、使用、退出

### 你的理解成立：上下文管理器可以把进入和退出清理集中管理

```python
with open("notes.txt", encoding="utf-8") as file:
    text = file.read()
```

离开代码块时会执行退出逻辑，普通异常也会经过退出处理。它帮助我们避免忘记关文件、关连接或释放锁。

| 写法 | 进入和退出调用什么 |
|---|---|
| `with obj` | `obj.__enter__()`、`obj.__exit__(...)` |
| `async with obj` | 等待 `obj.__aenter__()`、`obj.__aexit__(...)` |

是否使用 `async with`，由对象实现的协议决定，**不是看代码块里有没有 await**。

你的播放器就是普通 `with` 里使用 `await`：

```python
with stream:
    while stream.active:
        await asyncio.sleep(0.1)
```

Google API 的 `with build(...)` 也是同步上下文管理，所以前面不加 `async`。

### yield 和 finally 是一种编写方式，不是所有上下文管理器的固定源码

```python
from contextlib import contextmanager

@contextmanager
def show_lifecycle():
    print("进入")
    try:
        yield "可用的值"
    finally:
        print("退出清理")

with show_lifecycle() as value:
    print(value)
```

`yield` 前面准备资源，`yield` 处把控制权交给代码块，离开后继续退出流程。`@asynccontextmanager` 是异步版本，也可以用 `try/finally` 管理清理。

## 7. TaskGroup 和 AsyncExitStack：各自管理什么

### 你的理解成立：TaskGroup 统一管理通过它创建的任务

```python
async with asyncio.TaskGroup() as tasks:
    tasks.create_task(player.run())
    tasks.create_task(receive())
    print("这行仍在当前任务中执行")
```

并不是代码块里的每一行都自动变成新任务：

```python
tasks.create_task(func1())       # 新的组内任务
await func2()                   # 当前任务直接等待
asyncio.create_task(func3())     # 不是这个 TaskGroup 创建的任务
```

TaskGroup 的规则：

- 正常离开代码块时，等待组内任务全部结束。
- 某个组内任务出现未处理的普通异常，会取消其他组内任务，等待退出清理，然后把错误向外抛出，通常以 `ExceptionGroup` 汇总。
- 某个任务正常返回，不会因此取消其他任务。
- 单个子任务被取消，与它抛出普通异常的处理规则不同。
- 如果任务永远运行，正常离开代码块时也会一直等待；不是走到代码块末尾就自动取消所有任务。

### 你的理解成立：AsyncExitStack 用来登记退出清理

```python
async with AsyncExitStack() as stack:
    worker_task = asyncio.create_task(worker.run())
    stack.push_async_callback(stop_worker, worker_task)
```

`push_async_callback` 现在只登记，不立即调用。离开时才执行：

```python
await stop_worker(worker_task)
```

也能进入并登记一个异步上下文：

```python
client = await stack.enter_async_context(httpx2.AsyncClient())
```

等价效果是：现在进入客户端的上下文，退出 stack 时调用它的退出逻辑。多个清理按登记的逆序执行。

需要补充：仅仅把 `create_task` 写在 `AsyncExitStack` 里面，不会自动登记任务清理。

项目中把 TaskGroup 放在 stack 里面，是为了得到这个顺序：

```text
准备 worker、登记清理
    ↓
运行 TaskGroup 内的录音/接收/播放等任务
    ↓
TaskGroup 的任务退出并完成清理
    ↓
AsyncExitStack 再停止 worker、释放其他资源
```

这是一种生命周期设计，不是两个工具必须嵌套使用。

## 8. 取消、异常、Future、锁

### 你的理解成立：raise 后不会正常继续执行后面的 return

```python
def example():
    raise ValueError("失败")
    return "成功"  # 这条正常路径不会到达
```

异常会向外寻找 `except`。有 `finally` 或上下文退出逻辑时，仍先经过相应清理；清理逻辑本身也可能改变最终结果。

```python
try:
    result = await worker.call(name, arguments)
except Exception as exc:
    # worker.call 内 await future 抛出的普通异常可以到这里
    print(exc)
```

`ExceptionGroup` 本身也属于普通异常范围，所以项目可以用 `except Exception` 接住由普通异常组成的异常组。`format_error` 再递归提取里面的错误。

### set_exception 不是当场向其他任务 raise

```python
future.set_result(result)       # 将这个 future 标记为成功完成
future.set_exception(exc)      # 将这个 future 标记为异常完成
```

等待它的代码随后得到结果或异常：

```python
result = await future
```

Future 是“一份尚未填好的结果”；Task 则会执行协程，并保存执行结果或异常。

### cancel 是请求取消，await 才能等待退出清理

```python
task.cancel()
try:
    await task
except asyncio.CancelledError:
    pass
```

取消通常在异步等待处被交付，不是像拔电源一样立即杀掉执行。`CancelledError` 在当前 Python 中不属于 `Exception`，不要用普通 `except Exception` 误以为已经接住取消。

进入 `async with stdio_client(...)` 不会自动触发 worker 最外层的 `finally`。只有控制流程实际离开对应的 `try`，例如退出、异常或取消，才执行那个 `finally`。

### 两种锁不要混淆

| 锁 | 适合谁共享数据 | 等锁时怎样 |
|---|---|---|
| `asyncio.Lock` | 同一事件循环中的异步任务 | 异步等待，其他任务仍可运行 |
| `threading.Lock` / `RLock` | 多个线程 | 可能阻塞当前线程 |

播放器的音频回调与异步接收任务处于不同线程，所以用线程锁保护缓冲区。AlphaBrain 用异步锁串行处理提问，避免多条请求交错修改同一份历史。

`RLock` 允许同一个线程重复取得同一把锁；它并不是“让所有线程都能同时进去”。

## 9. WebSocket 与 async for

### 你的理解成立：self.ws 像一条持续打开的双向通道

```text
你的程序 ── 音频、配置、工具结果 ──→ Qwen 服务端
你的程序 ←── 转写、回答音频、事件 ── Qwen 服务端
```

它与一次普通 HTTP 请求得到一份响应的使用方式不同。一轮回答结束，连接仍可以继续收发下一轮消息。传统 WebSocket 建连常使用 HTTP Upgrade 握手，建立后通过 WebSocket 帧交换消息，不能把它简单当作不断重复普通 HTTP 请求。

### 你的理解成立：async for 自己就是循环，不需要为了监听再套 while True

```python
async for message in self.ws:
    event = json.loads(message)
    await self.handle(event, timers)
```

概念上相当于：

```text
异步等待下一条消息 → 处理 → 异步等待下一条消息 → 处理 → ……
```

没有消息时暂停等待，不是空转检查，也不是“这批消息读完就一定结束”。具体何时结束，由异步迭代对象决定；这里的 WebSocket 会在连接结束时结束迭代或抛出异常。

普通 `for` 要求同步迭代协议，`async for` 要求异步迭代协议。不能因为两者都是“流”就随意替换。

这个接收循环停止的常见原因：

- 连接正常关闭，迭代结束。
- 网络异常或消息处理异常向外抛出。
- 任务被取消。
- 代码主动 `break` 或 `return`。

`response.done` 只代表一轮模型生成结束，不代表 WebSocket 连接关闭，也不代表扬声器已经把缓冲区里的音频播完。

## 10. 音频大图：谁录音，谁接收，谁播放

### 你的理解成立：网络返回音频小块，缓冲区暂存，播放回调持续取数据

```text
麦克风 → send_microphone → Qwen 服务端
                              ↓
                         返回音频事件
                              ↓
                     QwenReceiver.handle
                              ↓
                       player.append
                              ↓
                        self.buffers
                              ↓
                 音频线程调用 player.callback
                              ↓
                            扬声器
```

你当前的实时语音路径直接发送音频，不是先在本地 Whisper 转文字再传模型。服务端会另外发回语音转写文字，客户端用它判断是否需要回应。

录音和音频事件中的 base64 是为了把二进制音频表示成 JSON 可携带的文本，不是语音识别，也不等于压缩音频。

### run、append、callback 的分工

| 方法 | 工作 |
|---|---|
| `player.run()` | 打开输出流，维持运行，设备故障后重试 |
| `player.append(...)` | 把模型返回的一段音频放入缓冲区 |
| `player.callback(...)` | 被音频库调用，从缓冲区取音频填进 `outdata` |

`tasks.create_task(player.run())` 本身只安排异步任务。播放回调的音频线程由 sounddevice/底层音频库管理。

## 11. OutputStream、callback、outdata、frames、offset

### 你的理解成立：callback 是你写的，调用时机由音频库负责

```python
stream = sd.OutputStream(
    samplerate=self.rate,
    channels=1,
    dtype="float32",
    device=device,
    blocksize=max(1, self.rate // 100),
    latency="low",
    callback=self.callback,
)
```

这里注册的是函数对象：

```python
callback=self.callback    # 交给库以后调用
```

不是立即调用函数：

```python
callback=self.callback()  # 现在执行，把返回值交给它；这里不是正确用法
```

`self.callback` 是绑定方法，已经带着 `self`。音频库调用时传后面的四个参数。

### 你的理解成立：outdata 不需要 return，因为修改的是同一个数组

```python
def callback(self, outdata, frames, time_info, status):
    outdata.fill(0)
    outdata[...] = 音频数据
```

上面是示意，`音频数据` 需要替换成实际数组。音频库提供 `outdata`，回调原地填充它，然后音频库使用填好的内容。

用普通列表类比：

```python
def fill(data):
    data.append("声音")

buffer = []
fill(buffer)
print(buffer)  # ["声音"]
```

修改对象的内容不需要返回整个对象。需要补充：如果只是 `data = 新对象`，那是在函数内重新绑定局部变量，并没有替换调用方原来持有的对象。

### frame 是一个时刻所有声道的采样集合

在你的单声道播放器里：

```text
1 frame = 1 个采样值
```

立体声则是：

```text
1 frame = 同一时刻的左声道采样 + 右声道采样
```

当前输出流是 `float32`、单声道，所以每帧 4 字节。原始 PCM16 单声道则每帧 2 字节。帧的字节数取决于声道数和数据类型，不能永远写成 4 字节。

帧的时间长度由采样率决定：

```text
24000Hz：每秒 24000 帧
240 帧 ÷ 24000 帧/秒 = 0.01 秒 = 10ms
```

`frames` 是这次回调需要填多少帧，是数量；`outdata` 才是装音频的数组。

### 每次 frames 不一定是 240

你的代码按设备采样率设置 `blocksize=self.rate // 100`，每次约 10ms：

| 输出采样率 | 配置的 blocksize |
|---|---:|
| 24000Hz | 240 |
| 44100Hz | 441 |
| 48000Hz | 480 |

当前配置使用固定的非零 blocksize，但回调仍以音频库实际传来的 `frames` 为准。如果使用别的流配置，例如 `blocksize=0`，回调块大小也可能变化。

### 一块音频包含多个 frame

```text
self.buffers
  ├─ 第一块：600 帧
  ├─ 第二块：1200 帧
  └─ 第三块：其他数量的帧
```

网络送来的一块，与播放回调每次需要的帧数，不必大小一致。

当前单声道数组里：

```python
self.buffers[0][0]  # 第一块的第一个采样值，如 0.12
self.buffers[0][1]  # 第二个采样值，如 -0.35
```

采样值可正、可负、可为零。它不是帧数，也不是一个还能继续切开的小音频数组。

可以取一块音频的一部分，但取的单位仍是完整 frame：

```python
samples[0:240]  # 取完整的 240 个采样值，不切开最后一个采样值
```

### 你的理解成立：offset 记录当前第一块已经取到了哪里

假设第一块 600 帧，每次需要 240 帧：

```text
第一次取 0:240     → offset=240
第二次取 240:480   → offset=480
第三次取 480:600   → 第一块取完，移除第一块，offset=0
                  → 本次还差 120 帧，继续从下一块取
```

```python
self.buffers.popleft()  # 移除第一块，下一块成为新的 buffers[0]
self.offset = 0
```

移出缓冲区表示音频已经交给输出流程，不等于用户的耳朵在同一瞬间听完。设备本身也有少量缓冲和延迟。

### 创建输出流不等于已经有声音

```python
stream = sd.OutputStream(...)  # 创建、配置并打开流资源，但尚未启动回调播放

with stream:                  # 启动流，退出上下文时停止并关闭
    ...
```

流开始运行后，音频库调用回调。如果应用缓冲区没有音频，`outdata` 保持全零，就播放静音。

### 你的理解成立：内层循环维持流，外层循环负责重试

```python
while True:                        # 外层：打开设备，故障后重试
    ...
    with stream:
        while stream.active:       # 内层：正常运行时停留在这里
            await asyncio.sleep(0.1)
    ...
```

每等 0.1 秒只回到内层检查 `stream.active`，不会重新查找设备。

`stream.active` 检查输出流是否仍运行，不保证物理扬声器一定有声音。`await sleep` 暂停的是当前异步任务，音频线程仍持续调用回调。

正常循环期间不会执行到外层 `finally`；离开对应 `try` 时才执行。异常或取消也会触发退出清理。

## 12. QwenReceiver：通过事件协调对话

### 你的理解成立：receiver.run 持续接消息，handle 按类型处理

它与 `player.run` 是不同对象的方法：

```text
QwenReceiver.run → 接收 WebSocket 事件
StreamingPlayer.run → 维持音频输出设备
PersonalWorker.run → 维持 MCP 连接并处理队列
```

`run` 只是方法名，并没有 Python 统一规定的特殊行为，要看所属类的实现。

主要事件及项目行为：

| 事件 | 处理 |
|---|---|
| `speech_started` | 记录新发声、降低旧音频音量、开始本地 VAD 确认 |
| `speech_stopped` | 记录用户说话结束和延迟计时起点 |
| `committed` | 标记音频输入已提交 |
| `input_audio_transcription.completed` | 用文字判断是否值得回应 |
| `response.created` | 记录模型开始生成哪次回答 |
| `response.audio.delta` | 解码音频并放进播放器缓冲区 |
| `response.function_call_arguments.done` | 将完整工具调用交给后台任务 |
| `response.done` | 记录 usage、费用，检查是否继续回答 |

这些是当前代码处理的事件分工，不是承诺服务端所有接口都兼容。项目对删除、截断等兼容性请求仍有明确错误处理。

### 暂停与真正打断不同

```text
检测到可能的人声 → 降低音量
持续人声达到阈值 → pause：保留未播音频
    ├─ 转写只是噪音 → resume：接着播放
    └─ 有意义的话语 → clear：丢弃旧音频，取消旧生成，按接口能力同步历史
```

本地 VAD 判断“像不像人声”；转写规则判断“是不是有意义的输入”，两者不是同一种判断。

`maybe_respond()` 集中决定何时回答：输入已提交、文字有意义、尚未请求过回答，并且没有需要等待的当前回答。工具结果全部返回后，也由它判断是否请求后续回答。

`response_pending` 记录“已经发送 response.create，但还没收到 response.created”，用于防止这段间隙内重复请求。

## 13. MCP：通道、会话、工具、返回结果

### 你的理解成立：先有传输通道，再在上面建立 MCP 会话

HTTP 连接例子：

```text
httpx2.AsyncClient
    → 管理 HTTP 请求、认证头和超时
streamable_http_client
    → 提供 MCP 消息的 read/write 通道
ClientSession(read, write)
    → 管理 MCP 协议会话
await session.initialize()
    → 协议握手
await session.list_tools() / call_tool(...)
    → 发现和使用工具
```

`ClientSession` 不是 Google OAuth 登录，也不只是“向服务器表明身份”；它是 MCP 协议层的会话。身份验证还取决于传输层认证等配置。

MCP 的常见服务器能力有 tools、resources、prompts。JSON-RPC 是消息表达格式；stdio、Streamable HTTP 是承载消息的传输方式。

### 你的理解成立：stdio 不需要监听 HTTP 端口

stdio 是标准输入输出机制，跨语言、跨操作系统使用，不是 Python 或 macOS 专属语法。

| 流 | 常见用途 |
|---|---|
| stdin | 进程读取输入 |
| stdout | 进程写主要输出 |
| stderr | 进程写诊断日志和错误 |

项目中：

```python
async with stdio_client(server) as (read, write):
    ...
```

客户端按参数启动 server 子进程，并建立管道：客户端发送对应子进程 stdin，子进程 stdout 上的消息被客户端接收。

手动在终端启动 stdio server 时，它可以等待标准输入，但不会因此有网络客户端自动连上。MCP SDK 的 stdio 客户端通常负责自己启动并管理该子进程；这不是说所有使用 stdio 的程序都必须由客户端启动。

### 你的理解成立：主进程 print 与 server stdout 是两回事

主程序的 stdout 通常输出到终端；个人 server 的 stdout 被管道接到 MCP 客户端。所以在主程序 `print` 不会自动混进子进程的协议流。

但 stdio server 自己不能随便往 stdout 打调试文字：

```python
print("调试日志", file=sys.stderr)
```

`print()` 默认写 stdout，但 stdout 不只来自 print，也可以来自 `sys.stdout.write` 等。logging 写向哪里取决于 handler；默认的标准流 handler 通常是 stderr，不能仅凭“用了 logger”就认定一定安全。

### 工具返回字符串，客户端通常收到的是结果对象

```text
server 工具 return "已完成：牛奶"
    ↓ SDK 包装并通过 JSON-RPC/传输通道发送
客户端得到 CallToolResult
    ├─ content：结果内容块，例如 TextContent
    └─ is_error：是否为工具错误
```

协议中的 `isError` 与项目 SDK 对象里的 `is_error` 是不同命名形式。给模型回传工具结果时，应保留错误状态和文本，不只看是否有字符串。

工具框架支持的 `ToolError` 会被转换成工具错误结果。它与 JSON-RPC 顶层的协议错误不是一回事；具体异常转换行为要看所用 SDK。

## 14. PersonalWorker：queue 传请求，future 送回结果

### 你的理解成立：worker 长期持有连接，从队列取请求并执行 call_tool

队列里的一项：

```python
(name, arguments, future)
```

工作流程：

```text
调用方 await worker.call(name, arguments)
    ↓
call 创建 future，把 tuple 放进 queue，然后 await future
    ↓
run 的 await queue.get() 拿到 tuple
    ↓
await session.call_tool(name, arguments=arguments)
    ↓
future.set_result(result)
    ↓
原来的 await future 取得结果，worker.call 返回给调用方
```

错误则走 `future.set_exception(exc)`，等待 future 的调用方收到异常。

结果不是重新放回 queue；填入的是 tuple 引用的那个 future 对象。

队列空时 `await queue.get()` 异步等待，不会不断空转。当前 worker 串行执行请求，同一时刻只取一个工具调用来处理。

需要补充：客户端超时，不足以证明远端写入没发生。因此添加任务、创建提醒等操作不能因为没收到结果就无条件重放。

## 15. dict、list、tuple、类型标注

### 你的理解成立：dict 可以用 get，也可以用方括号取值

```python
result = {"items": ["牛奶", "鸡蛋"]}

result.get("items")       # 返回列表
result["items"]           # 返回列表
result.get("missing")     # None
result.get("missing", []) # 键不存在时返回默认值 []
result["missing"]         # 抛 KeyError
```

需要补充：键存在但值是 `None` 时，`get("items", [])` 仍返回 `None`，不会自动换成默认值。

```python
if title in matches:
    ...
```

对于字典，`in` 检查的是键。对列表，`in` 检查的是元素。

`task_list.get("id")` 能不能用，取决于 `task_list` 是否是字典或实现了对应方法。变量名不会决定对象类型。Google API 的 JSON 结果通常解析成 dict，MCP SDK 结果则常是有属性的对象。

### append 与 extend

```python
items = []
items.append(["牛奶", "鸡蛋"])
# [["牛奶", "鸡蛋"]]：把整个列表当作一个元素加入

items = []
items.extend(["牛奶", "鸡蛋"])
# ["牛奶", "鸡蛋"]：逐个加入元素
```

方法名是 `extend`，不是 `extends`。字符串也能迭代，`extend("牛奶")` 会逐字加入，因此不要误把字符串当成标题列表。

### 多个返回值，通常是一个 tuple 加上解包

```python
def make_pair():
    return "read", "write"

read, write = make_pair()
```

函数返回的是 tuple `("read", "write")`，赋值时拆给两个变量。`as (read, write)` 也可以对上下文提供的值解包。

### 类型标注不负责自动创建数据，也不会替你运行时校验

```python
result: dict[str, object] = {
    "tasks_lists": [],
    "is_error": False,
    "id": "",
}
```

`object` 允许这些不同类型的值，但静态检查器只能确定取出的值是 object，不能据此保证它有 `append`。

一个简单办法是先保留明确类型的变量：

```python
task_lists: list[str] = []
result: dict[str, object] = {"tasks_lists": task_lists}
task_lists.append("购物清单")
```

需要固定字段类型时，可用 `TypedDict`：

```python
from typing import TypedDict

class TaskListResult(TypedDict):
    tasks_lists: list[str]
    is_error: bool
    id: str

result: TaskListResult = {"tasks_lists": [], "is_error": False, "id": ""}
result["tasks_lists"].append("购物清单")
```

字典项不是变量声明，不能写成 `"tasks_lists": list = []`。

## 16. 缓存、锁、分页与 Google API

### @cache / @lru_cache 缓存函数返回值

```python
@cache
def get_tasks_service():
    return build(...)
```

这是无参数函数，之后调用复用同一 service。`@lru_cache(maxsize=1)` 最多保留一组参数对应的结果；如果也是无参数函数，就很适合缓存一个 Whisper 模型。

缓存不等于所有内部对象都线程安全。缓存库保护自己的缓存结构，不代表返回的 HTTP transport 可以被多个线程同时使用；首次并发调用还可能重复计算，所以项目额外用锁保护。

OAuth access token 过期时，支持刷新且携带有效 refresh token 的凭据可以由认证传输层刷新。不是“加了 cache 就永不过期”。

### 清单名与任务标题都不是唯一的 API 主键

```text
清单名 → 查清单 id → 在指定 tasklist 中读写任务
任务标题 → 可能匹配多个 task id → 按每个 id 执行 patch
```

你的结论“两个同名牛奶都标记完成”通过 `dict[str, list[str]]` 实现：一个标题保存多个任务 id，再逐个完成。

分页中 `nextPageToken` 表示下一页入口。只处理第一页可能漏掉清单或任务。

```python
tasks.tasks().list(...).execute()
```

前面构造请求，`execute()` 才执行同步网络请求并等结果。

## 17. OAuth、时间与日历

### 你的理解成立：测试人员分别授权，各自访问自己的数据

应用配置标识“谁在请求权限”，scope 描述“请求访问什么”，登录用户决定“是否允许访问自己的数据”。项目拥有者与实际授权的数据拥有者不一定是同一个人。

```text
credentials.json：应用的 OAuth 配置
token.json：某个用户同意授权后获得的凭据
```

`run_local_server(port=0)` 临时在本机启动回调接收服务，随机使用一个可用端口。它接收浏览器授权返回的信息，不是 Google API 本身运行在本地。

`calendar.events` 针对日程，不代表拥有列出所有日历的 `calendarList` 权限。scope 与 API 方法要匹配。

### 你的理解成立：UTC 不随夏令时变化，本地时区偏移会变化

例如 Los Angeles 在不同日期可能使用 `-07:00` 或 `-08:00`，应由 `ZoneInfo("America/Los_Angeles")` 按目标日期计算，不交给模型猜缩写。

```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

local = datetime.fromisoformat("2026-11-03T08:00:00")
aware = local.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
utc_time = aware.astimezone(timezone.utc)
```

两步不同：

- `replace(tzinfo=...)`：给这个墙上时间指定时区含义，不改变显示的小时。
- `astimezone(...)`：同一真实时刻换成另一个时区的显示方式。

`2026-11-03T08:00:00-08:00` 是带偏移的本地时间表达，不是“显示成 UTC”；对应 UTC 要用 `astimezone(timezone.utc)`。

需要补充：只做 `replace` 不会自动拒绝夏令时切换造成的不存在时间、重复时间。项目遍历 fold，并通过本地→UTC→本地的往返检查处理这些情况。

### 你的理解成立：日历事件是时间段，不是持续响的闹钟

`start` 和 `end` 定义事件持续时间。`minutes_before=10` 表示开始前 10 分钟设置一次对应提醒，不表示整个事件期间每分钟重复提醒。最终通知行为还由日历客户端及其通知设置决定。

## 18. 文件路径、依赖、类型检查

### __file__ 和 cwd 不是同一件事

```python
Path(__file__).with_name(".env")
```

把当前脚本路径的文件名换成 `.env`，通常用于找脚本同目录配置；不是“当前在哪个文件夹敲 python，就找哪里的 .env”。`__file__` 不保证天然是绝对路径，项目用 `resolve()` 得到绝对位置。

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

当前 `core/paths.py` 的父目录是 `core`，再上一层是 `alpha`。

`StdioServerParameters(cwd=...)` 设置子进程的工作目录。它影响子进程的相对路径解析，不是切换整个电脑所有程序的目录。

### 依赖安装与记录

在 `alpha` 目录，使用项目自己的解释器，避免查到其他环境：

```bash
.venv/bin/python -m pip show mcp
.venv/bin/python -m pip freeze > requirements.txt
```

`freeze` 是当前环境已安装包的快照，包括间接依赖；它不会因为以后安装了新包就自动更新文件。这个重定向会覆盖已有文件，已有手工维护清单时先确认是否要替换。

### import-untyped 不等于库不能运行

它通常表示 mypy 缺少该库的类型声明。项目可以局部忽略：

```python
import sounddevice as sd  # type: ignore[import-untyped]
```

这只影响静态类型检查，不改变运行行为，也不代表整个文件的所有类型错误都应该忽略。

## 19. 对话记忆、工具数据与将来的客厅终端

### 你的理解成立：当前 Qwen 新建会话不会自动记得上一条连接的对话

你当前 Claude 路径由本地代码维护 `messages`，每次请求带上历史；Qwen 实时路径主要使用服务端当前会话的历史。客户端状态变量不是完整的对话记忆数据库。

本地保存用户转写、模型回答、工具结果，是程序自己的文件/数据库功能，不需要模型拥有“读本地硬盘”的权限。

但保存记录不等于恢复模型记忆：新会话还需要通过该接口实际支持的方式重新提供历史或摘要。不能假定当前 Omni 接口完整支持 Claude 的 messages 形式，也不能把其他 Qwen 产品的接口能力直接套过来。

对话记忆丢失不会撤销已完成的 Google Tasks、日历或设备操作。模型不知道刚才做了什么，与远端数据不存在，是两件事。

### 你的理解成立：客厅终端可以通过局域网实时传音频给楼上 Alpha

```text
客厅设备：麦克风、唤醒词、播放
    ⇄ 局域网音频与控制消息
楼上电脑：Alpha、云端模型连接、MCP 工具
```

唤醒词检测和理解中文指令是两个步骤。把现有程序搬到这个架构时，音频传输之外，还需要交换暂停、恢复、清空播放和已播放时长等状态。

## 20. 回到源码时怎么读

先沿入口向下读，不必一开始就理解所有变量：

| 想理解什么 | 推荐文件 |
|---|---|
| 录音、转写、朗读的终端流程 | [apps/cli.py](../apps/cli.py) |
| Claude 对话与多工具路由 | [brain/agent.py](../brain/agent.py) |
| Qwen 收发事件、打断、工具后续回答 | [voice/qwen_realtime.py](../voice/qwen_realtime.py) |
| 播放缓冲区、offset、callback | [voice/audio.py](../voice/audio.py) |
| 队列、future、重连与取消 | [integrations/personal_worker.py](../integrations/personal_worker.py) |
| MCP 目录发现、协议转换与 HA 调用 | [integrations/mcp_client.py](../integrations/mcp_client.py) |
| Google Tasks、日历与时间校验 | [servers/personal_server.py](../servers/personal_server.py) |
| OAuth 授权、复用与刷新 | [servers/google_auth.py](../servers/google_auth.py) |
| 两段式录音结束判断与 Whisper | [voice/listen.py](../voice/listen.py) |

读到陌生的一行，可以依次问：

1. 这一行是在创建对象、安排任务，还是已经执行实际操作？
2. 当前拿到的是函数、协程、Task、Future，还是最终结果？
3. 这里的 await 暂停的是哪个任务？其他哪些任务能继续？
4. 这个值是谁创建的、谁修改的、谁负责清理？
5. 循环什么时候继续，什么时候退出？异常会去哪里？
6. 这次“完成”指的是模型生成完、消息发完、工具执行完，还是扬声器播放完？
