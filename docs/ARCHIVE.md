# 对话存档与关联 ID

一轮对话"说了什么、做了什么、花了多少钱、实际播放得怎样"分别存在三个数据库里，
通过同一组 ID 关联，不重复保存：

```text
session_id（一段对话；空闲超过 ALPHA_CONTEXT_IDLE_MIN 就换新的）
  └─ turn_id（你说的一句话 → 大脑的一次 ask）
        ├─ data/archive.db     turns：原话、回答、状态、播放结果
        ├─ data/operations.db  operations：这一轮的每一次工具调用
        └─ data/costs.db       cost_events：这一轮的每一笔费用
```

## 存档 `data/archive.db`

| 表 | 内容 |
| --- | --- |
| `sessions` | `session_id`、开始时间、最后一轮时间、轮数、来源（以后的 ESP32 房间） |
| `turns` | `turn_id`、`session_id`、时间、用户原话（不含时间前缀）、生成的回答、`status`（ok / error）、`error`（只记异常类型）、`delivery`、`spoken_text`、模型、大脑耗时 |

`delivery` 区分"生成了什么"和"实际播放了什么"：

| 值 | 含义 |
| --- | --- |
| `not_reported` | 语音层没有报告（CLI、LiveKit，或者报告失败） |
| `played` | 完整播放 |
| `interrupted` | 播放途中被打断 |
| `not_played` | 还没播放就被新的输入作废，或者 TTS 失败 |

存档只由正式入口开启：`brain.server` 和 `apps/cli.py` 传入 `ARCHIVE_DB`。
测试和 eval 创建的 `AlphaBrain()` 默认不写存档，不会把假对话写进真实记录。
存档写入失败只记一条警告，不影响回答。

## 费用的归属规则

大脑在 `ask()` 里设置 `core/trace.py` 的两个上下文变量，`record_cost` 默认从这里取值。

| 费用 | session_id | turn_id |
| --- | --- | --- |
| 这一轮的 Claude 请求 | 当前会话 | 这一轮 |
| 由这一轮触发的自动记忆提取（component=`Memory`） | 当前会话 | 触发它的那一轮（后台任务复制了创建时的上下文） |
| 后台上下文压缩（component=`Context`） | 当前会话 | **空**：压缩跨越多轮，不硬分配给最后一轮 |
| 发请求前的同步压缩 | 当前会话 | **空** |
| 语音层的 TTS | 空 | 播报的那一轮（`turn` 事件里拿到，显式传入） |
| 语音层的 ASR、Jev | 空 | 空：发生在大脑生成 turn_id 之前 |
| 2026-10-20 之前的旧记录 | 空 | 空 |

同一笔费用会写两次（开始时一条"待核对"，拿到用量后补全）；归属以**第一次写入的非空值**为准。

## 语音层怎么拿到 turn_id

`/ask/stream` 的请求带了 `request_id`（Qwen 语音层会带）时，第一个事件是：

```json
{"type": "turn", "request_id": "<批次 ID>", "turn_id": "<32 位十六进制>"}
```

不带 `request_id` 的客户端（LiveKit）收不到这个事件，协议保持不变。
播报结束、被打断或被作废后，语音层调用 `POST /delivered {"turn_id": ..., "delivery": ...}`。
它只更新存档，不修改对话历史，也不撤销已经执行的操作；同一轮只报告一次，失败只记日志。

## 已知限制

- `spoken_text`（被打断时实际播放到的文字）暂时为空：Qwen TTS 不提供逐字的时间戳，
  只能按播放时长估算，留到做页面时再决定是否需要。
- 打断后，Qwen 语音层还没有像 LiveKit 那样调用 `/interrupted` 截断大脑的对话历史；
  这次只记录播放结果，不改变大脑的行为。
- LiveKit 版本不报告播放结果，`delivery` 保持 `not_reported`。

## 核对一轮的对话、操作和费用

```bash
sqlite3 data/archive.db "attach 'data/operations.db' as ops; attach 'data/costs.db' as c;
select substr(t.turn_id,1,8) turn, t.user_text, t.status, t.delivery,
       (select count(*) from ops.operations o where o.turn_id=t.turn_id) ops,
       (select round(sum(usd),5) from c.cost_events e where e.turn_id=t.turn_id) usd
from turns t order by t.created_at desc limit 5"
```
