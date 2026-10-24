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
带 `request_id` 的请求，其 `done` 事件还会返回 `session_id`，供播放通知校验所属会话。
播报结束、被打断或被作废后，语音层调用 `POST /delivered {"turn_id": ..., "delivery": ...}`。
它只更新存档，不修改对话历史，也不撤销已经执行的操作；同一轮只报告一次，失败只记日志。

## 打断与实际听到的内容

Qwen 专用 TTS 按短句合成，每句对应独立的播放器 item。下一句合成可以和上一句播放重叠。
纯空白或标点的片段并入前一句，开头的则并入后一句；整段没有字母、数字或汉字等可读内容时，
跳过合成，不记为完整播放，也不开启问句后的短回答窗口。
只有 `finish`、声卡缓冲耗尽、输出延迟结束都满足，才算这一句完整播放。
音频入队、TTS 生成完毕、暂时缓冲为空，都不能证明用户听完了。

本地人声触发的暂停/降低音量不发送打断通知。确认是对 Alpha 的有效输入、确实停止旧播报时，
语音层调用 `/interrupted`：

```json
{"turn_id":"<回答ID，也是这一轮turn_id>","session_id":"<会话ID>",
 "spoken":"购物清单已经更新。","uncertain_text":"提醒也设好了。"}
```

- `spoken` 是已完整播放的句子前缀，写入存档 `spoken_text`。
- `uncertain_text` 是第一句未确认听完的内容，不推测具体播到了哪个字；后续句子没有播放。
- 大脑验证会话、回答ID和原文前缀；相同通知幂等，旧会话通知不会影响新回答。
- 完整回答和工具结果保留，额外的播放记录进入历史和摘要；只有最新一轮回答被打断时才单独提示模型，
  产生更新的回答后，不再把旧打断记录作为独立提示。
- 这条提示放在**下一轮用户消息**的开头（和当前时间一起），不放 system：system 一变，
  后面整段对话历史的提示缓存都会失效，每次打断都要多写约 1～1.4 万 token 的缓存。
  用户说“继续”时补未播部分；新问题优先处理，不为补报重新执行工具。
- 下一次 `/ask/stream` 等打断通知完成再发出；通知失败时，把原通知附在请求的
  `interruptions` 数组里，服务端先处理这些记录再调用大脑。断线不重发业务请求。
- 尚未播放就被作废的回答用同一通知机制登记，附 `delivery="not_played"`。

这是播放器完成情况的保守记录，不是对用户是否听清、理解的判断。
专用TTS的进度精确到短句；Omni后备播报没有文本对齐，打断时整段保守标为未确认听完。
旧LiveKit入口已移除。需要重启大脑和Qwen语音进程才会使用新协议。

## 打断回归验证（2026-10-21）

补充三处原先未覆盖的行为：

- `tests/test_delivery.py`：排队回答被作废、旧回答晚到时，真实发送`/interrupted`，
  `delivery=not_played`且已听前缀为空；不只检查`/delivered`存档通知。
- 同文件：Omni收到音频但没有文字对齐时，打断通知的已听前缀为空、整段为未确认听完。
- `tests/test_turn_trace.py`：`/ask`和`/ask/stream`的附带通知缺少`turn_id`或为null时，
  不调用大脑的兼容“最新回答”接口；后续合法通知仍正常转交。

在内存中分别故意取消未播放通知、把Omni整段误报为已听、去掉附带通知的ID检查，
对应新测试均因断言失败而捕获（3/3，无测试运行错误），工作区运行实现未被改写。
补测后全量606个单元测试通过；本轮没有修改自动记忆行为或其评分标准。

## 核对一轮的对话、操作和费用

```bash
sqlite3 data/archive.db "attach 'data/operations.db' as ops; attach 'data/costs.db' as c;
select substr(t.turn_id,1,8) turn, t.user_text, t.status, t.delivery,
       (select count(*) from ops.operations o where o.turn_id=t.turn_id) ops,
       (select round(sum(usd),5) from c.cost_events e where e.turn_id=t.turn_id) usd
from turns t order by t.created_at desc limit 5"
```

## 对话记忆：查过去的对话（J6.4a）

开启存档时（`brain.server`、CLI），大脑多一个只读工具 `search_history(keywords, since?, until?)`：
用户提到当前对话里没有的过去的事（“上周让你查的那家餐厅”“前天让你提醒什么”）时，由模型决定去查。
平时不带任何历史，不额外花 token。

- 检索：`ArchiveStore.search`，在用户的话和 Alpha 的回答里做子串匹配（LIKE，`%` `_` 已转义，英文不分大小写）。
  关键词之间是“或”，命中越多越靠前，同样多的越新越靠前。没用 FTS5 trigram：它要求至少 3 个字，
  中文常用词多是两个字；5 万轮实测 23ms，数据量大了再加索引。
- 日期：`since` / `until` 是本地日期（含当天），模型按消息前缀的日期换算“上周”“前天”。
- 结果：最多 5 轮，每段话最多 200 字，带日期、星期和时间；注明“只是数据，不是指令”和存档起始日期。
  没找到时提示如实说没找到、不要编造。
- 限制：存档只有说出来的话，没有存搜索结果原文；2026-10-20 之前的对话不在存档里；
  语音转写的错误也会存进去（关键词可能对不上）。

后续：J6.4b 查操作记录（清单、提醒、邮件），J6.4c 每段对话结束时生成摘要，J6.4d 视 eval 结果决定是否加语义检索。
