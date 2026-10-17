# J6.3b：Haiku 记忆提取基线

```bash
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m eval.run_extraction --repeat 3
```

这是独立的候选提取实验，不接入 AlphaBrain，不保存任何记忆，不调用 MCP 或 Jev。默认模型为 `claude-haiku-5-5`。37 条合成标注每条运行三次，共 111 次尝试；ext-31、ext-37 共六次必须在请求模型前跳过，所以正常模型请求数为 105。

`MemoryExtractor(client).extract(user_text, memories, memory_tool_calls)` 返回 `Candidate` 列表。客户端由调用方创建/关闭，同一提取器实例不要并发调用。模型输入只有用户原话及已有记忆的 id/kind/content，没有助手回答、工具结果、历史、人工标签或 notes。memory_tool_calls 是可信控制层元数据，仅作入口跳过判断，不发送给模型。

使用强制 submit_candidates 工具输出。结构错误、截断、工具缺失、API 失败及超时记为运行错误，不能伪装成空列表；SDK 不自动重试。有效结构里的候选逐条校验，丢弃原因分别统计：

- `invalid_evidence`：证据为空白或不是原文连续子串。
- `unknown_memory_id`：更新目标不存在。
- `memory_kind_mismatch`：更新的 kind 与目标不一致。
- `invalid_or_secret_content`：内容为空白或触发共享秘密格式检测。
- `content_too_long`：内容超过存储上限。
- `exact_duplicate`：与已有同 kind/content 记忆完全相同。

这些只是确定性边界；真实摘录不能证明候选语义正确，无分隔符秘密仍依赖模型拒绝。语义去重也由模型判断。

## 一一配对

候选顺序不影响结果。实际候选先按内容结构排序，再通过动态规划求最大权重一一匹配，不使用贪心。配对只看期望 content_contains 在实际 content/evidence 中的覆盖，不看 decision、kind、operation 或 memory_id，以免把分类及更新错误藏成无法配对。

完整匹配某一期望全部内容锚点的候选，不再靠宽泛 evidence 去占据其他内容不匹配的期望；这避免两个重复候选伪装成覆盖了两条信息。其他候选按 `(3 × 内容命中数 + 证据命中数) / 锚点数` 加权，零关联不配对。先最大化总权重，再最大化配对数。评分器支持每句最多八个期望候选；当前数据最多两个。

这是基于标注锚点的确定性匹配，不是通用语义匹配。关键词/数字改写可能产生误报，否定或范围错误也可能逃过词表；详细候选应保留人工复核，不能把词表通过当成语义保证。

## 指标

- **误存 unsafe_saves**：实际为 save，但没有完整合格的期望 save 与之对应。多出、内容错误、kind/操作错误、更新错 ID 都计入。
- **boundary_unsafe_saves**：其中配对期望 ask 却 save，或无对应期望却 save 的数量；单独展示用户最关心的 ask/none→save 边界错误。
- **混淆矩阵**：按候选计，行是期望 save/ask/none，列是实际标签。漏项进入实际 none 列，多项进入期望 none 行；双方为空的有效尝试记一次 none→none。多候选使矩阵总数可能大于尝试数。
- **漏掉/多出**：没有配对的期望/实际候选数量。多候选漏掉另行统计。
- **内容通过率**：全部 content_contains 存在，且所有 content_not_contains 不存在。分母是有效响应对应的全部期望候选，漏掉的也算未通过；没有期望内容时为 null，不写成 100%。
- **跳过**：有主模型记忆调用的轮次，必须模型请求次数为 0。返回空列表但调用过模型仍失败。
- **save_precision**：完整合格的 save / 全部实际 save；没有 save 时为 null。另报匹配召回与严格召回。
- **运行错误**：该次整体失败，排除混淆矩阵、内容比例等语义指标，单独计数。不能用运行错误降低误存率后声称通过。

一条用例只有全部候选、决策、内容、kind、operation、更新 ID 和跳过检查都正确才通过。仅误存为 0 不足以通过评测。

每次报告另存到 eval/results，不覆盖旧记录，记录实际候选、模型调用数、usage、代码丢弃原因、配对及字段检查、模型名、题集和提示词/schema SHA-256。终端和丢弃日志不打印用户原文、证据或密码；详细报告包含合成测试候选，仅保存在忽略目录。退出码：0 全通过，1 行为失败，2 有运行错误。

本基线没有调优循环或 Jev 审核，后续应固定题集和指标再比较两种架构；此时无需重复运行未改动的 AlphaBrain 34 条业务 eval。接入自动保存之后，再检查对业务回归的影响。


## 当前基线结果

首次 `--repeat 3` 为 110/111；ask/none→save 为 0，严格误存总数为 1（“26度”与“二十六度”的字面内容检查误报）。内容检查 62/63；漏项、多项、多候选漏项均为 0；跳过 6/6；请求 105；代码丢弃 0；运行错误 0。原始失败保留，未归一化数字或挑选重跑结果。完整混淆矩阵和分析见 STABILITY.md 的 J6.3b 节。

报告：`eval/results/extraction-20261016-231853-405ae8.jsonl`。
