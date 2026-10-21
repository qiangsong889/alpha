# 配置参数

所有参数的名字、类型、默认值、范围、分组、由哪个进程读取、能不能在页面上修改，
都只写在 `core/settings.py` 的 `SETTINGS` 里。代码不再各自写 `os.getenv(名字, 默认值)`。

## 查看当前配置

```bash
.venv/bin/python -m core.settings
```

会打印：

- 大脑和语音各自的**错误**（✗，启动时会被拒绝）、**警告**（!）和**配置指纹**；
- 每个参数当前的值和来源：`默认` / `.env` / `启动时的环境变量（覆盖了 .env）`；密钥只显示"已设置 / 未设置"；
  🔒 表示不允许在页面上修改；
- `.env` 里出现、但清单里没有的**未识别参数**。它们可能已经不用了，也可能被 SDK、子进程或脚本读取，
  所以只提示，不自动删除。

## 代码里怎么读

```python
from core.settings import setting

timeout_s = setting('JEV_TIMEOUT_MS') / 1000   # 已经是 float，不用再转换
```

- 没设置或为空 → 用清单里的默认值；
- 不合法（类型、范围、选项、时区）→ 抛出 `SettingsError`，信息里带参数名和原因；
- 每次调用都重新读取环境变量，所以测试里 `patch.dict(os.environ, ...)` 照常生效。

## 启动时的检查

`brain.server`（`AlphaBrain.start`）和 `voice.qwen_alpha` 在 `load_dotenv` 之后调用 `require_settings(进程)`：

- **只检查这个进程会读的参数**：语音专用的参数写错了，不会让大脑也启动失败；
- **一次列出全部问题**，而不是跑到某一行才报错；
- 参数之间的关系也会检查，例如：
  - 压缩门槛必须小于上下文硬上限（大脑）；
  - 大脑必须有 `HA_URL`、`HA_TOKEN`、`ANTHROPIC_API_KEY`；语音必须有 `DASHSCOPE_API_KEY`、`QWEN_WORKSPACE_ID`；
  - Omni 路径打开本地 VAD 时，`QWEN_VAD` 必须是 `server_vad`；
- 警告不阻止启动，例如：Jev 没有密钥（会回退到本地规则）、选了旧的 Omni 路径、模型不在价格表里。

## 配置指纹

每个进程只用**自己会读的参数**算一个指纹（密钥只按"是否已设置"计入，不包含密钥的值）。
大脑启动后保存在 `AlphaBrain.settings_report`。D2 的运行状态会用它判断"已保存的配置是否已经生效"：
改了语音的参数，不会让大脑也显示"需要重启"。

## 旧路径

LiveKit 备用版本、旧版 Qwen 入口、OpenAI 实时版、按键说话读取的参数，登记为"旧路径"：
不会被报成未识别参数，也不会出现在页面上。这些文件本身还在直接读环境变量，这次没有改动。

## 新加一个参数

1. 在 `SETTINGS` 里加一条：名字、分组、说明、类型、默认值、范围或选项、`readers`；
   不应该从页面修改的设 `editable=False`；
2. 代码里用 `setting('名字')` 读取；
3. 如果它和别的参数有关系，在 `RULES` 里加一条规则；
4. 跑 `tests/test_settings.py`：会检查默认值能通过自己的校验。
