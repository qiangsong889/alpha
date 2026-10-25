"""用和大脑相同的代码拼出语音模式的 system prompt，写到 data/system_prompt.txt（含记忆，data/ 不进 git）。

在 alpha 目录运行：.venv/bin/python -m devtools.show_system_prompt

现场连 HA 拉 Assist 提示（设备列表），读当前记忆，调用 AlphaBrain._prepare_catalog()。不启动个人服务、网关和
Claude 客户端，不调用模型；存档按正式入口（brain.server）的情况视为开启，所以带上对话记忆的规则。
大脑在运行时也可以执行：只读记忆数据库，不拿操作库的进程锁。
"""
import asyncio
from zoneinfo import ZoneInfo

import httpx2
from dotenv import load_dotenv

from brain.agent import AlphaBrain
from core.memory import MemoryStore
from core.paths import ENV_FILE, OPERATIONS_DB, PROJECT_ROOT
from core.settings import setting
from integrations.mcp_client import load_ha_catalog

OUT = PROJECT_ROOT / 'data/system_prompt.txt'


async def main():
    load_dotenv(ENV_FILE)
    brain = AlphaBrain(style='voice')
    brain.reminder_tz = ZoneInfo(setting('REMINDER_TIMEZONE'))
    async with httpx2.AsyncClient(headers={'Authorization': f"Bearer {setting('HA_TOKEN')}"},
                                  timeout=30) as client:
        brain.ha_catalog = await load_ha_catalog(setting('HA_URL').rstrip('/') + '/api/mcp', client)
    brain.personal_tools = []  # 只影响工具列表；None 会在提示里写“个人服务不可用”
    brain.archive = object()   # 正式入口开着存档：带上 search_history 的规则
    brain.memory_store = MemoryStore(OPERATIONS_DB)
    try:
        brain._prepare_catalog()
        blocks = brain._request_system()
    finally:
        brain.memory_store.close()
    text = '\n\n'.join(f"===== 第 {i} 块{'（缓存断点）' if b.get('cache_control') else ''} =====\n{b['text']}"
                       for i, b in enumerate(blocks, 1))
    OUT.write_text(text + '\n')
    print(f"{len(blocks)} 块、{sum(len(b['text']) for b in blocks)} 字 → {OUT}")
    print(f"工具 {len(brain.tools)} 个（另有 web_search、alpha_voice_reply）：" + '、'.join(t['name'] for t in brain.tools))


if __name__ == '__main__':
    asyncio.run(main())
