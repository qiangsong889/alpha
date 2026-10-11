// 在浏览器里探测 HA 的 MCP Server，方便在 DevTools 的 Network 面板看原始请求和响应。
//
// 用法：
//   1. 浏览器打开你的 HA：http://192.168.1.100:8123 并登录
//   2. F12 打开 DevTools → Console，把整个文件粘贴进去回车
//      （同源请求，不会遇到 CORS）
//   3. 按提示粘贴长期访问令牌（只存在这个标签页的内存里，刷新页面就没了）
//   4. 切到 Network 面板，筛选 "mcp"，然后在 Console 里运行：
//        await mcpInit()
//        await mcpList()
//        await mcpCall("homeassistant__GetLiveContext")
//        await mcpCall("intent__HassTurnOn", { name: "Living Room Light" })

(() => {
  const ENDPOINT = "/api/mcp";
  const token = prompt("粘贴 HA 长期访问令牌（eyJ 开头）");
  if (!token) {
    console.warn("没有令牌，已取消");
    return;
  }
  let nextId = 1;

  // 服务端可能回 JSON，也可能回 SSE（text/event-stream），两种都处理。
  async function parseResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    const text = await response.text();
    if (contentType.includes("text/event-stream")) {
      const dataLines = text.split("\n").filter((line) => line.startsWith("data:"));
      return JSON.parse(dataLines.at(-1).slice(5).trim());
    }
    return JSON.parse(text);
  }

  async function rpc(method, params) {
    const payload = { jsonrpc: "2.0", id: nextId++, method };
    if (params !== undefined) payload.params = params;
    const response = await fetch(ENDPOINT, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
      },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }
    const message = await parseResponse(response);
    if (message.error) {
      throw new Error(`JSON-RPC 错误: ${JSON.stringify(message.error)}`);
    }
    return message.result;
  }

  window.mcpInit = () =>
    rpc("initialize", {
      protocolVersion: "2025-03-26",
      capabilities: {},
      clientInfo: { name: "mcp_probe_browser", version: "0.1" },
    });

  window.mcpList = async () => {
    const { tools } = await rpc("tools/list");
    console.table(tools.map((t) => ({ name: t.name, description: t.description.slice(0, 80) })));
    return tools;
  };

  window.mcpCall = async (name, args = {}) => {
    const result = await rpc("tools/call", { name, arguments: args });
    if (result.isError) console.warn("工具返回了错误（isError=true）");
    for (const block of result.content ?? []) console.log(block.text ?? block);
    return result;
  };

  console.log("已就绪：mcpInit() / mcpList() / mcpCall(name, args)");
})();
