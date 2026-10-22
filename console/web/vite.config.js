import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时：先启动控制台（python -m apps.console），再 npm run dev；接口和登录都转发给控制台。
// 把终端里打印的登录链接的端口 8770 换成 5173 打开一次，拿到 cookie。
const console = 'http://127.0.0.1:8770'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    proxy: { '/api': console, '/login': console },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
