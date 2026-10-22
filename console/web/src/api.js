import { useCallback, useEffect, useState } from 'react'

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

// 只读接口：同源请求，cookie 由浏览器自动带上（HttpOnly，脚本读不到）。
export async function getJson(path) {
  const response = await fetch(path, { credentials: 'same-origin', headers: { Accept: 'application/json' } })
  let body = null
  try {
    body = await response.json()
  } catch {
    // 非 JSON（例如控制台已经关闭）
  }
  if (!response.ok) throw new ApiError(response.status, body?.detail || `请求失败（${response.status}）`)
  return body
}

// 读取一个接口；refreshMs 大于 0 时在页面可见期间定时刷新。
export function useApi(path, { refreshMs = 0 } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: true })

  const load = useCallback(async () => {
    try {
      const data = await getJson(path)
      setState({ data, error: null, loading: false })
    } catch (error) {
      setState((previous) => ({ ...previous, error, loading: false }))
    }
  }, [path])

  useEffect(() => {
    load()
    if (!refreshMs) return undefined
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') load()
    }, refreshMs)
    return () => clearInterval(timer)
  }, [load, refreshMs])

  return { ...state, reload: load }
}
