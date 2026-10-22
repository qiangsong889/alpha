// 接口的加载 / 出错状态。401 说明登录 cookie 失效（例如控制台重启后令牌换了）。
export default function Load({ result, children }) {
  const { data, error, loading } = result
  if (loading && !data) return <p className="empty">加载中…</p>
  if (error && !data) {
    return (
      <div className="notice notice-bad">
        {error.status === 401 ? '登录已失效：控制台重启后令牌会更新，请用终端里最新打印的链接重新打开。' : error.message}
      </div>
    )
  }
  return (
    <>
      {error && <div className="notice notice-warn">刷新失败，下面是上一次的数据：{error.message}</div>}
      {children(data)}
    </>
  )
}
