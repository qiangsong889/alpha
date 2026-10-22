export function formatTime(seconds) {
  if (seconds == null) return '—'
  return new Date(seconds * 1000).toLocaleString('zh-CN', { hour12: false })
}

export function formatAgo(seconds, now = Date.now() / 1000) {
  if (seconds == null) return '—'
  const diff = Math.max(0, now - seconds)
  if (diff < 60) return `${Math.round(diff)} 秒前`
  if (diff < 3600) return `${Math.round(diff / 60)} 分钟前`
  if (diff < 86400) return `${Math.round(diff / 3600)} 小时前`
  return `${Math.round(diff / 86400)} 天前`
}

export function formatAge(seconds) {
  if (seconds == null) return '—'
  if (seconds < 1) return '不到 1 秒' // 两个时钟有微小偏差时可能是负数
  if (seconds < 60) return `${Math.round(seconds)} 秒`
  return `${Math.round(seconds / 60)} 分钟`
}

export function formatUsd(value) {
  if (value == null) return '—'
  const number = Number(value)
  if (number === 0) return '$0'
  if (number < 0.0001) return '<$0.0001'
  return number < 0.01 ? `$${number.toFixed(4)}` : `$${number.toFixed(3)}`
}

// 页面上显示参数值：密钥只有"已设置 / 未设置"，空字符串显示为"（空）"。
export function formatValue(value, { secret = false } = {}) {
  if (secret) return value ? '已设置' : '未设置'
  if (value === null || value === undefined) return '（未设置）'
  if (value === '') return '（空）'
  if (typeof value === 'boolean') return value ? '开' : '关'
  return String(value)
}

export const PROCESS_NAMES = { brain: '大脑', voice: '语音', personal: '个人服务', cli: '文字 CLI' }
