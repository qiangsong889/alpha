// 所有状态的中文名和颜色集中在这里：页面上看到的说法保持一致。
const STATES = {
  running: ['运行中', 'good'],
  starting: ['启动中', 'warn'],
  stale: ['心跳过期', 'bad'],
  stopped: ['已停止', 'muted'],
  not_started: ['未启动', 'muted'],
  unreadable: ['状态不可读', 'bad'],
  unreachable: ['连不上', 'bad'],
  connected: ['已连接', 'good'],
  reconnecting: ['重连中', 'warn'],
  available: ['可用', 'good'],
  unavailable: ['不可用', 'bad'],
  enabled: ['已开启', 'good'],
  disabled: ['未开启', 'muted'],
  applied: ['已生效', 'good'],
  pending_restart: ['等待重启', 'warn'],
  unknown: ['不知道', 'muted'],
  next_start: ['下次启动时读取', 'muted'],
}

export function stateLabel(state) {
  return STATES[state]?.[0] ?? state
}

export default function StatusBadge({ state, title }) {
  const [label, tone] = STATES[state] ?? [state ?? '—', 'muted']
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {label}
    </span>
  )
}
