import { Link } from 'react-router'
import { useApi } from '../api.js'
import Load from '../components/Load.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { formatAge, formatAgo, formatTime, formatUsd } from '../format.js'

const COMPONENTS = { home_assistant: 'Home Assistant', personal: '个人服务', archive: '对话存档', auto_memory: '自动记忆' }
const DELIVERY = { played: '播放完', interrupted: '被打断', not_played: '没播放', not_reported: '没有回报' }
const SPEAKER = { tts: 'TTS 原样朗读', omni: 'Omni（可能改写）' }
const INPUT = { asr: 'Silero + 独立 ASR', omni: 'Omni（旧路径）' }

export default function Overview() {
  const result = useApi('/api/overview', { refreshMs: 5000 })
  return (
    <>
      <h1>总览</h1>
      <Load result={result}>
        {(data) => (
          <>
            <Attention attention={data.attention} />
            <div className="grid two">
              <BrainCard status={data.processes.brain} now={data.now} />
              <VoiceCard status={data.processes.voice} />
            </div>
            <div className="grid two">
              <TodayCard today={data.today} />
              <CostCard costs={data.costs} />
            </div>
          </>
        )}
      </Load>
    </>
  )
}

function Row({ label, children }) {
  return (
    <div className="row">
      <span className="row-label">{label}</span>
      <span className="row-value">{children}</span>
    </div>
  )
}

function ProcessHeader({ title, status }) {
  return (
    <header className="card-header">
      <h2>{title}</h2>
      <StatusBadge state={status.state} title={status.detail} />
    </header>
  )
}

function Detail({ status }) {
  return status.detail ? <p className="muted small">{status.detail}</p> : null
}

function BrainCard({ status, now }) {
  const running = status.state === 'running'
  return (
    <section className="card">
      <ProcessHeader title="大脑" status={status} />
      <Detail status={status} />
      {running && (
        <>
          <Row label="模型">
            <code>{status.model}</code>
          </Row>
          <Row label="启动时间">{formatTime(status.started_at)}</Row>
          <Row label="最后一轮">{formatAgo(status.last_turn_at, now)}</Row>
          <Row label="配置指纹">
            <code>{status.settings_fingerprint}</code>
          </Row>
          <h3>依赖</h3>
          {Object.entries(status.components ?? {}).map(([name, component]) => (
            <Row key={name} label={COMPONENTS[name] ?? name}>
              <StatusBadge state={component.state} />
              {component.checked_at && <span className="muted small"> 检查于 {formatAgo(component.checked_at, now)}</span>}
            </Row>
          ))}
          <p className="muted small">HA 和个人服务是最近一次加载工具时的结果，不是实时探测。</p>
        </>
      )}
    </section>
  )
}

function VoiceCard({ status }) {
  const hasData = status.pid != null
  return (
    <section className="card">
      <ProcessHeader title="语音" status={status} />
      <Detail status={status} />
      {hasData && (
        <>
          <Row label="收音">{INPUT[status.input] ?? status.input}</Row>
          <Row label="播报">{SPEAKER[status.speaker] ?? status.speaker}</Row>
          <Row label="连接">
            {status.state === 'running' ? <StatusBadge state={status.connection} /> : <span className="muted">—</span>}
          </Row>
          {status.last_error && (
            <Row label="最后一次错误">
              <code>{status.last_error}</code>
            </Row>
          )}
          <Row label="心跳">{formatAge(status.age_s)}前</Row>
          <Row label="启动时间">{formatTime(status.started_at)}</Row>
          <Row label="配置指纹">
            <code>{status.settings_fingerprint}</code>
          </Row>
          {status.state !== 'running' && <p className="muted small">上面是进程最后一次写入的信息，不代表现在。</p>}
        </>
      )}
      {status.state === 'running' && <p className="muted small">连接状态是语音层自己判断的，可能比实际晚几秒。</p>}
    </section>
  )
}

function Unavailable({ section }) {
  return <p className="muted">没有数据：{section.detail}</p>
}

function TodayCard({ today }) {
  return (
    <section className="card">
      <header className="card-header">
        <h2>今天的对话</h2>
      </header>
      {!today.available ? (
        <Unavailable section={today} />
      ) : (
        <>
          <div className="stats">
            <Stat label="轮数" value={today.turns} />
            <Stat label="会话" value={today.sessions} />
            <Stat label="出错" value={today.errors} tone={today.errors ? 'bad' : undefined} />
            <Stat label="大脑平均耗时" value={today.avg_brain_ms == null ? '—' : `${(today.avg_brain_ms / 1000).toFixed(1)} 秒`} />
          </div>
          <h3>回答的播放结果</h3>
          {Object.keys(today.delivery).length === 0 && <p className="muted">今天还没有对话。</p>}
          {Object.entries(today.delivery).map(([name, count]) => (
            <Row key={name} label={DELIVERY[name] ?? name}>
              {count}
            </Row>
          ))}
        </>
      )}
    </section>
  )
}

function CostCard({ costs }) {
  return (
    <section className="card">
      <header className="card-header">
        <h2>今天的费用</h2>
      </header>
      {!costs.available ? (
        <Unavailable section={costs} />
      ) : (
        <>
          <div className="stats">
            <Stat label="已知" value={formatUsd(costs.known_usd)} />
            <Stat label="参考估算" value={formatUsd(costs.estimate_usd)} />
            <Stat label="待核对" value={`${costs.to_verify} 笔`} tone={costs.to_verify ? 'warn' : undefined} />
            <Stat label="平均每轮（已知）" value={formatUsd(costs.known_per_turn_usd)} />
          </div>
          {Object.keys(costs.components).length > 0 && (
            <table className="table compact">
              <thead>
                <tr>
                  <th>组件</th>
                  <th className="number">已知</th>
                  <th className="number">参考估算</th>
                  <th className="number">待核对</th>
                  <th className="number">记录</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(costs.components).map(([name, row]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td className="number">{formatUsd(row.known)}</td>
                    <td className="number">{Number(row.estimate) ? formatUsd(row.estimate) : '—'}</td>
                    <td className="number">{row.to_verify || '—'}</td>
                    <td className="number">{row.events}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="muted small">
            估算值，不是账单。"待核对"是缺少最终用量或价格的请求，金额不能当成零。平均每轮 = 今天的已知费用 ÷
            今天交给大脑的轮数；费用里也包括没有交给大脑的部分（例如 Jev 判断、语音识别）。
          </p>
        </>
      )}
    </section>
  )
}

function Stat({ label, value, tone }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ''}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function Attention({ attention }) {
  const items = []
  const unknown = attention.unknown_operations
  if (unknown.available && unknown.count > 0) {
    items.push(['bad', `${unknown.count} 个操作结果未知：可能已经执行，也可能没有，需要你确认。`])
  }
  for (const [process, problems] of Object.entries(attention.saved_problems)) {
    const name = process === 'brain' ? '大脑' : '语音'
    for (const error of problems.errors) items.push(['bad', `按现在保存的配置，${name}重启时会拒绝启动：${error}`])
    for (const warning of problems.warnings) items.push(['warn', `${name}：${warning}`])
  }
  if (attention.pending_restart.length > 0) {
    items.push([
      'warn',
      <>
        {attention.pending_restart.length} 个参数已保存、但要重启才会生效：{attention.pending_restart.join('、')}。
        <Link to="/settings?filter=pending_restart">查看</Link>
      </>,
    ])
  }
  if (attention.unrecognized_keys.length > 0) {
    items.push([
      'muted',
      `.env 里有 ${attention.unrecognized_keys.length} 个未识别的参数：${attention.unrecognized_keys.join('、')}（可能已经不用，也可能被 SDK 或脚本读取）。`,
    ])
  }
  if (items.length === 0) return null
  return (
    <section className="card attention">
      <header className="card-header">
        <h2>需要注意</h2>
      </header>
      <ul>
        {items.map(([tone, text], index) => (
          <li key={index} className={`attention-${tone}`}>
            {text}
          </li>
        ))}
      </ul>
    </section>
  )
}
