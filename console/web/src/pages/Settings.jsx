import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'
import { useApi } from '../api.js'
import Load from '../components/Load.jsx'
import Modal from '../components/Modal.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { PROCESS_NAMES, formatValue } from '../format.js'

const KINDS = { str: '文字', int: '整数', float: '数字', bool: '开关', choice: '选项', timezone: '时区', secret: '密钥' }
const FILTERS = [
  ['all', '全部'],
  ['pending_restart', '等待重启'],
  ['env', '.env 里设置过'],
  ['invalid', '不合法'],
]

export default function Settings() {
  const result = useApi('/api/settings', { refreshMs: 10000 })
  return (
    <>
      <h1>设置</h1>
      <p className="muted">
        只读。"已保存"是 .env 里的值（没写就是默认值），"状态"比较的是进程<strong>正在用</strong>的值。
      </p>
      <Load result={result}>{(data) => <SettingsView data={data} />}</Load>
    </>
  )
}

function SettingsView({ data }) {
  const [params, setParams] = useSearchParams()
  const group = params.get('group') ?? data.groups[0]
  const filter = params.get('filter') ?? 'all'
  const [selected, setSelected] = useState(null)
  // 按名字找：定时刷新后弹窗里显示的是最新的值。
  const selectedItem = data.items.find((item) => item.name === selected)

  const update = (changes) => {
    const next = new URLSearchParams(params)
    for (const [key, value] of Object.entries(changes)) {
      if (value == null) next.delete(key)
      else next.set(key, value)
    }
    setParams(next, { replace: true })
  }

  const counts = useMemo(() => {
    const result = {}
    for (const item of data.items) {
      if (item.state === 'pending_restart' || item.invalid) result[item.group] = (result[item.group] ?? 0) + 1
    }
    return result
  }, [data.items])

  // 筛选条件不是"全部"时跨分组显示，避免漏看别的分组里需要重启的参数。
  const items = data.items.filter((item) => {
    if (filter === 'all') return item.group === group
    if (filter === 'pending_restart') return item.state === 'pending_restart'
    if (filter === 'env') return item.source === 'env'
    if (filter === 'invalid') return Boolean(item.invalid)
    return true
  })

  return (
    <>
      <ProcessSummary processes={data.processes} />
      <div className="toolbar">
        {FILTERS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={`chip ${filter === key ? 'chip-active' : ''}`}
            onClick={() => update({ filter: key === 'all' ? null : key })}
          >
            {label}
          </button>
        ))}
      </div>
      {filter === 'all' && (
        <div className="tabs" role="tablist">
          {data.groups.map((name) => (
            <button
              key={name}
              type="button"
              role="tab"
              aria-selected={name === group}
              className={`tab ${name === group ? 'tab-active' : ''}`}
              onClick={() => update({ group: name })}
            >
              {name}
              {counts[name] ? <span className="tab-count">{counts[name]}</span> : null}
            </button>
          ))}
        </div>
      )}
      {items.length === 0 ? (
        <p className="empty">没有符合条件的参数。</p>
      ) : (
        <table className="table settings-table">
          <thead>
            <tr>
              <th>参数</th>
              <th>已保存</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.name} className="clickable" onClick={() => setSelected(item.name)}>
                <td>
                  <div className="setting-name">
                    <code>{item.name}</code>
                    {!item.editable && (
                      <span className="lock" title="不能在页面上修改">
                        🔒
                      </span>
                    )}
                    {filter !== 'all' && <span className="muted small"> · {item.group}</span>}
                  </div>
                  <div className="muted small">{item.description}</div>
                </td>
                <td>
                  <SavedValue item={item} />
                </td>
                <td>
                  <StatusBadge state={item.state} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {data.unrecognized_keys.length > 0 && (
        <section className="card">
          <h2>.env 里未识别的参数</h2>
          <p className="muted small">
            清单里没有登记。可能已经不用了，也可能被 SDK、子进程或脚本读取，所以只提示，不会自动删除。
          </p>
          <ul className="plain-list">
            {data.unrecognized_keys.map((key) => (
              <li key={key}>
                <code>{key}</code>
              </li>
            ))}
          </ul>
        </section>
      )}
      {selectedItem && <SettingModal item={selectedItem} onClose={() => setSelected(null)} />}
    </>
  )
}

function SavedValue({ item }) {
  if (item.invalid) return <span className="text-bad">不合法</span>
  return (
    <>
      <span className={item.secret ? '' : 'value'}>{formatValue(item.saved, { secret: item.secret })}</span>
      <span className="source">{item.source === 'env' ? '.env' : '默认'}</span>
    </>
  )
}

function ProcessSummary({ processes }) {
  return (
    <div className="grid two">
      {Object.entries(processes).map(([name, process]) => {
        const changed =
          process.running_fingerprint && process.saved_fingerprint && process.running_fingerprint !== process.saved_fingerprint
        return (
          <section key={name} className="card slim">
            <header className="card-header">
              <h2>{PROCESS_NAMES[name]}</h2>
              <StatusBadge state={process.state} title={process.detail} />
            </header>
            <div className="muted small">
              配置指纹：已保存 <code>{process.saved_fingerprint ?? '—'}</code> · 正在用{' '}
              <code>{process.running_fingerprint ?? '不知道'}</code>
              {changed && <span className="text-warn"> · 不一致，重启后生效</span>}
            </div>
          </section>
        )
      })}
    </div>
  )
}

function SettingModal({ item, onClose }) {
  const range =
    item.minimum != null || item.maximum != null ? `${item.minimum ?? '−∞'} ～ ${item.maximum ?? '∞'}` : null
  return (
    <Modal title={item.name} onClose={onClose}>
      <p>{item.description}</p>
      <dl className="details">
        <dt>分组</dt>
        <dd>{item.group}</dd>
        <dt>类型</dt>
        <dd>{KINDS[item.kind] ?? item.kind}</dd>
        {item.choices.length > 0 && (
          <>
            <dt>可选值</dt>
            <dd>
              {item.choices.map((choice) => (
                <code key={choice} className="choice">
                  {choice}
                </code>
              ))}
            </dd>
          </>
        )}
        {range && (
          <>
            <dt>范围</dt>
            <dd>{range}</dd>
          </>
        )}
        {!item.secret && (
          <>
            <dt>默认值</dt>
            <dd>{formatValue(item.default)}</dd>
          </>
        )}
        <dt>已保存</dt>
        <dd>
          {item.invalid ? (
            <span className="text-bad">{item.invalid}</span>
          ) : (
            <>
              {formatValue(item.saved, { secret: item.secret })}
              <span className="source">{item.source === 'env' ? '来自 .env' : '默认值（.env 里没写）'}</span>
            </>
          )}
        </dd>
        <dt>读取它的进程</dt>
        <dd>{item.readers.map((reader) => PROCESS_NAMES[reader] ?? reader).join('、')}</dd>
        <dt>能否在页面修改</dt>
        <dd>{item.editable ? '可以（D5 开放编辑）' : '不能：安全边界、连接地址或密钥，只能手动改 .env'}</dd>
      </dl>
      <h3>正在生效</h3>
      {item.processes.length === 0 ? (
        <p className="muted">没有常驻进程读取它；按需运行的程序（例如文字 CLI）下次启动时读取。</p>
      ) : (
        <table className="table compact">
          <thead>
            <tr>
              <th>进程</th>
              <th>正在用</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            {item.processes.map((row) => (
              <tr key={row.process}>
                <td>{PROCESS_NAMES[row.process]}</td>
                <td>{row.state === 'unknown' ? '不知道（进程没在运行）' : formatValue(row.effective, { secret: item.secret })}</td>
                <td>
                  <StatusBadge state={row.state} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Modal>
  )
}
