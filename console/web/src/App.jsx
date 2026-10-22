import { NavLink, Route, Routes } from 'react-router'
import Overview from './pages/Overview.jsx'
import Settings from './pages/Settings.jsx'

const PAGES = [
  { path: '/', label: '总览', element: <Overview /> },
  { path: '/settings', label: '设置', element: <Settings /> },
]

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          Alpha <span>控制台</span>
        </div>
        <nav>
          {PAGES.map((page) => (
            <NavLink key={page.path} to={page.path} end className="nav-link">
              {page.label}
            </NavLink>
          ))}
        </nav>
        <p className="sidebar-note">只读 · 只在本机</p>
      </aside>
      <main className="content">
        <Routes>
          {PAGES.map((page) => (
            <Route key={page.path} path={page.path} element={page.element} />
          ))}
          <Route path="*" element={<p className="empty">没有这个页面。</p>} />
        </Routes>
      </main>
    </div>
  )
}
