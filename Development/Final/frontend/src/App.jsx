import { useState, useEffect, useRef, useCallback } from 'react'
import './index.css'

const API_BASE = 'http://localhost:8000'
const WS_BASE = 'ws://localhost:8000'

const ZONE_NAMES = [
  'No Leak / Healthy',
  'Zone 1 — Intake',
  'Zone 2 — Charge Air',
  'Zone 3 — CAC/Manifold',
  'Zone 4 — Exhaust Manifold',
  'Zone 5 — DPF/SCR',
]

const EF_CHANNELS = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out', 'T_exh', 'dP_dpf']

const SENSOR_UNITS = {
  RPM: 'rpm', MAF: 'kg/h', MAP_intake: 'kPa', MAP_boost: 'kPa',
  MAP_cac_in: 'kPa', MAP_cac_out: 'kPa', T_intake: '°C', T_boost: '°C',
  T_cac_out: '°C', T_exh_manifold: '°C', T_dpf_in: '°C', T_dpf_out: '°C',
  fuel_qty: 'mg/str', dP_dpf: 'kPa',
}

// ─── Sidebar Component ──────────────────────────────────────────
function Sidebar({ activePage, setActivePage, isConnected }) {
  const navItems = [
    { id: 'dashboard', icon: '📊', label: 'Dashboard' },
    { id: 'engine', icon: '🔧', label: 'Engine Diagram' },
    { id: 'history', icon: '📋', label: 'History' },
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">🔍</div>
          <div>
            <h1>LeakSense Twin</h1>
            <span>Cat C18 Engine Diagnostics</span>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map(item => (
          <button
            key={item.id}
            id={`nav-${item.id}`}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => setActivePage(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="engine-badge">
          <span className="dot" style={{
            background: isConnected ? 'var(--accent-emerald)' : 'var(--accent-red)',
          }}></span>
          <span>CAT C18 • ADEM A4</span>
        </div>
      </div>
    </aside>
  )
}

// ─── Go/No-Go Indicator ─────────────────────────────────────────
function GoNoGoIndicator({ status }) {
  const isGo = status === 'GO'
  return (
    <div className={`card go-nogo-card ${isGo ? 'go-indicator' : 'nogo-indicator'}`}>
      <div className="card-header">
        <span className="card-title">Status</span>
        <span className="card-icon">{isGo ? '✅' : '🚨'}</span>
      </div>
      <div className={`go-nogo-status ${isGo ? 'go' : 'nogo'}`}>
        {status}
      </div>
      <div className="go-nogo-subtitle">
        {isGo ? 'Engine Healthy' : 'Leak Detected'}
      </div>
    </div>
  )
}

// ─── Confidence Ring ─────────────────────────────────────────────
function ConfidenceRing({ confidence, leakDetected }) {
  const pct = Math.round(confidence * 100)
  const radius = 54
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (pct / 100) * circumference

  let color = 'var(--accent-emerald)'
  if (confidence > 0.7) color = 'var(--accent-red)'
  else if (confidence > 0.4) color = 'var(--accent-amber)'

  return (
    <div className="card confidence-card">
      <div className="card-header">
        <span className="card-title">Confidence</span>
        <span className="card-icon">🎯</span>
      </div>
      <div className="confidence-ring">
        <svg viewBox="0 0 128 128">
          <circle className="ring-bg" cx="64" cy="64" r={radius} />
          <circle
            className="ring-fill"
            cx="64" cy="64" r={radius}
            stroke={color}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="confidence-value" style={{ color }}>{pct}%</div>
      </div>
      <div className="confidence-label">
        {leakDetected ? 'Anomaly Detected' : 'Normal Operation'}
      </div>
    </div>
  )
}

// ─── Leak Alert Card ─────────────────────────────────────────────
function LeakAlertCard({ data }) {
  const getSeverityClass = (sev) => {
    const map = { NONE: 'severity-none', SMALL: 'severity-small', MEDIUM: 'severity-medium', CRITICAL: 'severity-critical' }
    return map[sev] || 'severity-none'
  }

  return (
    <div className="card alert-card">
      <div className="card-header">
        <span className="card-title">Leak Analysis</span>
        <span className="card-icon">⚡</span>
      </div>
      <div className="alert-content">
        <div className="alert-zone">{data.suspected_zone || ZONE_NAMES[0]}</div>
        <div>
          <span className={`alert-severity ${getSeverityClass(data.severity)}`}>
            {data.severity || 'NONE'}
          </span>
        </div>
        {data.flow_loss_pct > 0 && (
          <div className="alert-flow-loss">
            Estimated Flow Loss: <strong>{data.flow_loss_pct?.toFixed(1)}%</strong>
          </div>
        )}
        <div className="alert-action">
          {data.recommended_action || 'System operating within normal parameters.'}
        </div>
      </div>
    </div>
  )
}

// ─── Sensor Grid Card ────────────────────────────────────────────
function SensorGrid({ sensors }) {
  const entries = Object.entries(sensors || {})
  return (
    <div className="card sensors-card">
      <div className="card-header">
        <span className="card-title">Live Sensors</span>
        <span className="card-icon">📡</span>
      </div>
      <div className="sensor-grid">
        {entries.map(([key, val]) => (
          <div className="sensor-item" key={key}>
            <div className="sensor-label">{key.replace(/_/g, ' ')}</div>
            <div className="sensor-value">
              {typeof val === 'number' ? val.toFixed(1) : val}
              <span className="sensor-unit">{SENSOR_UNITS[key] || ''}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Energy Field Heatmap ────────────────────────────────────────
function EnergyFieldHeatmap({ efData }) {
  const matrix = efData?.matrix || Array(6).fill(Array(6).fill(0))

  const getColor = (val) => {
    const v = Math.max(-1, Math.min(1, val))
    if (v >= 0) {
      const r = Math.round(255 * v)
      const g = Math.round(100 * (1 - v))
      const b = Math.round(50 * (1 - v))
      return `rgb(${r}, ${g}, ${b})`
    } else {
      const r = Math.round(50 * (1 + v))
      const g = Math.round(100 * (1 + v))
      const b = Math.round(255 * (-v))
      return `rgb(${r}, ${g}, ${b})`
    }
  }

  return (
    <div className="card heatmap-card">
      <div className="card-header">
        <span className="card-title">Energy Field</span>
        <span className="card-icon">🌡️</span>
      </div>
      <div className="heatmap-container">
        {matrix.map((row, i) => (
          <div className="heatmap-row" key={i}>
            <div className="heatmap-label">{EF_CHANNELS[i]}</div>
            {(Array.isArray(row) ? row : []).map((val, j) => (
              <div
                className="heatmap-cell"
                key={j}
                style={{ background: getColor(val || 0) }}
              >
                {typeof val === 'number' ? val.toFixed(2) : '0'}
              </div>
            ))}
          </div>
        ))}
        <div className="heatmap-col-labels">
          {EF_CHANNELS.map(ch => (
            <div className="heatmap-col-label" key={ch}>{ch}</div>
          ))}
        </div>
      </div>
      <div className="ef-metrics">
        <div className="ef-metric">
          <div className="ef-metric-value" style={{
            color: (efData?.global_deviation || 0) > 3 ? 'var(--accent-red)' : 'var(--accent-emerald)'
          }}>
            {(efData?.global_deviation || 0).toFixed(2)}
          </div>
          <div className="ef-metric-label">Global Deviation</div>
        </div>
        <div className="ef-metric">
          <div className="ef-metric-value" style={{
            color: (efData?.cosine_similarity || 1) < 0.9 ? 'var(--accent-amber)' : 'var(--accent-cyan)'
          }}>
            {(efData?.cosine_similarity || 1).toFixed(3)}
          </div>
          <div className="ef-metric-label">Cosine Similarity</div>
        </div>
        <div className="ef-metric">
          <div className="ef-metric-value" style={{ color: 'var(--accent-purple)' }}>
            {efData?.most_disrupted_sensor || 'N/A'}
          </div>
          <div className="ef-metric-label">Most Disrupted</div>
        </div>
      </div>
    </div>
  )
}

// ─── Residuals Card ──────────────────────────────────────────────
function ResidualsCard({ residuals }) {
  const entries = Object.entries(residuals || {})

  const getBarColor = (val) => {
    const abs = Math.abs(val)
    if (abs > 20) return 'var(--accent-red)'
    if (abs > 10) return 'var(--accent-amber)'
    return 'var(--accent-cyan)'
  }

  return (
    <div className="card residuals-card">
      <div className="card-header">
        <span className="card-title">Digital Twin Residuals</span>
        <span className="card-icon">🔬</span>
      </div>
      <div className="residuals-grid">
        {entries.map(([key, val]) => (
          <div className="residual-item" key={key}>
            <div className="residual-name">{key.replace('res_', '').replace(/_/g, ' ')}</div>
            <div className="residual-value" style={{ color: getBarColor(val) }}>
              {typeof val === 'number' ? val.toFixed(2) : val}
            </div>
            <div className="residual-bar">
              <div
                className="residual-bar-fill"
                style={{
                  width: `${Math.min(Math.abs(val || 0) * 2, 100)}%`,
                  background: getBarColor(val || 0),
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Zone Probabilities ──────────────────────────────────────────
function ZoneProbabilities({ probs }) {
  const probArray = probs || [1, 0, 0, 0, 0, 0]
  const colors = [
    'var(--accent-emerald)', 'var(--accent-cyan)', 'var(--accent-blue)',
    'var(--accent-purple)', 'var(--accent-amber)', 'var(--accent-red)'
  ]

  return (
    <div className="card zone-probs-card">
      <div className="card-header">
        <span className="card-title">Zone Probabilities</span>
        <span className="card-icon">📊</span>
      </div>
      <div className="zone-bars">
        {ZONE_NAMES.map((name, i) => (
          <div className="zone-bar-item" key={i}>
            <div className="zone-bar-label">{name.split('—')[0].trim()}</div>
            <div className="zone-bar-track">
              <div
                className="zone-bar-fill"
                style={{
                  width: `${(probArray[i] || 0) * 100}%`,
                  background: colors[i],
                }}
              >
                {(probArray[i] || 0) > 0.1 ? `${((probArray[i] || 0) * 100).toFixed(0)}%` : ''}
              </div>
            </div>
            <div className="zone-bar-value">{((probArray[i] || 0) * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Engine Diagram ──────────────────────────────────────────────
function EngineDiagram({ zoneIdx, leakDetected }) {
  const zones = [
    { id: 1, name: 'Zone 1', desc: 'Intake — Airflow meter → Compressor' },
    { id: 2, name: 'Zone 2', desc: 'Charge Air — Compressor → CAC' },
    { id: 3, name: 'Zone 3', desc: 'CAC/Manifold — CAC → Intake ports' },
    { id: 4, name: 'Zone 4', desc: 'Exhaust — Manifold → Turbine' },
    { id: 5, name: 'Zone 5', desc: 'Aftertreatment — DPF → SCR' },
  ]

  return (
    <div className="card engine-card">
      <div className="card-header">
        <span className="card-title">Cat C18 Engine — Zone Map</span>
        <span className="card-icon">🔧</span>
      </div>
      <div className="engine-diagram">
        {zones.map((zone, idx) => (
          <div key={zone.id} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div
              className={`engine-zone ${leakDetected && zoneIdx === zone.id ? 'leak' : 'healthy'}`}
            >
              <div className="engine-zone-name">{zone.name}</div>
              <div className="engine-zone-desc">{zone.desc}</div>
            </div>
            {idx < zones.length - 1 && <span className="zone-arrow">→</span>}
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── History Table ───────────────────────────────────────────────
function HistoryTable({ history }) {
  return (
    <div className="card history-card">
      <div className="card-header">
        <span className="card-title">Prediction History</span>
        <span className="card-icon">📋</span>
      </div>
      <table className="history-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Status</th>
            <th>Confidence</th>
            <th>Zone</th>
            <th>Severity</th>
            <th>Go/No-Go</th>
          </tr>
        </thead>
        <tbody>
          {(history || []).slice(-15).reverse().map((item, i) => (
            <tr key={i}>
              <td>{item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '-'}</td>
              <td>
                <span className={`status-dot ${item.leak_detected ? 'leak' : 'healthy'}`}></span>
                {item.leak_detected ? 'LEAK' : 'OK'}
              </td>
              <td>{(item.confidence * 100).toFixed(1)}%</td>
              <td>{item.suspected_zone?.split('—')[0]?.trim() || '-'}</td>
              <td>
                <span className={`alert-severity ${
                  item.severity === 'CRITICAL' ? 'severity-critical' :
                  item.severity === 'MEDIUM' ? 'severity-medium' :
                  item.severity === 'SMALL' ? 'severity-small' : 'severity-none'
                }`}>{item.severity || 'NONE'}</span>
              </td>
              <td style={{ color: item.go_no_go === 'GO' ? 'var(--accent-emerald)' : 'var(--accent-red)', fontWeight: 700 }}>
                {item.go_no_go}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Main App ────────────────────────────────────────────────────
function App() {
  const [activePage, setActivePage] = useState('dashboard')
  const [predictionData, setPredictionData] = useState(null)
  const [history, setHistory] = useState([])
  const [isConnected, setIsConnected] = useState(false)
  const [isDemoRunning, setIsDemoRunning] = useState(false)
  const [simZone, setSimZone] = useState(0)
  const [simSeverity, setSimSeverity] = useState(0)
  const wsRef = useRef(null)
  const demoWsRef = useRef(null)

  // Fetch history on load
  useEffect(() => {
    fetch(`${API_BASE}/api/history?limit=50`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) setHistory(data)
      })
      .catch(() => {})
  }, [])

  // Demo WebSocket
  const startDemo = useCallback(() => {
    if (demoWsRef.current) {
      demoWsRef.current.close()
    }
    const ws = new WebSocket(`${WS_BASE}/ws/demo`)
    ws.onopen = () => {
      setIsConnected(true)
      setIsDemoRunning(true)
    }
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      setPredictionData(data)
      setHistory(prev => [...prev.slice(-100), data])
    }
    ws.onclose = () => {
      setIsConnected(false)
      setIsDemoRunning(false)
    }
    ws.onerror = () => {
      setIsConnected(false)
      setIsDemoRunning(false)
    }
    demoWsRef.current = ws
  }, [])

  const stopDemo = useCallback(() => {
    if (demoWsRef.current) {
      demoWsRef.current.close()
      demoWsRef.current = null
    }
    setIsDemoRunning(false)
    setIsConnected(false)
  }, [])

  // Manual simulation
  const runSimulation = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rpm: 1800,
          leak_zone: simZone,
          leak_severity: simSeverity,
        }),
      })
      const data = await res.json()
      setPredictionData(data)
      setHistory(prev => [...prev.slice(-100), data])
    } catch (err) {
      console.error('Simulation error:', err)
    }
  }, [simZone, simSeverity])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (demoWsRef.current) demoWsRef.current.close()
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  const data = predictionData || {
    leak_detected: false,
    confidence: 0,
    suspected_zone: ZONE_NAMES[0],
    suspected_zone_idx: 0,
    zone_probabilities: [1, 0, 0, 0, 0, 0],
    severity: 'NONE',
    flow_loss_pct: 0,
    go_no_go: 'GO',
    recommended_action: 'Connect to backend to start monitoring.',
    residuals: {},
    energy_field: { matrix: Array(6).fill(Array(6).fill(0)), global_deviation: 0, cosine_similarity: 1, most_disrupted_sensor: 'N/A' },
    sensors: {},
  }

  return (
    <div className="app-container">
      <Sidebar activePage={activePage} setActivePage={setActivePage} isConnected={isConnected} />

      <main className="main-content">
        <div className="main-header">
          <h2>
            {activePage === 'dashboard' && '📊 Real-Time Dashboard'}
            {activePage === 'engine' && '🔧 Engine Diagram'}
            {activePage === 'history' && '📋 Detection History'}
          </h2>
          <div className="header-controls">
            <span className={`header-badge ${isConnected ? 'badge-connected' : 'badge-disconnected'}`}>
              <span className="status-dot" style={{
                background: isConnected ? 'var(--accent-emerald)' : 'var(--accent-red)',
                width: '6px', height: '6px'
              }}></span>
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
        </div>

        {/* Demo Controls — always visible */}
        <div className="dashboard-grid" style={{ marginBottom: '20px' }}>
          <div className="demo-controls">
            <button
              id="demo-toggle"
              className={`demo-btn ${isDemoRunning ? 'active' : ''}`}
              onClick={isDemoRunning ? stopDemo : startDemo}
            >
              {isDemoRunning ? '⏹ Stop Demo' : '▶ Start Live Demo'}
            </button>

            <select className="demo-select" value={simZone} onChange={e => setSimZone(Number(e.target.value))}>
              <option value={0}>Healthy (No Leak)</option>
              <option value={1}>Zone 1 — Intake</option>
              <option value={2}>Zone 2 — Charge Air</option>
              <option value={3}>Zone 3 — CAC/Manifold</option>
              <option value={4}>Zone 4 — Exhaust</option>
              <option value={5}>Zone 5 — DPF/SCR</option>
            </select>

            <select className="demo-select" value={simSeverity} onChange={e => setSimSeverity(Number(e.target.value))}>
              <option value={0}>No Fault</option>
              <option value={1}>Small (2%)</option>
              <option value={2}>Medium (8%)</option>
              <option value={3}>Large (15%)</option>
            </select>

            <button id="simulate-btn" className="demo-btn danger" onClick={runSimulation}>
              🧪 Inject & Predict
            </button>
          </div>
        </div>

        {/* Dashboard Page */}
        {activePage === 'dashboard' && (
          <div className="dashboard-grid">
            <GoNoGoIndicator status={data.go_no_go} />
            <ConfidenceRing confidence={data.confidence} leakDetected={data.leak_detected} />
            <LeakAlertCard data={data} />

            <SensorGrid sensors={data.sensors} />
            <EnergyFieldHeatmap efData={data.energy_field} />

            <ResidualsCard residuals={data.residuals} />
            <ZoneProbabilities probs={data.zone_probabilities} />
          </div>
        )}

        {/* Engine Diagram Page */}
        {activePage === 'engine' && (
          <div className="dashboard-grid">
            <EngineDiagram zoneIdx={data.suspected_zone_idx} leakDetected={data.leak_detected} />
            <GoNoGoIndicator status={data.go_no_go} />
            <ConfidenceRing confidence={data.confidence} leakDetected={data.leak_detected} />
            <LeakAlertCard data={data} />
            <ZoneProbabilities probs={data.zone_probabilities} />
          </div>
        )}

        {/* History Page */}
        {activePage === 'history' && (
          <div className="dashboard-grid">
            <HistoryTable history={history} />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
