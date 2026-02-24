import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const API = ''
const MODEL_COLORS = {
    dnn: '#06b6d4',
    cnn: '#8b5cf6',
    lstm: '#10b981',
    cnn_lstm: '#f59e0b',
    deep_fin_dlp: '#ec4899',
}
const MODEL_NAMES = {
    dnn: 'Basic DNN',
    cnn: '1D-CNN',
    lstm: 'BiLSTM',
    cnn_lstm: 'CNN-BiLSTM',
    deep_fin_dlp: 'DeepFinDLP',
}

export default function Training() {
    const [histories, setHistories] = useState({})
    const [activeMetric, setActiveMetric] = useState('val_acc')
    const [activeModels, setActiveModels] = useState(new Set(['dnn', 'cnn', 'lstm', 'cnn_lstm', 'deep_fin_dlp']))
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch(`${API}/api/training-histories`).then(r => r.json()).then(d => {
            setHistories(d)
            setLoading(false)
        }).catch(() => setLoading(false))
    }, [])

    const metricOptions = [
        { key: 'val_acc', label: 'Val Accuracy' },
        { key: 'val_loss', label: 'Val Loss' },
        { key: 'train_loss', label: 'Train Loss' },
        { key: 'train_acc', label: 'Train Accuracy' },
        { key: 'val_f1', label: 'Val F1' },
        { key: 'train_f1', label: 'Train F1' },
        { key: 'lr', label: 'Learning Rate' },
    ]

    const toggleModel = (modelId) => {
        setActiveModels(prev => {
            const next = new Set(prev)
            if (next.has(modelId)) next.delete(modelId)
            else next.add(modelId)
            return next
        })
    }

    // Build unified chart data
    const maxEpochs = Math.max(...Object.values(histories).map(h => (h.train_loss || []).length), 0)
    const chartData = Array.from({ length: maxEpochs }, (_, i) => {
        const point = { epoch: i + 1 }
        Object.entries(histories).forEach(([model, h]) => {
            if (activeModels.has(model)) {
                const metricMap = {
                    val_acc: h.val_acc, val_loss: h.val_loss,
                    train_loss: h.train_loss, train_acc: h.train_acc,
                    val_f1: h.val_f1, train_f1: h.train_f1, lr: h.lr,
                }
                const arr = metricMap[activeMetric]
                if (arr && arr[i] !== undefined) {
                    point[model] = arr[i]
                }
            }
        })
        return point
    }).filter(p => Object.keys(p).length > 1)

    if (loading) return <div className="loading"><div className="loading-spinner" />Loading training histories...</div>

    return (
        <div>
            <motion.section className="section" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="section-header">
                    <h2>Training Curves</h2>
                    <p>Track model convergence across epochs</p>
                </div>

                {/* Metric selector */}
                <div className="tabs" style={{ flexWrap: 'wrap' }}>
                    {metricOptions.map(m => (
                        <button key={m.key} className={`tab ${activeMetric === m.key ? 'active' : ''}`} onClick={() => setActiveMetric(m.key)}>
                            {m.label}
                        </button>
                    ))}
                </div>

                {/* Model toggles */}
                <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', flexWrap: 'wrap' }}>
                    {Object.entries(MODEL_NAMES).map(([id, name]) => (
                        <button
                            key={id}
                            onClick={() => toggleModel(id)}
                            style={{
                                padding: '6px 14px',
                                borderRadius: '20px',
                                border: `2px solid ${MODEL_COLORS[id]}`,
                                background: activeModels.has(id) ? `${MODEL_COLORS[id]}22` : 'transparent',
                                color: activeModels.has(id) ? MODEL_COLORS[id] : 'var(--text-muted)',
                                fontFamily: 'var(--font-sans)',
                                fontSize: '0.8rem',
                                fontWeight: 600,
                                cursor: 'pointer',
                                transition: 'var(--transition)',
                                opacity: activeModels.has(id) ? 1 : 0.5,
                            }}
                        >
                            {name}
                        </button>
                    ))}
                </div>

                <div className="card">
                    <div className="chart-container" style={{ height: '500px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis dataKey="epoch" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Epoch', position: 'bottom', fill: '#64748b' }} />
                                <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '12px', color: '#f1f5f9' }} />
                                <Legend wrapperStyle={{ color: '#94a3b8' }} />
                                {Object.entries(MODEL_NAMES).map(([id, name]) =>
                                    activeModels.has(id) && histories[id] ? (
                                        <Line key={id} type="monotone" dataKey={id} name={name} stroke={MODEL_COLORS[id]} strokeWidth={2} dot={false} connectNulls />
                                    ) : null
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </motion.section>

            {/* Per-model summary cards */}
            <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                <div className="section-header">
                    <h2>Training Summary</h2>
                </div>
                <div className="stats-grid">
                    {Object.entries(histories).map(([id, h]) => {
                        const epochs = h.train_loss?.length || 0
                        const bestAcc = h.val_acc ? Math.max(...h.val_acc) : 0
                        const bestF1 = h.val_f1 ? Math.max(...h.val_f1) : 0
                        return (
                            <div key={id} className="stat-card" style={{ textAlign: 'left' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                    <div style={{ fontWeight: 700 }}>{MODEL_NAMES[id] || id}</div>
                                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: MODEL_COLORS[id] }} />
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                                    <div><span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Epochs</span><div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{epochs}</div></div>
                                    <div><span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Best Val Acc</span><div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{(bestAcc * 100).toFixed(2)}%</div></div>
                                    <div><span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Best Val F1</span><div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{bestF1.toFixed(4)}</div></div>
                                    <div><span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Final Loss</span><div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{h.val_loss?.[epochs - 1]?.toFixed(4) || '-'}</div></div>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </motion.section>
        </div>
    )
}
