import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ScatterChart, Scatter, ZAxis } from 'recharts'

const API = ''
const barColors = ['#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899', '#3b82f6', '#ef4444', '#6366f1']

export default function Models() {
    const [models, setModels] = useState([])
    const [metric, setMetric] = useState('accuracy')
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch(`${API}/api/models`).then(r => r.json()).then(d => {
            setModels(d.models || [])
            setLoading(false)
        }).catch(() => setLoading(false))
    }, [])

    const metrics = [
        { key: 'accuracy', label: 'Accuracy (%)', getter: m => m.accuracy },
        { key: 'f1_weighted', label: 'F1 Weighted', getter: m => m.f1_weighted * 100 },
        { key: 'f1_macro', label: 'F1 Macro', getter: m => m.f1_macro * 100 },
        { key: 'auc_roc', label: 'AUC-ROC', getter: m => m.auc_roc * 100 },
        { key: 'precision', label: 'Precision', getter: m => m.precision * 100 },
        { key: 'recall', label: 'Recall', getter: m => m.recall * 100 },
    ]

    const currentMetric = metrics.find(m => m.key === metric)
    const chartData = models.map(m => ({
        name: m.name,
        value: currentMetric.getter(m),
        id: m.id,
    })).sort((a, b) => b.value - a.value)

    const scatterData = models.map(m => ({
        name: m.name,
        accuracy: m.accuracy,
        trainTime: m.train_time,
        f1: m.f1_weighted,
    }))

    if (loading) return <div className="loading"><div className="loading-spinner" />Loading...</div>

    return (
        <div>
            <motion.section className="section" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="section-header">
                    <h2>Model Comparison</h2>
                    <p>Compare all 8 models across different metrics</p>
                </div>

                <div className="tabs">
                    {metrics.map(m => (
                        <button key={m.key} className={`tab ${metric === m.key ? 'active' : ''}`} onClick={() => setMetric(m.key)}>
                            {m.label}
                        </button>
                    ))}
                </div>

                <div className="card">
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 40, left: 120, bottom: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" horizontal={false} />
                                <XAxis type="number" domain={metric === 'accuracy' ? [80, 100] : [0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={v => `${v.toFixed(0)}%`} />
                                <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} width={110} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '12px', color: '#f1f5f9' }} formatter={v => [`${v.toFixed(2)}%`, currentMetric.label]} />
                                <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={24}>
                                    {chartData.map((_, i) => <Cell key={i} fill={barColors[i % barColors.length]} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </motion.section>

            {/* Accuracy vs Training Time Scatter */}
            <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                <div className="section-header">
                    <h2>Accuracy vs Training Time</h2>
                    <p>Efficiency trade-off: higher accuracy vs training cost</p>
                </div>
                <div className="card">
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 40, left: 20, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis dataKey="trainTime" name="Train Time (s)" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Training Time (seconds)', position: 'bottom', fill: '#64748b', fontSize: 12 }} />
                                <YAxis dataKey="accuracy" name="Accuracy" domain={[82, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }} />
                                <ZAxis dataKey="f1" range={[100, 400]} name="F1 Score" />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '12px', color: '#f1f5f9' }} formatter={(v, name) => name === 'Train Time (s)' ? [`${v.toFixed(0)}s`, name] : [`${v.toFixed(2)}%`, name]} labelFormatter={(_, payload) => payload?.[0]?.payload?.name || ''} />
                                <Scatter data={scatterData} fill="#06b6d4" stroke="#06b6d4" strokeWidth={2} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </motion.section>

            {/* Per-Model Cards */}
            <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
                <div className="section-header">
                    <h2>Model Details</h2>
                </div>
                <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))' }}>
                    {models.map((m, i) => (
                        <motion.div key={m.id} className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                                <span className={`rank-badge ${i < 3 ? `rank-${i + 1}` : 'rank-default'}`}>{i + 1}</span>
                                <div>
                                    <div style={{ fontWeight: 700, fontSize: '1rem' }}>{m.name}</div>
                                    {m.id === 'deep_fin_dlp' && <span className="proposed-tag">Proposed</span>}
                                </div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                {[
                                    ['Accuracy', `${m.accuracy}%`],
                                    ['F1 (W)', m.f1_weighted],
                                    ['F1 (Macro)', m.f1_macro],
                                    ['AUC-ROC', m.auc_roc],
                                    ['Precision', m.precision],
                                    ['Train Time', m.train_time < 60 ? `${m.train_time}s` : `${(m.train_time / 60).toFixed(1)}m`],
                                ].map(([label, val]) => (
                                    <div key={label}>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '2px' }}>{label}</div>
                                        <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: '0.9rem' }}>{val}</div>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    ))}
                </div>
            </motion.section>
        </div>
    )
}
