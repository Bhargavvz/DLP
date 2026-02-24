import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Play, AlertTriangle, Shield, Zap, ChevronDown } from 'lucide-react'

const API = ''

const THREAT_COLORS = {
    'Normal Traffic': '#10b981',
    'DoS Attack': '#ef4444',
    'DDoS Attack': '#f59e0b',
    'Reconnaissance': '#8b5cf6',
    'Credential Theft': '#ec4899',
    'Web Attack': '#06b6d4',
    'Botnet Exfiltration': '#6366f1',
}

const THREAT_ICONS = {
    'Normal Traffic': '✅',
    'DoS Attack': '🔴',
    'DDoS Attack': '🟡',
    'Reconnaissance': '🟣',
    'Credential Theft': '🩷',
    'Web Attack': '🔵',
    'Botnet Exfiltration': '🟤',
}

export default function Predict() {
    const [samples, setSamples] = useState([])
    const [featureNames, setFeatureNames] = useState([])
    const [availableModels, setAvailableModels] = useState([])
    const [selectedModel, setSelectedModel] = useState('deep_fin_dlp')
    const [selectedSample, setSelectedSample] = useState(null)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [loadingSamples, setLoadingSamples] = useState(true)

    useEffect(() => {
        Promise.all([
            fetch(`${API}/api/predict/info`).then(r => r.json()),
            fetch(`${API}/api/predict/sample`).then(r => r.json()),
        ]).then(([info, sampleData]) => {
            setFeatureNames(info.feature_names || [])
            setAvailableModels(info.available_models || [])
            if (info.available_models?.length > 0) setSelectedModel(info.available_models[0])
            setSamples(sampleData.samples || [])
            setLoadingSamples(false)
        }).catch(() => setLoadingSamples(false))
    }, [])

    const runPrediction = async (sample) => {
        setSelectedSample(sample)
        setResult(null)
        setLoading(true)
        try {
            const res = await fetch(`${API}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: sample.features, model_name: selectedModel }),
            })
            const data = await res.json()
            if (res.ok) {
                setResult(data)
            } else {
                setResult({ error: data.detail || 'Prediction failed' })
            }
        } catch (err) {
            setResult({ error: 'Prediction failed: ' + err.message })
        }
        setLoading(false)
    }

    const runAllModels = async (sample) => {
        setSelectedSample(sample)
        setResult(null)
        setLoading(true)
        try {
            const results = await Promise.all(
                availableModels.map(m =>
                    fetch(`${API}/api/predict`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ features: sample.features, model_name: m }),
                    }).then(r => r.json())
                )
            )
            setResult({ multi: results.map((r, i) => ({ ...r, model: availableModels[i] })) })
        } catch (err) {
            setResult({ error: err.message })
        }
        setLoading(false)
    }

    const probsChart = result?.probabilities
        ? Object.entries(result.probabilities)
            .map(([name, prob]) => ({ name, probability: parseFloat((prob * 100).toFixed(2)) }))
            .sort((a, b) => b.probability - a.probability)
        : []

    if (loadingSamples) return <div className="loading"><div className="loading-spinner" />Loading prediction engine...</div>

    return (
        <div>
            <motion.section className="section" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="section-header">
                    <h2>Live Threat Prediction</h2>
                    <p>Select a real network traffic sample and run inference through trained models</p>
                </div>

                {/* Model Selector */}
                <div className="card" style={{ marginBottom: '24px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                        <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Select Model:</div>
                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                            {availableModels.map(m => (
                                <button key={m} onClick={() => setSelectedModel(m)}
                                    style={{
                                        padding: '8px 16px', borderRadius: '8px', border: '1px solid',
                                        borderColor: selectedModel === m ? 'var(--accent-primary)' : 'var(--border-color)',
                                        background: selectedModel === m ? 'rgba(6,182,212,0.15)' : 'transparent',
                                        color: selectedModel === m ? 'var(--accent-primary)' : 'var(--text-secondary)',
                                        fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 600,
                                        cursor: 'pointer', transition: 'var(--transition)',
                                    }}
                                >
                                    {m}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Sample Selector */}
                <div className="section-header">
                    <h2 style={{ fontSize: '1.3rem' }}>Sample Network Traffic</h2>
                    <p>Click a sample to classify it, or "Run All Models" to compare</p>
                </div>

                <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))' }}>
                    {samples.map((s, i) => (
                        <motion.div key={i} className="stat-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
                            style={{ textAlign: 'left', cursor: 'pointer', border: selectedSample === s ? '1px solid var(--accent-primary)' : undefined }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                <div>
                                    <span style={{ fontSize: '1.2rem', marginRight: '8px' }}>{THREAT_ICONS[s.label] || '❓'}</span>
                                    <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>{s.label}</span>
                                </div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{s.original_label}</div>
                            </div>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button onClick={() => runPrediction(s)}
                                    style={{
                                        flex: 1, padding: '8px', borderRadius: '8px',
                                        background: 'var(--gradient-accent)', border: 'none',
                                        color: '#fff', fontWeight: 600, fontSize: '0.8rem',
                                        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                                    }}
                                >
                                    <Play size={14} /> Predict
                                </button>
                                <button onClick={() => runAllModels(s)}
                                    style={{
                                        flex: 1, padding: '8px', borderRadius: '8px',
                                        background: 'rgba(139,92,246,0.2)', border: '1px solid rgba(139,92,246,0.3)',
                                        color: 'var(--accent-secondary)', fontWeight: 600, fontSize: '0.8rem',
                                        cursor: 'pointer',
                                    }}
                                >
                                    All Models
                                </button>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </motion.section>

            {/* Results */}
            <AnimatePresence>
                {loading && (
                    <motion.div className="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                        <div className="loading-spinner" /> Running inference...
                    </motion.div>
                )}

                {result && !result.error && !result.multi && (
                    <motion.section className="section" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                        <div className="card" style={{ borderLeft: `4px solid ${THREAT_COLORS[result.prediction] || '#64748b'}` }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px' }}>
                                <div style={{
                                    width: '64px', height: '64px', borderRadius: '16px',
                                    background: `${THREAT_COLORS[result.prediction]}22`,
                                    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '2rem',
                                }}>
                                    {result.prediction === 'Normal Traffic' ? <Shield size={32} color="#10b981" /> : <AlertTriangle size={32} color={THREAT_COLORS[result.prediction]} />}
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                                        Prediction ({result.model})
                                        {result.demo_mode && <span style={{ marginLeft: '8px', padding: '2px 8px', borderRadius: '8px', background: 'rgba(245,158,11,0.2)', color: '#f59e0b', fontSize: '0.7rem', fontWeight: 700 }}>DEMO MODE</span>}
                                    </div>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: THREAT_COLORS[result.prediction] }}>{result.prediction}</div>
                                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                                        Confidence: {(result.confidence * 100).toFixed(2)}%
                                    </div>
                                </div>
                                {selectedSample && (
                                    <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Ground Truth</div>
                                        <div style={{ fontWeight: 700, color: THREAT_COLORS[selectedSample.label] }}>{selectedSample.label}</div>
                                        <div style={{
                                            fontSize: '0.8rem', fontWeight: 600,
                                            color: result.prediction === selectedSample.label ? '#10b981' : '#ef4444',
                                        }}>
                                            {result.prediction === selectedSample.label ? '✓ Correct' : '✗ Incorrect'}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Probability bar chart */}
                            <div style={{ height: '280px' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={probsChart} layout="vertical" margin={{ left: 120, right: 30 }}>
                                        <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={v => `${v}%`} />
                                        <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} width={110} />
                                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '12px', color: '#f1f5f9' }} formatter={v => [`${v}%`, 'Probability']} />
                                        <Bar dataKey="probability" radius={[0, 6, 6, 0]} barSize={20}>
                                            {probsChart.map((p) => <Cell key={p.name} fill={THREAT_COLORS[p.name] || '#64748b'} />)}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </motion.section>
                )}

                {/* Multi-model results */}
                {result?.multi && (
                    <motion.section className="section" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                        <div className="section-header">
                            <h2 style={{ fontSize: '1.3rem' }}>Multi-Model Comparison</h2>
                            <p>Ground truth: <strong style={{ color: THREAT_COLORS[selectedSample?.label] }}>{selectedSample?.label}</strong></p>
                        </div>
                        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))' }}>
                            {result.multi.map((r, i) => (
                                <motion.div key={i} className="card" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.05 }}
                                    style={{ borderLeft: `3px solid ${r.prediction === selectedSample?.label ? '#10b981' : '#ef4444'}` }}
                                >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                                        <div style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>{r.model}</div>
                                        <div style={{
                                            padding: '2px 8px', borderRadius: '12px', fontSize: '0.7rem', fontWeight: 700,
                                            background: r.prediction === selectedSample?.label ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)',
                                            color: r.prediction === selectedSample?.label ? '#10b981' : '#ef4444',
                                        }}>
                                            {r.prediction === selectedSample?.label ? '✓ CORRECT' : '✗ WRONG'}
                                        </div>
                                    </div>
                                    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: THREAT_COLORS[r.prediction], marginBottom: '4px' }}>{r.prediction}</div>
                                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                        {(r.confidence * 100).toFixed(2)}% confidence
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.section>
                )}

                {result?.error && (
                    <motion.div className="card" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ borderLeft: '4px solid #ef4444', marginTop: '24px' }}>
                        <div style={{ color: '#ef4444', fontWeight: 600 }}>⚠️ {result.error}</div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
