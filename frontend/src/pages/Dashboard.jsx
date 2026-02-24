import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend } from 'recharts'
import { Shield, Database, Cpu, TrendingUp, Award, Zap, Activity, ChevronRight } from 'lucide-react'
import { Link } from 'react-router-dom'

const API = ''

function AnimatedCounter({ value, suffix = '', decimals = 0 }) {
    const [count, setCount] = useState(0)
    useEffect(() => {
        let start = 0
        const end = parseFloat(value)
        const duration = 1500
        const step = (end - start) / (duration / 16)
        const timer = setInterval(() => {
            start += step
            if (start >= end) {
                setCount(end)
                clearInterval(timer)
            } else {
                setCount(start)
            }
        }, 16)
        return () => clearInterval(timer)
    }, [value])
    return <span>{count.toFixed(decimals)}{suffix}</span>
}

export default function Dashboard() {
    const [models, setModels] = useState([])
    const [summary, setSummary] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        Promise.all([
            fetch(`${API}/api/models`).then(r => r.json()),
            fetch(`${API}/api/summary`).then(r => r.json()),
        ]).then(([modelsData, summaryData]) => {
            setModels(modelsData.models || [])
            setSummary(summaryData)
            setLoading(false)
        }).catch(() => setLoading(false))
    }, [])

    const barColors = ['#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899', '#3b82f6', '#ef4444', '#6366f1']

    const radarData = models.slice(0, 6).map(m => ({
        name: m.name.length > 12 ? m.id.toUpperCase() : m.name,
        Accuracy: m.accuracy,
        'F1': m.f1_weighted * 100,
        'AUC-ROC': m.auc_roc * 100,
        Precision: m.precision * 100,
        Recall: m.recall * 100,
    }))

    const containerVariants = {
        hidden: { opacity: 0 },
        show: { opacity: 1, transition: { staggerChildren: 0.1 } }
    }
    const itemVariants = {
        hidden: { opacity: 0, y: 20 },
        show: { opacity: 1, y: 0 }
    }

    if (loading) {
        return <div className="loading"><div className="loading-spinner" />Loading results...</div>
    }

    return (
        <div>
            {/* Hero */}
            <motion.section className="hero" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
                <div className="hero-badge"><Shield size={14} /> IEEE Research Project</div>
                <h1>
                    <span className="gradient-text">DeepFinDLP</span>
                    <br />Data Leakage Prevention
                </h1>
                <p>
                    Mitigating financial instability through deep learning-driven network traffic analysis.
                    {summary && ` Trained ${summary.total_models} models on ${summary.total_samples?.toLocaleString()} samples.`}
                </p>
            </motion.section>

            {/* Stats */}
            <motion.div className="stats-grid" variants={containerVariants} initial="hidden" animate="show">
                <motion.div className="stat-card" variants={itemVariants}>
                    <div className="stat-icon">🎯</div>
                    <div className="stat-value"><AnimatedCounter value={summary?.best_model?.accuracy || 99.83} decimals={2} suffix="%" /></div>
                    <div className="stat-label">Best Accuracy ({summary?.best_model?.name || 'XGBoost'})</div>
                </motion.div>
                <motion.div className="stat-card" variants={itemVariants}>
                    <div className="stat-icon">🧠</div>
                    <div className="stat-value"><AnimatedCounter value={summary?.total_models || 8} decimals={0} /></div>
                    <div className="stat-label">Models Trained</div>
                </motion.div>
                <motion.div className="stat-card" variants={itemVariants}>
                    <div className="stat-icon">📊</div>
                    <div className="stat-value"><AnimatedCounter value={2.83} decimals={2} suffix="M" /></div>
                    <div className="stat-label">Network Flow Records</div>
                </motion.div>
                <motion.div className="stat-card" variants={itemVariants}>
                    <div className="stat-icon">🔬</div>
                    <div className="stat-value"><AnimatedCounter value={summary?.num_classes || 7} decimals={0} /></div>
                    <div className="stat-label">DLP Threat Classes</div>
                </motion.div>
                <motion.div className="stat-card" variants={itemVariants}>
                    <div className="stat-icon">⚡</div>
                    <div className="stat-value">H200</div>
                    <div className="stat-label">NVIDIA GPU (150GB)</div>
                </motion.div>
            </motion.div>

            {/* Model Accuracy Comparison */}
            <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
                <div className="section-header">
                    <h2>Model Performance Comparison</h2>
                    <p>Accuracy across all trained models on CIC-IDS2017 test set</p>
                </div>
                <div className="card">
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={models} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-30} textAnchor="end" />
                                <YAxis domain={[80, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={v => `${v}%`} />
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '12px', color: '#f1f5f9' }}
                                    formatter={(val) => [`${val}%`, 'Accuracy']}
                                />
                                <Bar dataKey="accuracy" radius={[6, 6, 0, 0]}>
                                    {models.map((_, i) => <Cell key={i} fill={barColors[i % barColors.length]} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </motion.section>

            {/* Leaderboard Table */}
            <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
                <div className="section-header">
                    <h2>Model Leaderboard</h2>
                    <p>Ranked by test accuracy with comprehensive metrics</p>
                </div>
                <div className="card" style={{ overflowX: 'auto' }}>
                    <table className="model-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Model</th>
                                <th>Accuracy</th>
                                <th>F1 (Weighted)</th>
                                <th>F1 (Macro)</th>
                                <th>AUC-ROC</th>
                                <th>Train Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {models.map((m, i) => (
                                <tr key={m.id}>
                                    <td>
                                        <span className={`rank-badge ${i < 3 ? `rank-${i + 1}` : 'rank-default'}`}>
                                            {i + 1}
                                        </span>
                                    </td>
                                    <td>
                                        {m.name}
                                        {m.id === 'deep_fin_dlp' && <span className="proposed-tag">Proposed</span>}
                                    </td>
                                    <td>
                                        <div className="accuracy-bar-container">
                                            <div className="accuracy-bar">
                                                <div className="accuracy-bar-fill" style={{ width: `${(m.accuracy / 100) * 100}%` }} />
                                            </div>
                                            <span className="accuracy-value">{m.accuracy}%</span>
                                        </div>
                                    </td>
                                    <td>{m.f1_weighted}</td>
                                    <td>{m.f1_macro}</td>
                                    <td>{m.auc_roc}</td>
                                    <td>{m.train_time < 60 ? `${m.train_time}s` : `${(m.train_time / 60).toFixed(1)}m`}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </motion.section>

            {/* Radar chart */}
            {radarData.length > 0 && (
                <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
                    <div className="section-header">
                        <h2>Multi-Metric Radar</h2>
                        <p>Comparing models across multiple performance dimensions</p>
                    </div>
                    <div className="card">
                        <div className="chart-container">
                            <ResponsiveContainer width="100%" height="100%">
                                <RadarChart data={[
                                    { metric: 'Accuracy', ...Object.fromEntries(radarData.map(r => [r.name, r.Accuracy])) },
                                    { metric: 'F1 Score', ...Object.fromEntries(radarData.map(r => [r.name, r.F1])) },
                                    { metric: 'AUC-ROC', ...Object.fromEntries(radarData.map(r => [r.name, r['AUC-ROC']])) },
                                    { metric: 'Precision', ...Object.fromEntries(radarData.map(r => [r.name, r.Precision])) },
                                    { metric: 'Recall', ...Object.fromEntries(radarData.map(r => [r.name, r.Recall])) },
                                ]}>
                                    <PolarGrid stroke="rgba(148,163,184,0.15)" />
                                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                    <PolarRadiusAxis domain={[80, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                                    {radarData.slice(0, 4).map((r, i) => (
                                        <Radar key={r.name} name={r.name} dataKey={r.name} stroke={barColors[i]} fill={barColors[i]} fillOpacity={0.1} strokeWidth={2} />
                                    ))}
                                    <Legend wrapperStyle={{ color: '#94a3b8', fontSize: '12px' }} />
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </motion.section>
            )}

            {/* Quick Links */}
            <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}>
                <div className="stats-grid">
                    {[
                        { to: '/training', icon: <Activity size={24} />, title: 'Training Curves', desc: 'Loss, accuracy, F1 over epochs' },
                        { to: '/figures', icon: <TrendingUp size={24} />, title: 'Figures Gallery', desc: '14 publication-quality plots' },
                        { to: '/architecture', icon: <Cpu size={24} />, title: 'Architecture', desc: 'DeepFinDLP model design' },
                    ].map(link => (
                        <Link to={link.to} key={link.to} style={{ textDecoration: 'none' }}>
                            <div className="stat-card" style={{ textAlign: 'left', display: 'flex', alignItems: 'center', gap: '16px' }}>
                                <div style={{ color: 'var(--accent-primary)' }}>{link.icon}</div>
                                <div>
                                    <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text-primary)' }}>{link.title}</div>
                                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{link.desc}</div>
                                </div>
                                <ChevronRight size={18} style={{ marginLeft: 'auto', color: 'var(--text-muted)' }} />
                            </div>
                        </Link>
                    ))}
                </div>
            </motion.section>
        </div>
    )
}
