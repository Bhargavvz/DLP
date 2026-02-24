import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Cpu, ChevronRight, Layers, Brain, Eye, BarChart3, Network } from 'lucide-react'

const API = ''

const ARCH_BLOCKS = [
    { name: 'Input', detail: '69 features', icon: '📥', color: '#64748b' },
    { name: 'BatchNorm', detail: 'Normalization', icon: '📐', color: '#06b6d4' },
    { name: '1D Temporal Conv', detail: '3 layers: 128→256→512', icon: '🔬', color: '#8b5cf6' },
    { name: 'SE Block', detail: 'Channel recalibration', icon: '🎯', color: '#ec4899' },
    { name: 'BiLSTM', detail: '2 layers, 256 hidden', icon: '🔄', color: '#10b981' },
    { name: 'Multi-Head Attention', detail: '8 heads, 512 dim', icon: '👁️', color: '#f59e0b' },
    { name: 'Global Avg Pool', detail: 'Sequence → vector', icon: '📊', color: '#3b82f6' },
    { name: 'Residual FC Head', detail: '512→256, GELU', icon: '🧠', color: '#8b5cf6' },
    { name: 'Softmax', detail: '7 DLP classes', icon: '🎯', color: '#ef4444' },
]

const COMPONENTS = [
    {
        title: 'Temporal Convolutional Network',
        icon: <Layers size={24} />,
        desc: 'Three 1D convolutional layers (128→256→512 channels) with GELU activation and BatchNorm extract local temporal patterns from network traffic features.',
        color: '#8b5cf6',
    },
    {
        title: 'Squeeze-and-Excitation Block',
        icon: <Eye size={24} />,
        desc: 'Adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels. Learns to amplify informative features and suppress less useful ones.',
        color: '#ec4899',
    },
    {
        title: 'Bidirectional LSTM',
        icon: <Brain size={24} />,
        desc: 'Two-layer BiLSTM with 256 hidden units captures sequential dependencies in both forward and backward directions. LayerNorm and orthogonal initialization ensure stable training.',
        color: '#10b981',
    },
    {
        title: 'Multi-Head Self-Attention',
        icon: <Network size={24} />,
        desc: '8-head self-attention mechanism captures global interactions across the entire feature sequence. Residual connections prevent degradation in deep architectures.',
        color: '#f59e0b',
    },
    {
        title: 'Residual Classification Head',
        icon: <BarChart3 size={24} />,
        desc: 'Feed-forward classifier with skip connections (512→256→7). GELU activation and dropout ensure robust generalization to unseen network traffic patterns.',
        color: '#06b6d4',
    },
]

const CLASSES = [
    { name: 'Normal Traffic', color: '#10b981', desc: 'Legitimate network operations' },
    { name: 'DoS Attack', color: '#ef4444', desc: 'Denial of Service patterns' },
    { name: 'DDoS Attack', color: '#f59e0b', desc: 'Distributed Denial of Service' },
    { name: 'Reconnaissance', color: '#8b5cf6', desc: 'Network scanning & probing' },
    { name: 'Credential Theft', color: '#ec4899', desc: 'FTP/SSH brute force attacks' },
    { name: 'Web Attack', color: '#06b6d4', desc: 'XSS, SQL injection, brute force' },
    { name: 'Botnet Exfiltration', color: '#6366f1', desc: 'Bot-driven data exfiltration' },
]

export default function Architecture() {
    const [summary, setSummary] = useState(null)

    useEffect(() => {
        fetch(`${API}/api/summary`).then(r => r.json()).then(setSummary).catch(() => { })
    }, [])

    return (
        <div>
            {/* Architecture Flow */}
            <motion.section className="section" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="section-header">
                    <h2>DeepFinDLP Architecture</h2>
                    <p>Temporal Convolutional Transformer with Squeeze-and-Excitation</p>
                </div>

                <div className="card">
                    <div className="arch-flow">
                        {ARCH_BLOCKS.map((block, i) => (
                            <motion.div key={block.name} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.1 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                    <div className="arch-block" style={{ borderColor: `${block.color}40` }}>
                                        <div style={{ fontSize: '1.5rem', marginBottom: '4px' }}>{block.icon}</div>
                                        <div className="block-name">{block.name}</div>
                                        <div className="block-detail">{block.detail}</div>
                                    </div>
                                    {i < ARCH_BLOCKS.length - 1 && <ChevronRight size={20} className="arch-arrow" />}
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </motion.section>

            {/* Component Details */}
            <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
                <div className="section-header">
                    <h2>Key Components</h2>
                    <p>Novel architecture combining CNNs, RNNs, and Attention mechanisms</p>
                </div>

                <div style={{ display: 'grid', gap: '20px' }}>
                    {COMPONENTS.map((comp, i) => (
                        <motion.div key={comp.title} className="card" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 + i * 0.1 }}
                            style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}
                        >
                            <div style={{
                                width: '50px', height: '50px', borderRadius: '12px',
                                background: `${comp.color}22`, color: comp.color,
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                flexShrink: 0,
                            }}>
                                {comp.icon}
                            </div>
                            <div>
                                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '6px' }}>{comp.title}</h3>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6 }}>{comp.desc}</p>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </motion.section>

            {/* DLP Classes */}
            <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
                <div className="section-header">
                    <h2>DLP Threat Classes</h2>
                    <p>7 categories of data leakage threats identified in financial network traffic</p>
                </div>

                <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))' }}>
                    {CLASSES.map((cls, i) => (
                        <motion.div key={cls.name} className="stat-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 + i * 0.05 }}
                            style={{ textAlign: 'left', display: 'flex', gap: '14px', alignItems: 'center' }}
                        >
                            <div style={{
                                width: '14px', height: '14px', borderRadius: '4px',
                                background: cls.color, flexShrink: 0,
                            }} />
                            <div>
                                <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{cls.name}</div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{cls.desc}</div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </motion.section>

            {/* Tech Stack */}
            <motion.section className="section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }}>
                <div className="section-header">
                    <h2>H200 GPU Optimizations</h2>
                </div>
                <div className="stats-grid">
                    {[
                        ['⚡', 'BFloat16', 'Mixed precision training for 2x speedup'],
                        ['🔥', 'torch.compile', 'Max-autotune kernel fusion'],
                        ['📦', 'Batch 4096', 'Leveraging 150GB VRAM'],
                        ['🔄', 'Cosine LR', 'Warm restarts scheduling'],
                        ['✂️', 'Gradient Clip', 'Max norm 1.0 for stability'],
                        ['🛑', 'Early Stop', 'Patience 15, best checkpoint'],
                    ].map(([icon, title, desc]) => (
                        <div key={title} className="stat-card" style={{ textAlign: 'left' }}>
                            <div style={{ fontSize: '1.5rem', marginBottom: '8px' }}>{icon}</div>
                            <div style={{ fontWeight: 700, marginBottom: '4px' }}>{title}</div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{desc}</div>
                        </div>
                    ))}
                </div>
            </motion.section>
        </div>
    )
}
