import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, ZoomIn } from 'lucide-react'

const API = ''

const FIGURE_DESCRIPTIONS = {
    '01_class_distribution': 'Distribution of 7 DLP threat categories in the CIC-IDS2017 dataset',
    '02_correlation_heatmap': 'Feature correlation matrix showing redundant/complementary features',
    '03_feature_importance': 'Top features ranked by mutual information scores',
    '04_training_loss': 'Training & validation loss curves for all DL models',
    '05_training_acc': 'Training & validation accuracy progression over epochs',
    '06_training_f1': 'F1 score evolution during model training',
    '07_roc_curves': 'ROC curves comparing discrimination power across models',
    '08_precision_recall_curves': 'Precision-Recall trade-off for each model',
    '09_confusion_matrix_proposed': 'Confusion matrix for the proposed DeepFinDLP model',
    '10_confusion_matrices_baselines': 'Confusion matrices for baseline models side-by-side',
    '11_model_comparison': 'Bar chart comparing all models by accuracy, F1, and AUC-ROC',
    '14_per_class_f1': 'Per-class F1 scores across all models',
    '15_training_time': 'Training time comparison across all models',
    '16_architecture_diagram': 'Architecture diagram of the proposed DeepFinDLP model',
}

export default function Figures() {
    const [figures, setFigures] = useState([])
    const [lightbox, setLightbox] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch(`${API}/api/figures`).then(r => r.json()).then(d => {
            setFigures(d.figures || [])
            setLoading(false)
        }).catch(() => setLoading(false))
    }, [])

    if (loading) return <div className="loading"><div className="loading-spinner" />Loading figures...</div>

    return (
        <div>
            <motion.section className="section" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="section-header">
                    <h2>Research Figures</h2>
                    <p>{figures.length} publication-quality figures at 300 DPI — click to expand</p>
                </div>

                <div className="figures-grid">
                    {figures.map((fig, i) => (
                        <motion.div
                            key={fig.filename}
                            className="figure-card"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.05 }}
                            onClick={() => setLightbox(fig)}
                        >
                            <div style={{ position: 'relative' }}>
                                <img src={fig.url} alt={fig.name} loading="lazy" />
                                <div style={{
                                    position: 'absolute', top: '8px', right: '8px',
                                    background: 'rgba(0,0,0,0.6)', borderRadius: '8px', padding: '6px',
                                    display: 'flex', alignItems: 'center',
                                }}>
                                    <ZoomIn size={16} color="#fff" />
                                </div>
                            </div>
                            <div className="figure-info">
                                <h3>{fig.name.replace(/_/g, ' ').replace(/^\d+\s*/, '')}</h3>
                                <p>{FIGURE_DESCRIPTIONS[fig.name] || `${fig.size_kb} KB`}</p>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </motion.section>

            {/* Lightbox */}
            <AnimatePresence>
                {lightbox && (
                    <motion.div
                        className="lightbox-overlay"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setLightbox(null)}
                    >
                        <motion.div initial={{ scale: 0.8 }} animate={{ scale: 1 }} exit={{ scale: 0.8 }} onClick={(e) => e.stopPropagation()}>
                            <div style={{ position: 'relative' }}>
                                <img src={lightbox.url} alt={lightbox.name} />
                                <button
                                    onClick={() => setLightbox(null)}
                                    style={{
                                        position: 'absolute', top: '-12px', right: '-12px',
                                        width: '36px', height: '36px', borderRadius: '50%',
                                        background: '#ef4444', border: 'none', cursor: 'pointer',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    }}
                                >
                                    <X size={18} color="#fff" />
                                </button>
                                <div style={{
                                    textAlign: 'center', padding: '16px',
                                    color: '#f1f5f9', fontSize: '1rem', fontWeight: 600,
                                }}>
                                    {lightbox.name.replace(/_/g, ' ').replace(/^\d+\s*/, '')}
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
