import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import { Shield } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Models from './pages/Models'
import Training from './pages/Training'
import Figures from './pages/Figures'
import Architecture from './pages/Architecture'
import Predict from './pages/Predict'

function App() {
    return (
        <Router>
            <div className="app-container">
                <nav className="navbar">
                    <div className="navbar-inner">
                        <NavLink to="/" className="navbar-logo">
                            <div className="logo-icon"><Shield size={22} /></div>
                            DeepFinDLP
                        </NavLink>
                        <ul className="navbar-links">
                            <li><NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''} end>Dashboard</NavLink></li>
                            <li><NavLink to="/models" className={({ isActive }) => isActive ? 'active' : ''}>Models</NavLink></li>
                            <li><NavLink to="/training" className={({ isActive }) => isActive ? 'active' : ''}>Training</NavLink></li>
                            <li><NavLink to="/figures" className={({ isActive }) => isActive ? 'active' : ''}>Figures</NavLink></li>
                            <li><NavLink to="/architecture" className={({ isActive }) => isActive ? 'active' : ''}>Architecture</NavLink></li>
                            <li><NavLink to="/predict" className={({ isActive }) => isActive ? 'active' : ''}>Predict</NavLink></li>
                        </ul>
                    </div>
                </nav>

                <main className="page-content">
                    <Routes>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/models" element={<Models />} />
                        <Route path="/training" element={<Training />} />
                        <Route path="/figures" element={<Figures />} />
                        <Route path="/architecture" element={<Architecture />} />
                        <Route path="/predict" element={<Predict />} />
                    </Routes>
                </main>

                <footer className="footer">
                    <p>DeepFinDLP — Mitigating Financial Instability Through Deep Learning-Driven Data Leakage Prevention</p>
                </footer>
            </div>
        </Router>
    )
}

export default App
