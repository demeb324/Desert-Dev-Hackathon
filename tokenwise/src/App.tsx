import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard.tsx";
import About from "./pages/About";
import OptiGreenYearlyStats from "./pages/OptiGreenYearlyStats";
import FooterCustom from "./components/FooterCustom.tsx";
import { ErrorBoundary } from "./components/ErrorBoundary";

const App: React.FC = () => {
    return (
        <ErrorBoundary>
            <Router>
                <div className="min-h-screen bg-gradient-to-br from-emerald-400 to-cyan-400 text-white">
                    <Navbar />
                    <main className="p-6">
                        <Routes>
                            <Route path="/" element={<Dashboard />} />
                            <Route path="/about" element={<About />} />
                            <Route path="/statistics" element={<OptiGreenYearlyStats />} />
                        </Routes>
                    </main>
                    <FooterCustom/>
                </div>
            </Router>
        </ErrorBoundary>
    );
};

export default App;
