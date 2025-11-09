import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard.tsx";
import About from "./pages/About";

const App: React.FC = () => {
    return (
        <Router>
            <div className="min-h-screen bg-gradient-to-br from-green-300 via-green-600 to-green-950 text-white">
                <Navbar />
                <main className="p-6">
                    <Routes>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/about" element={<About />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
};

export default App;
