import { Link, useLocation } from "react-router-dom";
import {FaBrain, FaChartLine, FaCog, FaInfoCircle, FaLeaf} from "react-icons/fa";
import TypewriterText from "./typewriter.tsx";

const Navbar: React.FC = () => {
    const location = useLocation();

    const linkClass = (path: string) =>
        `flex items-center gap-1 hover:text-cyan-400 ${
            location.pathname === path ? "text-cyan-400 font-semibold" : "text-gray-300"
        }`;

    return (
        <nav className="flex items-center justify-between px-6 py-4 bg-gray-800 shadow-lg">
            <Link to="/" className="flex items-center gap-2 text-xl font-bold text-cyan-400">
                <FaLeaf className="text-6xl font-extrabold ml-20 font-mono" />
                <TypewriterText text="OPTIGREEN" style="text-6xl" />
            </Link>
            <div className="flex items-center gap-6">
                <Link to="/" className={linkClass("/")}>
                    <FaCog /> Dashboard
                </Link>
                <Link to="/about" className={linkClass("/about")}>
                    <FaInfoCircle /> About
                </Link>
                <Link to="/statistics" className={linkClass("/statistics")}>
                    <FaChartLine /> Statistics
                </Link>
            </div>
        </nav>
    );
};

export default Navbar;
