import { Link, useLocation } from "react-router-dom";
import { FaBrain, FaCog, FaInfoCircle } from "react-icons/fa";

const Navbar: React.FC = () => {
    const location = useLocation();

    const linkClass = (path: string) =>
        `flex items-center gap-1 hover:text-cyan-400 ${
            location.pathname === path ? "text-cyan-400 font-semibold" : "text-gray-300"
        }`;

    return (
        <nav className="flex items-center justify-between px-6 py-4 bg-gray-800 shadow-lg">
            <Link to="/" className="flex items-center gap-2 text-xl font-bold text-cyan-400">
                <FaBrain className="text-5xl font-extrabold" /> OPTIGREEN
            </Link>
            <div className="flex items-center gap-6">
                <Link to="/" className={linkClass("/")}>
                    <FaCog /> Dashboard
                </Link>
                <Link to="/about" className={linkClass("/about")}>
                    <FaInfoCircle /> About
                </Link>
            </div>
        </nav>
    );
};

export default Navbar;
