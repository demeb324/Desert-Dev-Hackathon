import { FaChartBar } from "react-icons/fa";

const TokenAnalyzer: React.FC = () => {
    return (
        <div className="bg-gray-800 p-5 rounded-2xl shadow-xl">
            <h2 className="text-xl font-semibold mb-3 flex items-center gap-2 text-yellow-300">
                <FaChartBar /> Token Analyzer
            </h2>
            <div className="bg-gray-900 p-4 rounded-lg h-64 flex flex-col justify-center items-center text-gray-400">
                <p className="text-sm mb-2">Visual Representation Coming Soon...</p>
                <div className="w-3/4 bg-gray-700 rounded-full h-3">
                    <div className="bg-cyan-500 h-3 rounded-full w-2/3"></div>
                </div>
            </div>
            <div className="mt-4 text-right text-lg">
                <span className="text-gray-400">Cost: </span>
                <span className="text-green-400 font-bold">$0.002</span>
            </div>
        </div>
    );
};

export default TokenAnalyzer;
