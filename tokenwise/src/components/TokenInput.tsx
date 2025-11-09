import { useState } from "react";
import { FaMicrochip } from "react-icons/fa";
import type { TokenInputProps } from "../types/type";
import { useTokenContext} from "../context/TokenContext";
import { estimateTokens } from "../utils/tokenUtils";

const TokenInput: React.FC<TokenInputProps> = ({ onInputReady }) => {
    const [tokens, setTokens] = useState<number>(0);
    const [inputValue, setInputValue] = useState("");
    const [cost, setCost] = useState<number>(0);

    const {  updateTokenData} = useTokenContext();

    const handleInput = () => {
        // Estimate tokens via utility
        const tokenEstimate = estimateTokens(inputValue);
        setTokens(tokenEstimate);

        // Example cost estimation (customize per your pricing)
        const costPerToken = 0.000002; // Example: $0.002 / 1K tokens → $0.000002/token
        const totalCost = parseFloat((tokenEstimate * costPerToken).toFixed(6));
        setCost(totalCost);

        // Send prompt up to parent
        onInputReady(inputValue);

        // Update global context
        updateTokenData({ tokens: tokenEstimate, cost: totalCost, text: inputValue });
    };

    return (
        <div className="bg-gray-800 p-5 rounded-2xl shadow-xl">
            <h2 className="text-xl font-semibold mb-3 flex items-center gap-2 text-cyan-300">
                <FaMicrochip /> Prompt Input
            </h2>

            <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                className="w-full p-3 bg-gray-900 rounded-lg text-gray-200 focus:ring-2 focus:ring-cyan-500 outline-none"
                placeholder="Type or paste your prompt here..."
                rows={8}
            />

            {/* Analyze button + stats */}
            <div className="flex justify-between mt-3 text-sm text-gray-400 items-center">
        <span>
          Tokens:{" "}
            <span className="text-cyan-400 font-bold transition-all">{tokens}</span>
        </span>
                <button
                    className="bg-cyan-500 hover:bg-cyan-600 px-4 py-1 rounded-lg text-black font-semibold"
                    onClick={handleInput}
                >
                    Analyze
                </button>
            </div>

            {/* Analyzer output */}
            <div className="mt-4 bg-gray-900 p-3 rounded-lg text-gray-300">
                <div className="flex justify-between mb-2">
                    <span>Estimated Cost:</span>
                    <span className="text-cyan-400">${cost}</span>
                </div>

                {/* Visual progress bar */}
                <div className="h-2 w-full bg-gray-700 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-cyan-500 transition-all duration-500"
                        style={{
                            width: `${Math.min(tokens / 20, 100)}%`, // scale for visual
                        }}
                    ></div>
                </div>
            </div>

            {/* Model selection */}
            <div className="mt-3">
                <label className="text-gray-400">Model:</label>
                <select className="ml-2 bg-gray-700 rounded px-2 py-1 text-gray-200">
                    <option>GPT-4</option>
                    <option>GPT-3.5</option>
                    <option>Claude 3</option>
                </select>
            </div>
        </div>
    );
};

export default TokenInput;
