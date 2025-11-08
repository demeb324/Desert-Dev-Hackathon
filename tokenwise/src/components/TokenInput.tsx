import { useState } from "react";
import { FaMicrochip } from "react-icons/fa";

import type {TokenInputProps} from "../types/type.tsx";

const TokenInput: React.FC<TokenInputProps> = ({ onInputReady }) => {

    const [tokens, setTokens] = useState<number>(0);
    const [inputValue, setInputValue] = useState('');

    const handleInput = () => {
      //  const text = e.target.value;
      //  setPrompt(text);
        // Estimate tokens (~1.3 tokens per word)
        setTokens(Math.ceil(inputValue.split(/\s+/).length * 1.3));
        onInputReady(inputValue)
         setInputValue(inputValue)
        onInputReady(inputValue)
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

            <div className="flex justify-between mt-3 text-sm text-gray-400">
        <span>
          Tokens: <span className="text-cyan-400 font-bold">{tokens}</span>
        </span>
                <button className="bg-cyan-500 hover:bg-cyan-600 px-4 py-1 rounded-lg text-black font-semibold" onClick={handleInput}>
                    Analyze
                </button>
            </div>

            <div className="mt-3">
                <label className="text-gray-400">Model:</label>
                <select className="ml-2 bg-gray-700 rounded px-2 py-1">
                    <option>GPT-4</option>
                    <option>GPT-3.5</option>
                    <option>Claude 3</option>
                </select>
            </div>
        </div>
    );
};

export default TokenInput;
