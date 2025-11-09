import { useState, useEffect, useCallback } from "react";
import { FaMicrochip } from "react-icons/fa";
import type { TokenInputProps } from "../types/type";
import { useTokenContext } from "../context/TokenContext";
import { lmStudioService } from "@/services/lmStudioService";
import { LoadingSpinner } from "./LoadingSpinner";

// Debounce delay in milliseconds
const DEBOUNCE_DELAY = 500;

const TokenInput: React.FC<TokenInputProps> = ({ onInputReady }) => {
    const [inputValue, setInputValue] = useState("");
    const [debouncedValue, setDebouncedValue] = useState("");

    const { tokens, setTokens, cost, setCost, setLoading, setError } = useTokenContext();
    const [isCountingTokens, setIsCountingTokens] = useState(false);

    // Debounce input value
    useEffect(() => {
        const timer = setTimeout(() => {
            setDebouncedValue(inputValue);
        }, DEBOUNCE_DELAY);

        return () => clearTimeout(timer);
    }, [inputValue]);

    // Calculate tokens and cost when debounced value changes
    useEffect(() => {
        if (!debouncedValue || debouncedValue.trim() === "") {
            setTokens(0);
            setCost(0);
            return;
        }

        const updateTokenCount = async () => {
            setIsCountingTokens(true);
            setError(null);

            try {
                const tokenCount = await lmStudioService.getTokenCount(debouncedValue);
                setTokens(tokenCount);

                // Calculate cost (example pricing: $0.000002/token)
                const costPerToken = 0.000002;
                const totalCost = parseFloat((tokenCount * costPerToken).toFixed(6));
                setCost(totalCost);
            } catch (err: any) {
                console.error("Failed to count tokens:", err);
                // Don't show error to user for token counting - use fallback
                const words = debouncedValue.match(/\b\w+\b/g);
                const fallbackTokens = words ? Math.ceil(words.length * 1.3) : 0;
                setTokens(fallbackTokens);

                const costPerToken = 0.000002;
                const totalCost = parseFloat((fallbackTokens * costPerToken).toFixed(6));
                setCost(totalCost);
            } finally {
                setIsCountingTokens(false);
            }
        };

        updateTokenCount();
    }, [debouncedValue, setTokens, setCost, setError]);

    const handleAnalyze = useCallback(() => {
        // Send prompt up to parent
        onInputReady(inputValue);
    }, [inputValue, onInputReady]);

    const maxTokens = 2048; // Visual max for progress bar
    const progressPercentage = Math.min((tokens / maxTokens) * 100, 100);

    return (
        <div className="bg-gray-800 p-5 rounded-2xl shadow-xl">
            <h2 className="text-xl font-semibold mb-3 flex items-center gap-2 text-cyan-300">
                <FaMicrochip /> Prompt Input
            </h2>

            <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                className="w-full p-3 bg-gray-900 rounded-lg text-gray-200 focus:ring-2 focus:ring-cyan-500 outline-none resize-none"
                placeholder="Type or paste your prompt here..."
                rows={8}
            />

            {/* Analyze button + stats */}
            <div className="flex justify-between mt-3 text-sm text-gray-400 items-center">
                <span className="flex items-center gap-2">
                    Tokens:{" "}
                    {isCountingTokens ? (
                        <LoadingSpinner size="sm" />
                    ) : (
                        <span className="text-cyan-400 font-bold transition-all">{tokens}</span>
                    )}
                </span>
                <button
                    className="bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-1 rounded-lg text-black font-semibold transition-colors"
                    onClick={handleAnalyze}
                    disabled={!inputValue || inputValue.trim() === ""}
                >
                    Analyze
                </button>
            </div>

            {/* Analyzer output */}
            <div className="mt-4 bg-gray-900 p-3 rounded-lg text-gray-300">
                <div className="flex justify-between mb-2">
                    <span>Estimated Cost:</span>
                    <span className="text-cyan-400">${cost.toFixed(6)}</span>
                </div>

                {/* Visual progress bar */}
                <div className="h-2 w-full bg-gray-700 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 transition-all duration-500"
                        style={{
                            width: `${progressPercentage}%`,
                        }}
                    ></div>
                </div>
                <div className="text-xs text-gray-500 mt-1 text-right">
                    {tokens} / {maxTokens} tokens
                </div>
            </div>

            {/* LM Studio Badge */}
            <div className="mt-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-xs text-gray-400">LM Studio</span>
                </div>
                {isCountingTokens && (
                    <span className="text-xs text-gray-500 animate-pulse">
                        Counting tokens...
                    </span>
                )}
            </div>
        </div>
    );
};

export default TokenInput;
