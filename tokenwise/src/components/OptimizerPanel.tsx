import { FaMagic, FaCopy, FaCheck, FaPlay } from "react-icons/fa";
import type { TokenOptimizerProps } from "../types/type.tsx";
import { useState, useEffect, forwardRef, useImperativeHandle } from "react";
import { lmStudioService } from "@/services/lmStudioService";
import { LoadingSpinner } from "./LoadingSpinner";
import { ErrorAlert } from "./ErrorAlert";

export interface OptimizerPanelHandle {
    autoOptimizeAndExecute: () => Promise<void>;
}

const OptimizerPanel = forwardRef<OptimizerPanelHandle, TokenOptimizerProps>(({ tokenInput }, ref) => {
    const [optimizedResult, setOptimizedResult] = useState<string>("");
    const [beforeTokens, setBeforeTokens] = useState<number>(0);
    const [afterTokens, setAfterTokens] = useState<number>(0);
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isCopied, setIsCopied] = useState(false);
    const [isCountingTokens, setIsCountingTokens] = useState(false);

    // State for execution feature
    const [llmResponse, setLlmResponse] = useState<string>("");
    const [isExecuting, setIsExecuting] = useState(false);
    const [responseTokens, setResponseTokens] = useState<number>(0);
    const [executeError, setExecuteError] = useState<string | null>(null);
    const [isResponseCopied, setIsResponseCopied] = useState(false);

    // Workflow step indicator
    const [workflowStep, setWorkflowStep] = useState<'idle' | 'optimizing' | 'executing' | 'complete'>('idle');

    // Expose autoOptimizeAndExecute method to parent via ref
    useImperativeHandle(ref, () => ({
        autoOptimizeAndExecute: async () => {
            if (!tokenInput || tokenInput.trim() === "") {
                setError("No prompt to analyze");
                return;
            }

            try {
                // Step 1: Optimize
                setWorkflowStep('optimizing');
                setIsOptimizing(true);
                setError(null);
                setExecuteError(null);
                setLlmResponse("");
                setResponseTokens(0);

                const optimized = await lmStudioService.optimizePrompt(tokenInput);
                setOptimizedResult(optimized);

                const count = await lmStudioService.getTokenCount(optimized);
                setAfterTokens(count);
                setIsOptimizing(false);

                // Step 2: Execute
                setWorkflowStep('executing');
                setIsExecuting(true);

                const response = await lmStudioService.executePrompt(optimized);
                setLlmResponse(response);

                const respCount = await lmStudioService.getTokenCount(response);
                setResponseTokens(respCount);
                setIsExecuting(false);

                setWorkflowStep('complete');

                // Reset to idle after a moment
                setTimeout(() => setWorkflowStep('idle'), 2000);
            } catch (err: any) {
                console.error("Auto-analysis failed:", err);
                setIsOptimizing(false);
                setIsExecuting(false);
                setWorkflowStep('idle');
                setError(err.message || "Auto-analysis failed. Please try again.");
            }
        }
    }));

    // Update before token count when input changes
    useEffect(() => {
        if (!tokenInput || tokenInput.trim() === "") {
            setBeforeTokens(0);
            return;
        }

        const updateTokenCount = async () => {
            setIsCountingTokens(true);
            try {
                const count = await lmStudioService.getTokenCount(tokenInput);
                setBeforeTokens(count);
            } catch (err) {
                console.error("Failed to count tokens:", err);
                // Silently fail for token counting - not critical
            } finally {
                setIsCountingTokens(false);
            }
        };

        updateTokenCount();
    }, [tokenInput]);

    // Handle optimization
    const handleOptimize = async () => {
        if (!tokenInput || tokenInput.trim() === "") {
            setError("Please enter a prompt to optimize");
            return;
        }

        setIsOptimizing(true);
        setError(null);
        // Clear previous execution results when re-optimizing
        setLlmResponse("");
        setResponseTokens(0);
        setExecuteError(null);

        try {
            // Optimize the prompt
            const optimized = await lmStudioService.optimizePrompt(tokenInput);
            setOptimizedResult(optimized);

            // Get token count for optimized version
            const count = await lmStudioService.getTokenCount(optimized);
            setAfterTokens(count);
        } catch (err: any) {
            console.error("Optimization failed:", err);
            setError(err.message || "Failed to optimize prompt. Please try again.");
            setOptimizedResult("");
            setAfterTokens(0);
        } finally {
            setIsOptimizing(false);
        }
    };

    // Handle executing the optimized prompt
    const handleExecute = async () => {
        if (!optimizedResult || optimizedResult.trim() === "") {
            setExecuteError("Please optimize a prompt first");
            return;
        }

        setIsExecuting(true);
        setExecuteError(null);

        try {
            // Execute the optimized prompt
            const response = await lmStudioService.executePrompt(optimizedResult);
            setLlmResponse(response);

            // Get token count for the response
            const count = await lmStudioService.getTokenCount(response);
            setResponseTokens(count);
        } catch (err: any) {
            console.error("Execution failed:", err);
            setExecuteError(err.message || "Failed to execute prompt. Please try again.");
            setLlmResponse("");
            setResponseTokens(0);
        } finally {
            setIsExecuting(false);
        }
    };

    // Handle copy to clipboard
    const handleCopy = async () => {
        if (!optimizedResult) return;

        try {
            await navigator.clipboard.writeText(optimizedResult);
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 2000);
        } catch (err) {
            console.error("Failed to copy text:", err);
            alert("Failed to copy prompt. Please try again or copy manually.");
        }
    };

    // Handle copy response to clipboard
    const handleCopyResponse = async () => {
        if (!llmResponse) return;

        try {
            await navigator.clipboard.writeText(llmResponse);
            setIsResponseCopied(true);
            setTimeout(() => setIsResponseCopied(false), 2000);
        } catch (err) {
            console.error("Failed to copy response:", err);
            alert("Failed to copy response. Please try again or copy manually.");
        }
    };

    const tokenReduction = beforeTokens > 0 ? Math.round(((beforeTokens - afterTokens) / beforeTokens) * 100) : 0;

    return (
        <div className="bg-gray-800 p-5 rounded-2xl shadow-xl">
            <h2 className="text-xl font-semibold mb-3 flex items-center gap-2 text-pink-300">
                <FaMagic /> Optimizer
            </h2>

            {/* Error Alert */}
            {error && (
                <ErrorAlert
                    error={error}
                    onRetry={handleOptimize}
                    onDismiss={() => setError(null)}
                    className="mb-4"
                />
            )}

            {/* Token Count Table */}
            <table className="w-full text-sm text-gray-300 mb-3">
                <thead>
                    <tr className="border-b border-gray-600">
                        <th className="text-left py-1">Section</th>
                        <th>Before</th>
                        <th>After</th>
                        <th>Saved</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Tokens</td>
                        <td className="text-center">
                            {isCountingTokens ? (
                                <LoadingSpinner size="sm" />
                            ) : (
                                beforeTokens
                            )}
                        </td>
                        <td className="text-green-400 text-center">{afterTokens}</td>
                        <td className="text-cyan-400 text-center">
                            {afterTokens > 0 && tokenReduction >= 0 ? `${tokenReduction}%` : "-"}
                        </td>
                    </tr>
                </tbody>
            </table>

            {/* Workflow Step Indicator */}
            {workflowStep !== 'idle' && (
                <div className="mb-2 text-sm text-cyan-400 animate-pulse flex items-center gap-2">
                    {workflowStep === 'optimizing' && '🔄 Step 1/2: Optimizing prompt...'}
                    {workflowStep === 'executing' && '🔄 Step 2/2: Running optimized prompt...'}
                    {workflowStep === 'complete' && '✅ Analysis complete!'}
                </div>
            )}

            {/* Optimized Result Display */}
            <div className="bg-gray-900 rounded-lg p-3 text-gray-400 text-sm min-h-[100px] flex items-center justify-center relative">
                {isOptimizing ? (
                    <LoadingSpinner text="Optimizing prompt..." />
                ) : optimizedResult ? (
                    <strong className="text-gray-200">{optimizedResult}</strong>
                ) : (
                    <span className="text-gray-500 italic">
                        Click "Analyze & Run" to optimize your prompt
                    </span>
                )}
            </div>

            {/* Copy Optimized Prompt Button (subtle) */}
            {optimizedResult && (
                <div className="mt-2 flex justify-end">
                    <button
                        className="bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs px-3 py-1 rounded flex items-center gap-1 transition-colors"
                        onClick={handleCopy}
                        disabled={isOptimizing}
                    >
                        {isCopied ? <FaCheck /> : <FaCopy />}
                        {isCopied ? "Copied!" : "Copy"}
                    </button>
                </div>
            )}

            {/* Execute Error Alert */}
            {executeError && (
                <div className="mt-3">
                    <ErrorAlert
                        error={executeError}
                        onRetry={handleExecute}
                        onDismiss={() => setExecuteError(null)}
                    />
                </div>
            )}

            {/* LLM Response Section */}
            {(isExecuting || llmResponse) && (
                <div className="mt-4 border-t border-gray-700 pt-4">
                    <h3 className="text-md font-semibold mb-2 text-green-300">
                        LLM Response:
                    </h3>

                    {/* Response Display */}
                    <div className="bg-gray-900 rounded-lg p-4 text-gray-300 text-sm min-h-[120px] flex items-center justify-center">
                        {isExecuting ? (
                            <LoadingSpinner text="Executing optimized prompt..." />
                        ) : llmResponse ? (
                            <div className="w-full">
                                <p className="whitespace-pre-wrap">{llmResponse}</p>
                            </div>
                        ) : (
                            <span className="text-gray-500 italic">
                                No response yet
                            </span>
                        )}
                    </div>

                    {/* Response Token Count and Copy Button */}
                    {llmResponse && (
                        <div className="mt-2 flex items-center justify-between">
                            <span className="text-xs text-gray-400">
                                Response tokens: <span className="text-green-400 font-semibold">{responseTokens}</span>
                            </span>
                            <button
                                className="bg-green-500 hover:bg-green-600 text-white text-xs font-semibold px-3 py-1 rounded flex items-center gap-1 transition-colors"
                                onClick={handleCopyResponse}
                            >
                                {isResponseCopied ? <FaCheck /> : <FaCopy />}
                                {isResponseCopied ? "Copied!" : "Copy Response"}
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
});

OptimizerPanel.displayName = 'OptimizerPanel';

export default OptimizerPanel;
