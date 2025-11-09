import {FaMagic, FaCopy, FaCheck} from "react-icons/fa";
import type {TokenOptimizerProps} from "../types/type.tsx";
import {optimize_prompt} from "../../../scripts/optimizer.ts";
import {useState} from "react";
import {estimateTokens} from "../utils/tokenUtils.tsx";

const OptimizerPanel: React.FC<TokenOptimizerProps> = ({ tokenInput }) => {
    // Logic to process/optimize the tokenInput goes here
   // const optimizedResult = tokenInput ? tokenInput.toUpperCase() : ' Optimized prompt preview shown here...'; // Example optimization
    const optimizedResult=  tokenInput ? optimize_prompt(tokenInput) : ' Optimized prompt preview shown here...';

    const beforeTokens = estimateTokens(tokenInput)
    const afterTokens = estimateTokens(optimizedResult)
   // State to manage the visual feedback (e.g., change icon/text after successful copy)
    const [isCopied, setIsCopied] = useState(false);

    // 1. Define the asynchronous copy function
    const handleCopy = async () => {
        try {
            // Use the browser's Clipboard API to write the text
            await navigator.clipboard.writeText(optimizedResult);

            // 2. Provide feedback to the user
            setIsCopied(true);

            // Reset the feedback state after a short delay
            setTimeout(() => {
                setIsCopied(false);
            }, 2000);

        } catch (err) {
            console.error('Failed to copy text: ', err);
            // Optional: Add an alert or visual indicator for an error
            alert('Failed to copy prompt. Please try again or copy manually.');
        }
    };

    return (
        <div className="bg-gray-800 p-5 rounded-2xl shadow-xl">
            <h2 className="text-xl font-semibold mb-3 flex items-center gap-2 text-pink-300">
                <FaMagic /> Optimizer
            </h2>

            <table className="w-full text-sm text-gray-300 mb-3">
                <thead>
                <tr className="border-b border-gray-600">
                    <th className="text-left py-1">Section</th>
                    <th>Before</th>
                    <th>After</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                    <td>Prompt</td>
                    <td className="text-center">{beforeTokens}</td>
                    <td className="text-green-400 text-center">{afterTokens}</td>
                </tr>
                </tbody>
            </table>

            <div className="bg-gray-900 rounded-lg p-3 text-gray-400 text-sm">
               <strong>{optimizedResult}</strong>
            </div>

            <button className="mt-3 w-full bg-pink-500 hover:bg-pink-600 text-black font-semibold py-2 rounded-lg flex items-center justify-center gap-2" onClick={handleCopy}>
                {/* Conditionally show the Check icon on success */}
                {isCopied ? <FaCheck /> : <FaCopy />}

                {/* Conditionally change the button text */}
                {isCopied ? 'Copied!' : 'Copy Optimized Prompt'}
            </button>
        </div>
    );
};

export default OptimizerPanel;
