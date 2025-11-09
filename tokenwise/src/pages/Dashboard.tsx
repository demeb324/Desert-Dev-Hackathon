import TokenInput from "../components/TokenInput";
import OptimizerPanel from "../components/OptimizerPanel";
import type { OptimizerPanelHandle } from "../components/OptimizerPanel";
import {useState, useRef} from "react";
import type {TokenOptimizeInput} from "../types/type.tsx";

const Dashboard: React.FC = () => {
// 💡 State is lifted up to the common parent
    const [optimizerInput, setOptimizerInput] = useState<TokenOptimizeInput>('');
    const optimizerRef = useRef<OptimizerPanelHandle>(null);

    const handleAnalyze = (prompt: string) => {
        setOptimizerInput(prompt);

        // Trigger auto-optimization and execution after state update
        setTimeout(() => {
            optimizerRef.current?.autoOptimizeAndExecute();
        }, 0);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br text-white p-5">
            <div className="max-w-6xl mx-auto">
                <div className="space-y-6 mt-6">
                    <TokenInput onInputReady={handleAnalyze}/>
                    <OptimizerPanel ref={optimizerRef} tokenInput={optimizerInput}/>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;

