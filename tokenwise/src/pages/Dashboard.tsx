import TokenInput from "../components/TokenInput";
import TokenAnalyzer from "../components/TokenAnalyzer";
import OptimizerPanel from "../components/OptimizerPanel";
import {useState} from "react";
import type {TokenOptimizeInput} from "../types/type.tsx";

const Dashboard: React.FC = () => {
// 💡 State is lifted up to the common parent
    const [optimizerInput, setOptimizerInput] = useState<TokenOptimizeInput>('');
    return (
        <div className="min-h-screen bg-gradient-to-br text-white p-5">
            <div className="max-w-7xl mx-auto">
                <div className="grid md:grid-cols-4 gap-6 mt-6">
                    <div className="md:col-span-1">
                        <TokenInput onInputReady={setOptimizerInput}/>
                    </div>
                    <div className="md:col-span-2">
                        <TokenAnalyzer />
                    </div>
                    <div className="md:col-span-1">
                        <OptimizerPanel tokenInput={optimizerInput}/>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;

