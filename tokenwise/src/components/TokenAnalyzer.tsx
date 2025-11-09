import React, { useMemo } from "react";
import { FaChartBar, FaDollarSign, FaBolt, FaLeaf } from "react-icons/fa";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";
import { useTokenContext } from "../context/TokenContext.tsx";
import {
    estimateCostUSD,
    estimateEnergyJoules,
    joulesToKWh,
    estimateCO2grams,
    parseSections,
    PRICE_PER_1K_TOKENS_BY_MODEL,
} from "../utils/tokenUtils";

const COLORS = ["#06b6d4", "#f59e0b", "#ef4444", "#10b981"];

const TokenAnalyzer: React.FC = () => {
    const { prompt, tokens, model } = useTokenContext();

    // breakdown into sections (SYSTEM/USER/ASSISTANT)
    const sections = useMemo(() => parseSections(prompt), [prompt]);

    const sectionData = useMemo(() => {
        // estimate tokens per section proportionally by words then refine
        const totalWords = prompt.trim() ? prompt.trim().split(/\s+/).length : 0;
        const base = tokens > 0 ? tokens : 0;
        if (totalWords === 0 || base === 0) {
            return [{ name: "USER", tokens: base, text: prompt }];
        }
        return sections.map((s, idx) => {
            const words = s.text.trim() ? s.text.trim().split(/\s+/).length : 0;
            const share = words / Math.max(1, totalWords);
            const t = Math.max(0, Math.round(base * share));
            return { name: s.name, tokens: t, text: s.text };
        });
    }, [prompt, sections, tokens]);

    const cost = estimateCostUSD(model, tokens);
    const energyJ = estimateEnergyJoules(model, tokens);
    const energyKwh = joulesToKWh(energyJ);
    const co2g = estimateCO2grams(energyKwh);

    const pieData = sectionData.map((s) => ({ name: s.name, value: s.tokens }));
    const barData = sectionData.map((s) => ({ section: s.name, tokens: s.tokens }));

    return (
        <div className="bg-gray-800 p-5 rounded-2xl shadow-xl">
            <h2 className="text-xl font-semibold mb-3 flex items-center gap-2 text-yellow-300">
                <FaChartBar /> Token Analyzer
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-900 rounded-lg p-3 h-56">
                    <h3 className="text-sm text-gray-300 mb-2">Token distribution</h3>
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={70} innerRadius={30} label>
                                {pieData.map((_, idx) => (
                                    <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                <div className="bg-gray-900 rounded-lg p-3 h-56 flex flex-col justify-between">
                    <div>
                        <div className="text-xs text-gray-400">Estimated total tokens</div>
                        <div className="text-3xl font-bold text-cyan-400">{tokens}</div>
                        <div className="mt-2 text-sm text-gray-300">
                            Price (per 1k tokens): <span className="font-semibold text-green-300">${PRICE_PER_1K_TOKENS_BY_MODEL[model]}</span>
                        </div>
                    </div>

                    <div className="flex gap-2 mt-3">
                        <div className="flex-1 bg-gray-800 px-3 py-2 rounded">
                            <div className="text-xs text-gray-400 flex items-center gap-2"><FaDollarSign /> Cost</div>
                            <div className="text-lg font-bold text-green-400 ">${cost.toFixed(5)}</div>
                        </div>
                        <div className="flex-1 bg-gray-800 px-3 py-2 rounded">
                            <div className="text-xs text-gray-400 flex items-center gap-2"><FaBolt /> Energy</div>
                            <div className="text-lg font-bold text-yellow-400">{energyJ.toExponential(1)}J</div>
                        </div>
                        <div className="flex-1 bg-gray-800 px-3 py-2 rounded">
                            <div className="text-xs text-gray-400 flex items-center gap-2"><FaLeaf /> CO₂</div>
                            <div className="text-lg font-bold text-green-300">{co2g.toFixed(3)}g</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-4 bg-gray-900 rounded-lg p-3">
                <h3 className="text-sm text-gray-300 mb-2">Tokens per section</h3>
                <div style={{ height: 160 }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={barData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                            <XAxis dataKey="section" stroke="#9ca3af" />
                            <YAxis stroke="#9ca3af" />
                            <Tooltip />
                            <Bar dataKey="tokens" fill="#06b6d4" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="mt-4 bg-gray-900 rounded-lg p-3 text-sm text-gray-300">
                <div className="font-semibold mb-2">Insights</div>
                <ul className="list-disc ml-5">
                    <li>Token count drives cost & energy: {tokens} tokens ≈ <span className="font-medium">${cost.toFixed(6)}</span>.</li>
                    <li>Estimated energy ~ <span className="font-medium">{energyKwh.toExponential(3)}</span> kWh ({energyJ.toExponential(2)} J).</li>
                    <li>Estimated CO₂ footprint ~ <span className="font-medium">{co2g.toFixed(3)} g</span> (using grid intensity = 400 gCO₂/kWh).</li>
                </ul>
            </div>
        </div>
    );
};

export default TokenAnalyzer;
