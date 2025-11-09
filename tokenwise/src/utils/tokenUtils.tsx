// lightweight utils for client-side token analytics
export const estimateTokens = (text: string): number => {
    // Simple heuristic: ~1.3 words per token, but handle empty string
    if (!text || text.trim().length === 0) return 0;
    const words = text.trim().split(/\s+/).length;
    // adjust slightly by punctuation density and length
    const punctuationFactor = (text.match(/[.,!?;:()"]/g)?.length ?? 0) * 0.02;
    const est = Math.ceil(words * 1.3 + punctuationFactor * words);
    return Math.max(1, est);
};

/**
 * Parse prompt into sections if user uses explicit markers:
 * e.g. "SYSTEM: ... \n USER: ... \n ASSISTANT: ..."
 * falls back to single USER section.
 */
export const parseSections = (text: string) => {
    const sections: { name: string; text: string }[] = [];
    const upper = text.toUpperCase();
    if (upper.includes("SYSTEM:") || upper.includes("USER:") || upper.includes("ASSISTANT:")) {
        // crude split: look for markers and slice
        const re = /(SYSTEM:|USER:|ASSISTANT:)/gi;
        const parts = text.split(re).map(s => s.trim()).filter(Boolean);
        // parts alternates marker, content
        for (let i = 0; i < parts.length; i += 2) {
            const marker = parts[i].replace(":", "").trim();
            const content = parts[i + 1] ?? "";
            sections.push({ name: marker, text: content });
        }
    } else {
        sections.push({ name: "USER", text });
    }
    return sections;
}

/** cost and energy model helpers (estimates) */
export const PRICE_PER_1K_TOKENS_BY_MODEL: Record<string, number> = {
    // example numbers (USD per 1k tokens) — replace with accurate values or call to backend
    "gpt-4": 0.03,
    "gpt-3.5": 0.002,
    "claude-3": 0.025,
};

export const ENERGY_JOULES_PER_TOKEN_BY_MODEL: Record<string, number> = {
    // rough illustrative values: joules per token (very approximate)
    "gpt-4": 0.0005,
    "gpt-3.5": 0.0002,
    "claude-3": 0.0004,
};

export const estimateCostUSD = (model: string, tokens: number): number => {
    const pricePer1k = PRICE_PER_1K_TOKENS_BY_MODEL[model] ?? 0.01;
    return (pricePer1k / 1000) * tokens;
};

export const estimateEnergyJoules = (model: string, tokens: number): number => {
    const jPerToken = ENERGY_JOULES_PER_TOKEN_BY_MODEL[model] ?? 0.0003;
    return jPerToken * tokens;
};

export const joulesToKWh = (joules: number): number => {
    return joules / 3_600_000;
};

// default carbon intensity (g CO2 per kWh) — can be configurable by user/location
export const DEFAULT_GRID_GCO2_PER_KWH = 400;
export const estimateCO2grams = (kwh: number, gCO2PerKwh = DEFAULT_GRID_GCO2_PER_KWH) => {
    return kwh * gCO2PerKwh;
};
