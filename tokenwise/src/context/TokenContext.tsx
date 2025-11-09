import React, { createContext, useContext, useState } from "react";

export type ModelName = "gpt-4" | "gpt-3.5" | "claude-3";

export interface TokenState {
    prompt: string;
    setPrompt: (p: string) => void;
    tokens: number;
    setTokens: (n: number) => void;
    model: ModelName;
    setModel: (m: ModelName) => void;
    updateTokenData: (data: { prompt?: string; tokens?: number; model?: ModelName }) => void;
    resetTokenData: () => void;
}

const TokenContext = createContext<TokenState | undefined>(undefined);

export const useTokenContext = (): TokenState => {
    const ctx = useContext(TokenContext);
    if (!ctx) throw new Error("useTokenContext must be used inside TokenProvider");
    return ctx;
};

export const TokenProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [prompt, setPrompt] = useState<string>("");
    const [tokens, setTokens] = useState<number>(0);
    const [model, setModel] = useState<ModelName>("gpt-4");

    // 🧩 Unified update function (optional convenience)
    const updateTokenData = (data: { prompt?: string; tokens?: number; model?: ModelName }) => {
        if (data.prompt !== undefined) setPrompt(data.prompt);
        if (data.tokens !== undefined) setTokens(data.tokens);
        if (data.model !== undefined) setModel(data.model);
    };

    // 🔄 Reset function (for "Clear" button etc.)
    const resetTokenData = () => {
        setPrompt("");
        setTokens(0);
        setModel("gpt-4");
    };

    return (
        <TokenContext.Provider
            value={{
                prompt,
                setPrompt,
                tokens,
                setTokens,
                model,
                setModel,
                updateTokenData,
                resetTokenData,
            }}
        >
            {children}
        </TokenContext.Provider>
    );
};
