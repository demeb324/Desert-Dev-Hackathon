import React, { createContext, useContext, useState } from "react";

export interface TokenState {
    prompt: string;
    setPrompt: (p: string) => void;
    tokens: number;
    setTokens: (n: number) => void;
    cost: number;
    setCost: (c: number) => void;
    isLoading: boolean;
    setLoading: (loading: boolean) => void;
    error: string | null;
    setError: (error: string | null) => void;
    updateTokenData: (data: {
        prompt?: string;
        tokens?: number;
        cost?: number;
        isLoading?: boolean;
        error?: string | null;
    }) => void;
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
    const [cost, setCost] = useState<number>(0);
    const [isLoading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    // 🧩 Unified update function (optional convenience)
    const updateTokenData = (data: {
        prompt?: string;
        tokens?: number;
        cost?: number;
        isLoading?: boolean;
        error?: string | null;
    }) => {
        if (data.prompt !== undefined) setPrompt(data.prompt);
        if (data.tokens !== undefined) setTokens(data.tokens);
        if (data.cost !== undefined) setCost(data.cost);
        if (data.isLoading !== undefined) setLoading(data.isLoading);
        if (data.error !== undefined) setError(data.error);
    };

    // 🔄 Reset function (for "Clear" button etc.)
    const resetTokenData = () => {
        setPrompt("");
        setTokens(0);
        setCost(0);
        setLoading(false);
        setError(null);
    };

    return (
        <TokenContext.Provider
            value={{
                prompt,
                setPrompt,
                tokens,
                setTokens,
                cost,
                setCost,
                isLoading,
                setLoading,
                error,
                setError,
                updateTokenData,
                resetTokenData,
            }}
        >
            {children}
        </TokenContext.Provider>
    );
};
