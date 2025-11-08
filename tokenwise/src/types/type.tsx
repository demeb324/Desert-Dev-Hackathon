// Define the data structure for the token input
export type TokenOptimizeInput = string;

// 1. Props for the Sender (Token Input)
export interface TokenInputProps {
    // A function that the component will call to send the data up to the parent
    onInputReady: (input: TokenOptimizeInput) => void;
}

// 2. Props for the Receiver (Token Optimizer)
export interface TokenOptimizerProps {
    // The data that the component will receive from the parent
    tokenInput: TokenOptimizeInput;
}