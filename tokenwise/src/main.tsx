import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import {TokenProvider} from "./context/TokenContext.tsx";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
        <TokenProvider>
        <App />
        </TokenProvider>
    </React.StrictMode>
);
