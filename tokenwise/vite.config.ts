import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from "@tailwindcss/vite";

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [ tailwindcss(),react()],
    server: {
        port: 5173, // you can change if needed
        open: true, // automatically opens the browser
    },
    build: {
        outDir: 'dist',
    },
    resolve: {
        alias: {
            '@': '/src', // optional: allows clean imports like "@/components/Navbar"
        },
    },
});
