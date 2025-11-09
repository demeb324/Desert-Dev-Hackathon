# OptiGreen - AI Prompt Optimizer

OptiGreen is a web application that analyzes and optimizes AI prompts to reduce token usage, cost, and environmental impact. Built with React, TypeScript, and Vite, it integrates with LM Studio for AI-powered prompt optimization.

## Features

- **Token Analysis**: Real-time token counting and cost estimation
- **AI-Powered Optimization**: Uses LM Studio's local LLM to optimize prompts intelligently
- **Energy Impact Tracking**: Calculate and visualize energy consumption and CO2 emissions
- **Interactive Dashboard**: View token distribution, cost breakdown, and optimization results
- **Responsive Design**: Modern UI with TailwindCSS and dark theme

## Prerequisites

Before running OptiGreen, you need:

1. **Node.js** (v18 or higher)
2. **LM Studio** installed and running
   - Download from [lmstudio.ai](https://lmstudio.ai/)
   - Load a model (recommended: Llama 3 or similar)
   - Start the local server (default port: 1234)

## Installation

1. **Clone the repository**
   ```bash
   cd tokenwise
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env.local
   ```

   Edit `.env.local` if needed:
   ```env
   VITE_LM_STUDIO_URL=http://localhost:1234/v1
   VITE_LM_STUDIO_TIMEOUT=30000
   ```

## Running the Application

### Step 1: Start LM Studio

1. Open LM Studio
2. Load a model (e.g., Llama 3 8B, Mistral 7B, or similar)
3. Go to the "Server" tab
4. Start the local server on port 1234
5. Verify the server is running (you should see a green indicator)

### Step 2: Start the Development Server

```bash
npm run dev
```

The application will open automatically at `http://localhost:5173`

## Usage

1. **Enter a Prompt**
   - Type or paste your AI prompt in the input area
   - Tokens are counted automatically (with 500ms debounce)
   - View estimated cost and token count

2. **Analyze the Prompt**
   - Click the "Analyze" button to visualize token distribution
   - View breakdown by section (SYSTEM/USER/ASSISTANT)
   - See cost, energy, and CO2 estimates

3. **Optimize the Prompt**
   - Click "Optimize" in the Optimizer panel
   - LM Studio processes the prompt to remove unnecessary words
   - View before/after token counts and percentage saved
   - Copy the optimized prompt to clipboard

## Project Structure

```
tokenwise/
├── src/
│   ├── components/       # React components
│   │   ├── Navbar.tsx
│   │   ├── TokenInput.tsx
│   │   ├── TokenAnalyzer.tsx
│   │   ├── OptimizerPanel.tsx
│   │   ├── LoadingSpinner.tsx
│   │   ├── ErrorAlert.tsx
│   │   └── ErrorBoundary.tsx
│   ├── pages/            # Page components
│   │   ├── Dashboard.tsx
│   │   └── About.tsx
│   ├── context/          # React Context
│   │   └── TokenContext.tsx
│   ├── services/         # API services
│   │   └── lmStudioService.ts
│   ├── types/            # TypeScript definitions
│   │   └── type.tsx
│   ├── utils/            # Utility functions
│   │   └── tokenUtils.tsx
│   └── App.tsx           # Main app component
├── .env.example          # Environment variables template
├── .env.local            # Local environment config (gitignored)
├── vite.config.ts        # Vite configuration
├── tailwind.config.js    # Tailwind CSS config
└── package.json          # Dependencies
```

## Technologies Used

- **React 19** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **TailwindCSS 4** - Styling
- **React Router 7** - Client-side routing
- **Recharts** - Data visualization
- **React Icons** - Icon library
- **LM Studio** - Local LLM API

## Troubleshooting

### "Cannot connect to LM Studio"

**Solutions:**
1. Verify LM Studio is running and server is started
2. Check that the server is on port 1234
3. Ensure a model is loaded in LM Studio
4. Try restarting both LM Studio and the dev server

### CORS Errors

The Vite dev server includes a proxy to avoid CORS issues. If you still encounter CORS errors:

1. Restart the development server: `npm run dev`
2. Check that LM Studio server is running locally (not on a different machine)
3. Verify the URL in `.env.local` is correct

### Token Count Shows "?"

This happens when LM Studio is unreachable. The app uses a fallback estimation (word count × 1.3), but for accurate counts, ensure LM Studio is connected.

### Slow Optimization

Optimization speed depends on:
- The model size loaded in LM Studio
- Your hardware (CPU/GPU/RAM)
- Prompt length

Typical optimization times:
- Short prompts (<50 tokens): 2-5 seconds
- Medium prompts (50-200 tokens): 5-15 seconds
- Long prompts (>200 tokens): 15-30 seconds

## Development

### Build for Production

```bash
npm run build
```

Output will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

### Lint

```bash
npm run lint
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_LM_STUDIO_URL` | `http://localhost:1234/v1` | LM Studio API base URL |
| `VITE_LM_STUDIO_TIMEOUT` | `30000` | Request timeout (milliseconds) |

## Contributing

This is a demo/research project. Feel free to fork and experiment!

## License

MIT

## Acknowledgments

- Built with [Vite](https://vitejs.dev/)
- Powered by [LM Studio](https://lmstudio.ai/)
- UI inspired by modern AI tool interfaces
