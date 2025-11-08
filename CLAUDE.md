# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a tech talks demo repository (2025) containing a TypeScript-based text optimizer module. The project focuses on optimizing text/prompts by removing polite lexicon and normalizing whitespace.

## Architecture

The codebase consists of a single TypeScript module (`scripts/optimizer.ts`) that exports a text optimization pipeline:

1. **Lexicon Removal** (`remove_lexicon`): Strips polite/unnecessary words from text using a configurable lexicon array (currently: "please", "thank you", "thanks")
2. **Whitespace Normalization** (`remove_duplicate_whitespace`): Trims and collapses multiple whitespace characters into single spaces
3. **Pipeline** (`opimize_prompt`): Combines both transformations in sequence

The main export is `opimize_prompt()` which applies both optimizations to input strings.

## Development Commands

This repository currently has no build system, test framework, or package.json configured. To work with the TypeScript code:

```bash
# Run TypeScript compiler directly
npx tsc scripts/optimizer.ts

# Or use ts-node to execute
npx ts-node scripts/optimizer.ts
```

## Known Issues

- The main export function has a typo: `opimize_prompt` should likely be `optimize_prompt`
