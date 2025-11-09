/**
 * LM Studio API Service
 *
 * Provides integration with LM Studio's local API for:
 * - AI-powered prompt optimization
 * - Accurate token counting
 * - Connection health checks
 */

// Configuration
// Use Vite proxy in development to avoid CORS issues
const LM_STUDIO_URL = import.meta.env.DEV
  ? '/api/lm-studio/v1'  // Use proxy in development
  : (import.meta.env.VITE_LM_STUDIO_URL || 'http://localhost:1234/v1');
const LM_STUDIO_TIMEOUT = parseInt(import.meta.env.VITE_LM_STUDIO_TIMEOUT || '30000');
const LM_STUDIO_MODEL = import.meta.env.VITE_LM_STUDIO_MODEL || '';

// Types
export interface LMStudioError {
  message: string;
  code?: string;
  details?: unknown;
}

interface ChatCompletionRequest {
  model: string;  // Required by LM Studio
  messages: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
  temperature?: number;
  max_tokens?: number;
}

interface ChatCompletionResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
  usage?: {
    total_tokens: number;
    prompt_tokens: number;
    completion_tokens: number;
  };
}

/**
 * LM Studio API Service
 */
export class LMStudioService {
  private baseURL: string;
  private timeout: number;

  constructor(baseURL: string = LM_STUDIO_URL, timeout: number = LM_STUDIO_TIMEOUT) {
    this.baseURL = baseURL;
    this.timeout = timeout;
  }

  /**
   * Test connection to LM Studio server
   */
  async testConnection(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout for health check

      const response = await fetch(`${this.baseURL}/models`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      console.error('LM Studio connection test failed:', error);
      return false;
    }
  }

  /**
   * Optimize a prompt using LM Studio
   *
   * @param prompt - The original prompt to optimize
   * @param optimizationInstruction - Optional custom optimization instruction
   * @returns The optimized prompt text
   */
  async optimizePrompt(
    prompt: string,
    optimizationInstruction: string = 'Rewrite this prompt to be more concise while preserving its core meaning and intent. Remove unnecessary words, polite language, and redundancy. Maintain clarity and specificity.'
  ): Promise<string> {
    if (!prompt || prompt.trim() === '') {
      throw this.createError('Cannot optimize empty prompt');
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const requestBody: ChatCompletionRequest = {
        model: LM_STUDIO_MODEL,  // Required by LM Studio
        messages: [
          {
            role: 'system',
            content: optimizationInstruction,
          },
          {
            role: 'user',
            content: `Original prompt: ${prompt}\n\nOptimized prompt:`,
          },
        ],
        temperature: 0.3, // Lower temperature for more consistent optimization
        max_tokens: 1000, // Reasonable limit for optimized prompts
      };

      // Debug logging
      console.log('[LM Studio] Sending optimization request:', {
        url: `${this.baseURL}/chat/completions`,
        model: requestBody.model,
        messagesCount: requestBody.messages.length,
      });

      const response = await fetch(`${this.baseURL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[LM Studio] API error response:', {
          status: response.status,
          statusText: response.statusText,
          body: errorText,
        });
        throw this.createError(
          `LM Studio API error: ${response.status} ${response.statusText}`,
          response.status.toString(),
          errorText
        );
      }

      const data: ChatCompletionResponse = await response.json();

      if (!data.choices || data.choices.length === 0) {
        throw this.createError('No response from LM Studio');
      }

      const optimizedPrompt = data.choices[0].message.content.trim();

      if (!optimizedPrompt) {
        throw this.createError('LM Studio returned empty optimization');
      }

      return optimizedPrompt;
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw this.createError('Request timed out. Please try again.');
        }
        if (error.message.includes('Failed to fetch')) {
          throw this.createError('Cannot connect to LM Studio. Please ensure LM Studio is running on port 1234 and a model is loaded.');
        }
      }
      throw error;
    }
  }

  /**
   * Get accurate token count for text using LM Studio
   *
   * @param text - The text to count tokens for
   * @returns The number of tokens
   */
  async getTokenCount(text: string): Promise<number> {
    if (!text || text.trim() === '') {
      return 0;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      // Use chat completions with max_tokens=0 to get token count without generation
      const requestBody: ChatCompletionRequest = {
        model: LM_STUDIO_MODEL,  // Required by LM Studio
        messages: [
          {
            role: 'user',
            content: text,
          },
        ],
        max_tokens: 0, // Don't generate, just count tokens
      };

      console.log('[LM Studio] Counting tokens with model:', LM_STUDIO_MODEL);

      const response = await fetch(`${this.baseURL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        // If this fails, fall back to a simple estimation
        console.warn('Token count API failed, using estimation');
        return this.estimateTokenCount(text);
      }

      const data: ChatCompletionResponse = await response.json();

      if (data.usage && data.usage.prompt_tokens) {
        return data.usage.prompt_tokens;
      }

      // Fallback to estimation if usage data not available
      return this.estimateTokenCount(text);
    } catch (error) {
      console.warn('Token count error, using estimation:', error);
      return this.estimateTokenCount(text);
    }
  }

  /**
   * Fallback token estimation
   * Uses a simple heuristic: word count * 1.3
   */
  private estimateTokenCount(text: string): number {
    const words = text.match(/\b\w+\b/g);
    const wordCount = words ? words.length : 0;
    return Math.ceil(wordCount * 1.3);
  }

  /**
   * Create a standardized error object
   */
  private createError(message: string, code?: string, details?: unknown): LMStudioError {
    return {
      message,
      code,
      details,
    };
  }
}

// Export a singleton instance
export const lmStudioService = new LMStudioService();
