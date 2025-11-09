/**
 * Text Optimizer Module
 *
 * Provides utilities for optimizing text and prompts through a flexible phase-based
 * pipeline. Current optimizations include removing polite lexicon, removing commas,
 * and normalizing whitespace. New phases can be easily added to the optimizationPhases array.
 */

/**
 * Lexicon of regular expressions matching polite words to be removed during text optimization.
 * These patterns use word boundaries to match whole words only and are case-insensitive.
 * Each pattern is applied globally across the input text.
 */
const lexicon = [
  /\bplease\b/gi,
  /\bthank you\b/gi,
  /\bthanks\b/gi,
  /\bi'm sorry\b/gi,
  /\bsorry\b/gi,
  /\bno problem\b/gi,
  /\bthe following\b/gi,
  /\bvery\b/gi,
  /\bextremely\b/gi,
  /\bcompletely\b/gi,
  /\babsolutely\b/gi,
  /\bincredibly\b/gi,
  /\bgenerally\b/gi,
  /\bessentially\b/gi,
  /\breally\b/gi,
  /\btell me\b/gi,
  /\bcan you\b/gi,
  /\bcould you\b/gi,
  /\bwould you\b/gi,
];

/**
 * Removes all occurrences of predefined lexicon patterns from the input text.
 *
 * This function iterates through the lexicon array of RegExp objects and applies
 * each pattern to remove matching text. Patterns use word boundaries to avoid
 * partial word matches (e.g., 'please' won't match 'pleased').
 *
 * @param input - The text string to process
 * @returns The text with all lexicon patterns removed
 */
const remove_lexicon = (input: string): string => {
  let modifiedString = input;
  for (const regex of lexicon) {
    modifiedString = modifiedString.replace(regex, '');
  }

  return modifiedString;
};

/**
 * Normalizes whitespace in the input text by trimming and collapsing multiple spaces.
 *
 * This function removes leading/trailing whitespace and replaces any sequence of
 * whitespace characters (spaces, tabs, newlines) with a single space.
 *
 * @param input - The text string to process
 * @returns The text with normalized whitespace
 */
const remove_duplicate_whitespace = (input: string): string => {
  return input.trim().replace(/\s+/g, ' ');

};

/**
 * Removes all comma characters from the input text.
 *
 * @param input - The text string to process
 * @returns The text with all commas removed
 */
const remove_commas = (input: string): string => {
  return input.replace(/,/g, '');
};

/**
 * Ordered list of optimization phases to apply to text.
 * Each phase is a function that takes a string and returns a processed string.
 * Phases are applied sequentially in the order defined here.
 */
const optimizationPhases = [
  remove_lexicon,
  remove_commas,
  remove_duplicate_whitespace,
];

/**
 * Optimizes text/prompts by applying a series of transformations.
 *
 * This function iterates through the optimizationPhases array, applying each
 * phase sequentially. The current pipeline includes:
 * 1. Removes unnecessary polite words from the lexicon
 * 2. Removes all commas
 * 3. Normalizes whitespace by trimming and collapsing duplicates
 *
 * Use this function to clean and optimize text for more efficient prompts.
 * New phases can be added by defining a function and adding it to optimizationPhases.
 *
 * @param input - The text or prompt string to optimize
 * @returns The optimized text with all phases applied
 * @example
 * ```typescript
 * optimize_prompt("Please   help me,  thank you")
 * // Returns: "help me"
 *
 * optimize_prompt("I'm sorry, but no problem! Thanks for understanding.")
 * // Returns: "but for understanding."
 * ```
 */
export const optimize_prompt = (input: string): string => {
  let text = input;
  for (const phase of optimizationPhases) {
    text = phase(text);
  }
  return text;
};
