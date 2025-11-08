/**
 * Text Optimizer Module
 *
 * Provides utilities for optimizing text and prompts by removing unnecessary
 * polite lexicon and normalizing whitespace.
 */

/**
 * Lexicon of polite words to be removed during text optimization.
 * These words are considered unnecessary for prompt efficiency.
 */
const lexicon = [
  "please",
  "thank you",
  "thanks",
];

/**
 * Removes all occurrences of predefined lexicon words from the input text.
 *
 * This function iterates through the lexicon array and removes all instances
 * of each word, using regex to ensure special characters are properly escaped.
 *
 * @param input - The text string to process
 * @returns The text with all lexicon words removed
 */
const remove_lexicon = (input: string): string => {
  let modifiedString = input;
  for (const substring of lexicon) {
    // Use a regular expression with the 'g' flag to replace all occurrences
    // Escape special characters in the substring if it might contain them
    const escapedSubstring = substring.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(escapedSubstring, 'g');
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
 * Optimizes text/prompts by applying a series of transformations.
 *
 * This is the main optimization pipeline that:
 * 1. Removes unnecessary polite words from the lexicon
 * 2. Normalizes whitespace by trimming and collapsing duplicates
 *
 * Use this function to clean and optimize text for more efficient prompts.
 *
 * @param input - The text or prompt string to optimize
 * @returns The optimized text with lexicon removed and whitespace normalized
 * @example
 * ```typescript
 * opimize_prompt("Please   help me,  thank you")
 * // Returns: "help me,"
 * ```
 */
export const opimize_prompt = (input: string): string => {
  let text = input;
  text = remove_lexicon(text);
  text = remove_duplicate_whitespace(text);
  return text;
};
