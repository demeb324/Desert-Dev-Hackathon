#!/usr/bin/env python3
"""
Text Optimizer Module

Python port of optimizer.ts - Provides utilities for optimizing text and prompts
through a flexible phase-based pipeline. Current optimizations include removing
polite lexicon, removing commas, and normalizing whitespace.
"""

import re
from typing import List, Callable


# Lexicon of polite words/phrases to remove during text optimization
# These patterns use word boundaries to match whole words only and are case-insensitive
LEXICON_PATTERNS = [
    r'\bplease\b',
    r'\bthank you\b',
    r'\bthanks\b',
    r"\bi'm sorry\b",
    r'\bsorry\b',
    r'\bno problem\b',
]


def remove_lexicon(text: str) -> str:
    """
    Removes all occurrences of predefined lexicon patterns from the input text.

    This function iterates through the LEXICON_PATTERNS list and applies each pattern
    to remove matching text. Patterns use word boundaries to avoid partial word matches
    (e.g., 'please' won't match 'pleased').

    Args:
        text: The text string to process

    Returns:
        The text with all lexicon patterns removed
    """
    modified_text = text
    for pattern in LEXICON_PATTERNS:
        modified_text = re.sub(pattern, '', modified_text, flags=re.IGNORECASE)
    return modified_text


def remove_commas(text: str) -> str:
    """
    Removes all comma characters from the input text.

    Args:
        text: The text string to process

    Returns:
        The text with all commas removed
    """
    return text.replace(',', '')


def remove_duplicate_whitespace(text: str) -> str:
    """
    Normalizes whitespace in the input text by trimming and collapsing multiple spaces.

    This function removes leading/trailing whitespace and replaces any sequence of
    whitespace characters (spaces, tabs, newlines) with a single space.

    Args:
        text: The text string to process

    Returns:
        The text with normalized whitespace
    """
    return re.sub(r'\s+', ' ', text).strip()


# Ordered list of optimization phases to apply to text
# Each phase is a function that takes a string and returns a processed string
# Phases are applied sequentially in the order defined here
OPTIMIZATION_PHASES: List[Callable[[str], str]] = [
    remove_lexicon,
    remove_commas,
    remove_duplicate_whitespace,
]


def optimize_prompt(text: str) -> str:
    """
    Optimizes text/prompts by applying a series of transformations.

    This function iterates through the OPTIMIZATION_PHASES list, applying each
    phase sequentially. The current pipeline includes:
    1. Removes unnecessary polite words from the lexicon
    2. Removes all commas
    3. Normalizes whitespace by trimming and collapsing duplicates

    Use this function to clean and optimize text for more efficient prompts.
    New phases can be added by defining a function and adding it to OPTIMIZATION_PHASES.

    Args:
        text: The text or prompt string to optimize

    Returns:
        The optimized text with all phases applied

    Examples:
        >>> optimize_prompt("Please   help me,  thank you")
        'help me'

        >>> optimize_prompt("I'm sorry, but no problem! Thanks for understanding.")
        'but for understanding.'
    """
    result = text
    for phase in OPTIMIZATION_PHASES:
        result = phase(result)
    return result


if __name__ == "__main__":
    # Quick test
    test_prompts = [
        "Please help me, thank you",
        "I'm sorry, but no problem! Thanks for understanding.",
        "Could you please explain this,   thank you very much?",
    ]

    print("Optimizer Test:")
    print("=" * 70)
    for prompt in test_prompts:
        optimized = optimize_prompt(prompt)
        print(f"\nOriginal:  {prompt}")
        print(f"Optimized: {optimized}")
    print("\n" + "=" * 70)
