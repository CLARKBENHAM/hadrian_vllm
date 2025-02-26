# %%
# evaluation.py
import re
import json
import os
import pandas as pd
import logging

from hadrian_vllm.utils import string_to_unicode_hex


def load_gdt_symbols():
    """Load the GD&T symbols mapping"""
    with open("data/gd_t_symbols.json", "r") as f:
        return json.load(f)


def normalize_text(text):
    """Normalize text by removing whitespace and case"""
    if text is None:
        return ""
    return re.sub(r"\s+", "", text.lower())


def normalize_gdt(text, symbols_mapping):
    """
    Normalize GD&T text by handling symbol/word equivalence.
    """
    if text is None:
        return ""

    symbol_to_name = symbols_mapping.get("symbol_to_name", {})

    normalized = text

    # Replace symbols with placeholders
    for symbol, names in symbol_to_name.items():
        if symbol in normalized:
            # Replace symbol with a unique placeholder
            normalized = normalized.replace(symbol, placeholder)

            # Also create versions with the name instead of symbol
            alternatives = [normalized.replace(placeholder, name) for name in names]
            normalized = normalized.replace(placeholder, symbol)  # restore original
            alternatives.append(normalized)

            return [normalize_text(alt) for alt in alternatives]

    # Replace names with placeholders
    for name, symbol in name_to_symbol.items():
        if name.lower() in normalized.lower():
            # Create case-insensitive pattern
            pattern = re.compile(re.escape(name), re.IGNORECASE)

            # Replace name with a unique placeholder
            placeholder = f"__NAME_{hash(name)}_"
            normalized = pattern.sub(placeholder, normalized)

            # Create versions with the symbol instead of name
            alt_with_symbol = normalized.replace(placeholder, symbol)
            normalized = normalized.replace(placeholder, name)  # restore original

            return [normalize_text(normalized), normalize_text(alt_with_symbol)]

    # If no replacements were made, return the normalized text
    return [normalize_text(normalized)]


def normalize_gdt(text, symbols_mapping):
    """
    Normalize GD&T text by converting both symbols and their corresponding names
    to a placeholder. Longer matching strings are replaced first.

    Also logs an info-level warning if the text contains any Unicode characters
    that are not ASCII and are not present as keys in symbol_to_name.
    """

    def string_to_unicode_hex(input_string):
        return "".join([f"{ord(char):04x}" for char in input_string])

    if text is None:
        return ""

    symbol_to_name = symbols_mapping.get("symbol_to_name", {})

    # Log if text contains non-ASCII characters that are not in the mapping keys.
    # (Assumes keys are the allowed Unicode characters.)
    non_ascii_chars = {c for c in text if ord(c) > 127 and c not in symbol_to_name}
    if non_ascii_chars:
        logging.info(f"Unrecognized unicode characters in text: {', '.join(non_ascii_chars)}")

    # Build a list of alternatives: both the symbols (keys) and their name alternatives.
    alternatives = []
    alternative_map = {}
    for symbol, names in symbol_to_name.items():
        alternatives.append(symbol)
        alternatives.extend(names)
        placeholder = f"__{string_to_unicode_hex(symbol)}__"
        for n in names + [symbol]:
            alternative_map[n] = placeholder

    # Remove duplicates and sort by length descending to replace longer strings first.
    alternatives = list(set(alternatives))
    alternatives.sort(key=len, reverse=True)

    # Create a regex pattern that matches any of the alternatives.
    pattern = re.compile("|".join(map(re.escape, alternatives)))

    # Replace all occurrences with their corresponding unique placeholder.
    normalized = pattern.sub(lambda match: alternative_map[match.group(0)], text)

    return normalize_text(normalized)


def evaluate_answer(prediction, ground_truth):
    """
    Evaluate whether a prediction matches the ground truth, accounting for
    symbol/word equivalence and whitespace variations.

    Args:
        prediction: The predicted answer (string or list of strings)
        ground_truth: The ground truth answer

    Returns:
        Float in [0, 1] indicating correctness
    """
    if ground_truth is None or (isinstance(ground_truth, float) and pd.isna(ground_truth)):
        # No ground truth available
        raise ValueError("Ground truth is missing or NaN. Cannot evaluate.")

    # Load GD&T symbols mapping
    symbols_mapping = load_gdt_symbols()

    # Handle the case where prediction is a list (multiple completions)
    if isinstance(prediction, list):
        for pred in prediction:
            if evaluate_single_answer(pred, ground_truth, symbols_mapping) == 1.0:
                return 1.0
        return 0.0
    else:
        return evaluate_single_answer(prediction, ground_truth, symbols_mapping)


def evaluate_single_answer(prediction, ground_truth, symbols_mapping):
    """Evaluate a single prediction against the ground truth"""
    if prediction is None:
        return 0.0

    # Get normalized versions of the prediction and ground truth
    pred_normalized = normalize_gdt(prediction, symbols_mapping)
    gt_normalized = normalize_gdt(ground_truth, symbols_mapping)

    # Check if any normalized version of the prediction matches any normalized version of the ground truth
    if isinstance(pred_normalized, list) and isinstance(gt_normalized, list):
        for p in pred_normalized:
            for g in gt_normalized:
                if p == g:
                    return 1.0
    elif isinstance(pred_normalized, list):
        for p in pred_normalized:
            if p == gt_normalized[0]:
                return 1.0
    elif isinstance(gt_normalized, list):
        for g in gt_normalized:
            if pred_normalized[0] == g:
                return 1.0
    else:
        if pred_normalized == gt_normalized:
            return 1.0
    return 0.0


def calculate_metrics(df, result_column):
    """
    Calculate metrics for a result column.

    Args:
        df: DataFrame with results
        result_column: Column containing the results

    Returns:
        Dictionary with metrics (correct, incorrect, unanswered, accuracy)
    """
    # Count correct, incorrect, and unanswered
    correct = 0
    incorrect = 0
    unanswered = 0
    total = 0

    for _, row in df.iterrows():
        if "Specification" not in row or pd.isna(row["Specification"]):
            # Skip rows without ground truth
            continue

        ground_truth = row["Specification"]

        if result_column not in row or pd.isna(row[result_column]) or row[result_column] == "":
            unanswered += 1
        else:
            try:
                prediction = row[result_column]
                if evaluate_answer(prediction, ground_truth) == 1.0:
                    correct += 1
                else:
                    incorrect += 1
            except Exception as e:
                print(f"Error evaluating answer: {e}")
                incorrect += 1

        total += 1

    # Calculate metrics
    if total == 0:
        raise ValueError("No valid rows found for evaluation")

    accuracy = correct / total
    percent_correct = correct / total * 100
    percent_incorrect = incorrect / total * 100
    percent_unanswered = unanswered / total * 100

    return {
        "correct": correct,
        "incorrect": incorrect,
        "unanswered": unanswered,
        "total": total,
        "accuracy": accuracy,
        "percent_correct": percent_correct,
        "percent_incorrect": percent_incorrect,
        "percent_unanswered": percent_unanswered,
    }
