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
    # print("normed")
    # print(pred_normalized)
    # print(gt_normalized)

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


def evaluate_single_answer_easy(prediction, ground_truth, symbols_mapping):
    """Removing as many rules as possible
    Evaluate a single prediction against the ground truth
    """
    if prediction is None:
        return 0.0

    # Get normalized versions of the prediction and ground truth
    # print(
    #     "pred",
    #     prediction,
    #     undo_formatting_rules(prediction),
    #     normalize_gdt(undo_formatting_rules(prediction), symbols_mapping),
    #     sep="\n",
    # )
    # print(
    #     "gt",
    #     ground_truth,
    #     undo_formatting_rules(ground_truth),
    #     normalize_gdt(undo_formatting_rules(ground_truth), symbols_mapping),
    #     sep="\n",
    # )

    pred_normalized = normalize_gdt(undo_formatting_rules(prediction), symbols_mapping)
    gt_normalized = normalize_gdt(undo_formatting_rules(ground_truth), symbols_mapping)
    assert isinstance(pred_normalized, str) and isinstance(gt_normalized, str)
    if pred_normalized == gt_normalized:
        return 1.0
    return 0.0


def undo_formatting_rules(text):
    """
    Egregious: GRIB
    Undo the formatting rules from prompt4.txt while preserving Unicode characters

        note: Can't make lower case, since GD&T transform rules which are used later asume proper capitalization
    """
    if not text:
        return text

    # Get only the first line (avoiding regex for newline split)
    # this is actually cheating since I should explain rule properly?
    text = text.strip().split("\n")[0]

    # 1. Remove pipe characters (from Feature Control Frames rule)
    text = text.replace("|", "")

    # 2. Remove "All Around" text
    text = re.sub(r"all\s*around", "", text, flags=re.IGNORECASE)

    # 3. Handle datum feature format
    text = re.sub(r"datum\s*feature\s*([a-z])", r"df\1", text, flags=re.IGNORECASE)

    # 4. Simplify special text elements
    # just smushed to 0 for all this stuff if even identifies basics
    # GRIB
    # text = text.replace("representedlineelement", "rle")
    # text = re.sub(r"leaderdirectednote([a-z0-9]+)", r"ldn\1", text)
    # text = re.sub(r"crosshatchbetween([a-z0-9]+)and([a-z0-9]+)", r"ch\1\2", text)
    text = re.sub(r"represented\s*line\s*element", "rle", text, flags=re.IGNORECASE)
    text = re.sub(r"leader\s*directed\s*note\s*\w*", "ldn", text, flags=re.IGNORECASE)
    text = re.sub(r"crosshatch.*", "ch", text, flags=re.IGNORECASE)

    # 5. Remove untoleranced surfaces text
    # text = text.replace("appliestoalluntolerancedsurfaces", "")
    # grib
    text = re.sub(r"applies\s*to\s*all\s*untoleranced\s*surfaces", "aus", text, flags=re.IGNORECASE)

    # 6. Handle profile symbol conversion
    # GD&T expects Profile
    text = re.sub(r"profile\s*surface", "Profile", text, flags=re.IGNORECASE)
    text = re.sub(r"profile\s*of\s*a\s*line", "Profile", text, flags=re.IGNORECASE)
    text = re.sub(r"profile\s*of\s*a\s*surface", "Profile", text, flags=re.IGNORECASE)

    # 7. Handle numeric formatting and punctuation
    # Convert 0.XX to .XX (safer way to handle leading zeros)
    text = re.sub(r"(\D)0\.(\d+)", r"\1.\2", text)

    # 8. Remove dashes and separators - without regex that might corrupt Unicode
    for char in ["-", "/"]:
        text = text.replace(char, "")

    # 9. Remove all whitespace at the end
    text = re.sub(r"\s+", "", text)

    return text


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


if __name__ == "__main__":

    symbols_mapping = load_gdt_symbols()
    with open("data/results_printout.txt", "r") as f:


    with open("data/results_printout.txt", "r") as f:
        content = f.read()  # Read the entire file as a single string

        # Find all result pairs in the content
        results = re.findall(r"Real: `(.*?)`.*?Model: `(.*?)`", content, re.DOTALL)
        ct = 0
        for real, model in results:
            #     pass
            # if True:
            #     line = (
            #         "0.0 Real: `Profile 1.2|A|B|C`, Model: `Profile of a Surface 1.2|A|B|C\nAll Around`"
            #     )
            #     real_match = re.search(r"Real: `(.*?)`", line, re.DOTALL)
            #     model_match = re.search(r"Model: `(.*?)`", line, re.DOTALL)
            #     if real_match and model_match:
            #         real = real_match.group(1)
            #         model = model_match.group(1)
            #     print(real, model)
            #     print(list(zip(real, model)))
            #     print(evaluate_single_answer(model, real, symbols_mapping=symbols_mapping))
            #     print(evaluate_single_answer_easy(model, real, symbols_mapping=symbols_mapping))

            if not evaluate_single_answer(model, real, symbols_mapping=symbols_mapping):
                if evaluate_single_answer_easy(model, real, symbols_mapping=symbols_mapping):
                    ct += 1
                    print(f"Found a case fixed by easy evaluation:")
                    print(f"Original: Real: {real}, Model: {model}")

                    real_norm = normalize_gdt(real, symbols_mapping)
                    model_norm = normalize_gdt(model, symbols_mapping)

                    # Handle both string and list returns from normalize_gdt
                    if isinstance(real_norm, list):
                        real_undone = [undo_formatting_rules(r) for r in real_norm]
                    else:
                        real_undone = undo_formatting_rules(real_norm)

                    if isinstance(model_norm, list):
                        model_undone = [undo_formatting_rules(m) for m in model_norm]
                    else:
                        model_undone = undo_formatting_rules(model_norm)

                    print(f"After rules undone: {real_undone} vs {model_undone}")
                    print("-" * 50)
    print(ct)

#%%
# same as above but prints out header.
if __name__ == "__main__":
    symbols_mapping = load_gdt_symbols()
    with open("data/results_printout.txt", "r") as f:
        content = f.read()  # Read the entire file as a single string
        lines = content.split('\n')

        # Find all section headers (line with "python" and preceding line)
        sections = []
        current_section_start = 0
        for i, line in enumerate(lines):
            if line.startswith("python"):
                header_line = lines[i-1] if i > 0 else ""

                # Find the first line with "result" after the header
                first_result_line = None
                for j in range(i+1, len(lines)):
                    if "result" in lines[j].lower():
                        first_result_line = j
                        break

                # If no result line found, set to the end of this section
                if first_result_line is None:
                    first_result_line = i+1

                # Create section data
                section = {
                    "start_line": i-1 if i > 0 else i,
                    "header": f"{header_line}\n{line}" if header_line else line,
                    "first_result_line": first_result_line,
                    "printed": False
                }
                sections.append(section)

                # Set end line for previous section
                if len(sections) > 1:
                    sections[-2]["end_line"] = i-1 if i > 0 else i

        # Set end line for last section
        if sections:
            sections[-1]["end_line"] = len(lines)

        # Find all result pairs in the content
        results = re.findall(r"Real: `(.*?)`.*?Model: `(.*?)`", content, re.DOTALL)
        ct = 0

        # Process each result pair
        for idx, (real, model) in enumerate(results):
            if not evaluate_single_answer(model, real, symbols_mapping=symbols_mapping):
                if evaluate_single_answer_easy(model, real, symbols_mapping=symbols_mapping):
                    ct += 1

                    # Find which section this result belongs to
                    result_line = None
                    for i, line in enumerate(lines):
                        if f"Real: `{real}`" in line and f"Model: `{model}`" in line:
                            result_line = i
                            break

                    # Find and print the section header and intro if not already printed
                    if result_line is not None:
                        for section in sections:
                            if section["start_line"] <= result_line < section["end_line"] and not section["printed"]:
                                print("\n" + "=" * 80)
                                print(f"SECTION HEADER:")
                                print(section['header'])

                                # Print everything between header and first result line
                                print("\nSECTION INTRODUCTION:")
                                header_end = section["start_line"] + (2 if section["start_line"] > 0 else 1)  # Account for header being 1-2 lines
                                for i in range(header_end, section["first_result_line"]):
                                    print(lines[i])

                                print("=" * 80 + "\n")
                                section["printed"] = True
                                break

                    print(f"Found a case fixed by easy evaluation:")
                    print(f"Original: Real: {real}, Model: {model}")

                    real_norm = normalize_gdt(real, symbols_mapping)
                    model_norm = normalize_gdt(model, symbols_mapping)

                    # Handle both string and list returns from normalize_gdt
                    if isinstance(real_norm, list):
                        real_undone = [undo_formatting_rules(r) for r in real_norm]
                    else:
                        real_undone = undo_formatting_rules(real_norm)

                    if isinstance(model_norm, list):
                        model_undone = [undo_formatting_rules(m) for m in model_norm]
                    else:
                        model_undone = undo_formatting_rules(model_norm)

                    print(f"After rules undone: {real_undone} vs {model_undone}")
                    print("-" * 50)

        if ct == 0:
            print("No changes detected with the easy evaluation function.")
        else:
            print(f"Total cases fixed by easy evaluation: {ct}")