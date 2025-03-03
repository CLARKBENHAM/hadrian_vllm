#!/usr/bin/env python
# aggregate_results.py

# Lots of mistakes happen in same place
# some is dumb answer naming (eg. dataum )
# including too much data
# getting wrong symbol

import os
import glob
import pandas as pd
import argparse
import datetime
import time
from collections import defaultdict

# Import evaluation functionality from existing code
from hadrian_vllm.evaluation import evaluate_answer, load_gdt_symbols
from hadrian_vllm.utils import load_csv


def aggregate_results(
    base_csv_path, results_dir="data/results", threshold=0.5, hours=None, verbose=False
):
    """
    Analyze results across multiple prediction files to find consistently difficult rows.

    Args:
        base_csv_path: Path to the original CSV with ground truth
        results_dir: Directory containing result CSV files
        threshold: Fraction of correct predictions required to consider a row "correct" in aggregate
        hours: If provided, only consider files modified within the past X hours
        verbose: Whether to print detailed information

    Returns:
        DataFrame with aggregated results and statistics
    """
    print(f"Loading base CSV from {base_csv_path}")
    base_df = load_csv(base_csv_path)

    # Get list of result files
    result_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")

    # Filter files by modification time if hours parameter is provided
    if hours is not None:
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)  # Convert hours to seconds

        filtered_files = []
        for file_path in result_files:
            mod_time = os.path.getmtime(file_path)
            if mod_time >= cutoff_time:
                filtered_files.append(file_path)

        result_files = filtered_files
        print(f"Found {len(result_files)} result files modified in the past {hours} hours")
    else:
        print(f"Found {len(result_files)} result files")

    # Create index for faster lookups in base_df
    base_df_index = {}
    for _, row in base_df.iterrows():
        if (
            pd.notna(row["Assembly ID"])
            and pd.notna(row["Page ID"])
            and pd.notna(row["Element ID"])
        ):
            key = f"{row['Assembly ID']}_{row['Page ID']}_{row['Element ID']}"
            base_df_index[key] = row["Specification"]

    # Dictionary to store results for each row
    all_results = defaultdict(
        lambda: {
            "correct": 0,
            "incorrect": 0,
            "unanswered": 0,
            "total": 0,
            "predictions": {},  # Store each model's prediction
        }
    )

    # Process each result file
    for result_file in result_files:
        print(f"Processing {os.path.basename(result_file)}")
        result_df = pd.read_csv(result_file)

        # Find the result column (should be the last one)
        result_columns = [col for col in result_df.columns if col.startswith("Specification ")]
        if not result_columns:
            print(f"Warning: No result columns found in {result_file}, skipping")
            continue

        result_column = result_columns[0]
        print(f"  Using result column: {result_column}")

        # Process each row
        for _, row in result_df.iterrows():
            # Create a unique key for this row
            if pd.isna(row["Assembly ID"]) or pd.isna(row["Page ID"]) or pd.isna(row["Element ID"]):
                continue

            row_key = f"{row['Assembly ID']}_{row['Page ID']}_{row['Element ID']}"

            # Get ground truth from base_df, not from result_df
            ground_truth = base_df_index.get(row_key)
            if ground_truth is None or pd.isna(ground_truth) or ground_truth == "":
                # Skip rows without ground truth
                continue

            # Get the prediction
            prediction = row.get(result_column)

            # Store prediction for this model
            model_name = os.path.basename(result_file).replace(".csv", "")
            all_results[row_key]["predictions"][model_name] = prediction

            # Evaluate the prediction
            if pd.isna(prediction) or prediction == "":
                all_results[row_key]["unanswered"] += 1
            else:
                try:
                    # Use the existing evaluation function
                    if evaluate_answer(prediction, ground_truth) == 1.0:
                        all_results[row_key]["correct"] += 1
                    else:
                        all_results[row_key]["incorrect"] += 1
                except Exception as e:
                    print(f"Error evaluating answer for {row_key}: {e}")
                    all_results[row_key]["incorrect"] += 1

            all_results[row_key]["total"] += 1

            # Store ground truth for later reference
            if "ground_truth" not in all_results[row_key]:
                all_results[row_key]["ground_truth"] = ground_truth
                all_results[row_key]["assembly_id"] = row["Assembly ID"]
                all_results[row_key]["page_id"] = row["Page ID"]
                all_results[row_key]["element_id"] = row["Element ID"]

    # Calculate statistics for each row
    results_list = []
    for row_key, stats in all_results.items():
        if stats["total"] == 0:
            continue

        # Calculate fraction of predictions that were correct
        attempted = stats["correct"] + stats["incorrect"]
        correct_fraction = stats["correct"] / attempted if attempted > 0 else 0

        # Mark as difficult if below threshold
        is_difficult = correct_fraction < threshold if attempted > 0 else None

        results_list.append(
            {
                "Assembly ID": stats["assembly_id"],
                "Page ID": stats["page_id"],
                "Element ID": stats["element_id"],
                "Ground Truth": stats["ground_truth"],
                "Total Evaluations": stats["total"],
                "Correct": stats["correct"],
                "Incorrect": stats["incorrect"],
                "Unanswered": stats["unanswered"],
                "Attempted": attempted,
                "Correct Fraction": correct_fraction,
                "Is Difficult": is_difficult,
                "Predictions": stats["predictions"],
            }
        )

    # Create DataFrame from results
    results_df = pd.DataFrame(results_list)

    # Sort by correct_fraction ascending (most difficult first)
    results_df = results_df.sort_values("Correct Fraction")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate results from multiple prediction files")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
        help="Path to the original CSV with ground truth",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/results",
        help="Directory containing result CSV files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fraction of correct predictions required to consider a row 'correct' in aggregate",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Only consider files modified within the past X hours",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/aggregate_results.csv",
        help="Path to save the aggregated results",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument(
        "--print_difficult", action="store_true", help="Print difficult rows to console"
    )
    parser.add_argument("--eval-easy", action="store_true", help="Use easy evaluation mode")
    args = parser.parse_args()

    # Run the aggregation
    results_df = aggregate_results(
        args.csv, args.results_dir, args.threshold, args.hours, args.verbose
    )

    # Save the results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"Saved aggregated results to {args.output}")

    # Print summary statistics
    total_rows = len(results_df)
    difficult_rows = len(results_df[results_df["Is Difficult"] == True])
    easy_rows = len(results_df[results_df["Is Difficult"] == False])
    unattempted_rows = len(results_df[results_df["Is Difficult"].isna()])

    print("\nSummary:")
    print(f"Total rows: {total_rows}")
    print(
        f"Difficult rows (below {args.threshold} correct fraction):"
        f" {difficult_rows} ({difficult_rows/total_rows:.1%})"
    )
    print(
        f"Easy rows (above {args.threshold} correct fraction):"
        f" {easy_rows} ({easy_rows/total_rows:.1%})"
    )
    print(f"Unattempted rows: {unattempted_rows} ({unattempted_rows/total_rows:.1%})")

    # Print difficult rows if requested
    if args.print_difficult and difficult_rows > 0:
        difficult_df = results_df[results_df["Is Difficult"] == True].head(50)  # limited
        print(f"\{len(difficult_df)} Difficult rows:")
        for _, row in difficult_df.iterrows():
            print(
                f"Assembly: {row['Assembly ID']}, Page: {row['Page ID']}, Element:"
                f" {row['Element ID']}"
            )
            print(f"Ground Truth: {row['Ground Truth']}")
            print(f"Correct: {row['Correct']}/{row['Attempted']} ({row['Correct Fraction']:.1%})")

            # Print individual model predictions
            print("\nModel predictions:")
            for model, prediction in row["Predictions"].items():
                prediction_str = str(prediction) if not pd.isna(prediction) else "UNANSWERED"
                is_correct = (
                    evaluate_answer(prediction, row["Ground Truth"]) == 1.0
                    if not pd.isna(prediction)
                    else False
                )
                status = "✓" if is_correct else "✗" if not pd.isna(prediction) else "-"
                print(f"  {status} {model}: {prediction_str}")

            print("-" * 80)


if __name__ == "__main__":
    main()

# python scripts/aggr_multiple_evaluations.py  --threshold 0.01 --eval-easy
# python scripts/aggr_multiple_evaluations.py  --threshold 0.01 --eval-easy --print_difficult --hours 20
