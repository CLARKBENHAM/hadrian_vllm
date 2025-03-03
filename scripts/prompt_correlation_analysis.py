# %%
#!/usr/bin/env python
# correlation_analysis.py
# See how correlated various runs are, r=0.8
# so changing prompt doesn't matter that much
import os
import glob
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools

# Import evaluation functionality from existing code
from hadrian_vllm.evaluation import evaluate_answer, load_gdt_symbols
from hadrian_vllm.utils import load_csv


def analyze_model_correlations(
    base_csv_path,
    results_dir="data/results",
    hours=None,
    output_prefix="data/correlation",
    plot=True,
):
    """
    Analyze correlations between different model prediction runs.

    Args:
        base_csv_path: Path to the original CSV with ground truth
        results_dir: Directory containing result CSV files
        hours: If provided, only consider files modified within the past X hours
        output_prefix: Prefix for output files
        plot: Whether to generate correlation plots

    Returns:
        Tuple of (correlation_df, prediction_matrix_df)
    """
    print(f"Loading base CSV from {base_csv_path}")
    base_df = load_csv(base_csv_path)

    # Get list of result files
    result_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")

    # Filter files by modification time if hours parameter is provided
    if hours is not None:
        import time

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

    # Create a dictionary to store binary correctness for each model and row
    models = []
    correctness_by_model = {}
    row_keys = set()

    # First pass: collect model names and row keys
    for result_file in result_files:
        model_name = os.path.basename(result_file).replace(".csv", "")
        models.append(model_name)
        correctness_by_model[model_name] = {}

        print(f"Processing {model_name}")
        result_df = pd.read_csv(result_file)

        # Find the result column
        result_columns = [col for col in result_df.columns if col.startswith("Specification ")]
        if not result_columns:
            print(f"Warning: No result columns found in {result_file}, skipping")
            continue

        result_column = result_columns[0]

        # Process each row
        for _, row in result_df.iterrows():
            if pd.isna(row["Assembly ID"]) or pd.isna(row["Page ID"]) or pd.isna(row["Element ID"]):
                continue

            row_key = f"{row['Assembly ID']}_{row['Page ID']}_{row['Element ID']}"
            row_keys.add(row_key)

            # Get ground truth from base_df
            ground_truth = base_df_index.get(row_key)
            if ground_truth is None or pd.isna(ground_truth) or ground_truth == "":
                continue

            # Get the prediction
            prediction = row.get(result_column)

            # Evaluate the prediction
            if pd.isna(prediction) or prediction == "":
                correctness_by_model[model_name][row_key] = np.nan
            else:
                try:
                    is_correct = evaluate_answer(prediction, ground_truth) == 1.0
                    correctness_by_model[model_name][row_key] = 1.0 if is_correct else 0.0
                except Exception as e:
                    print(f"Error evaluating answer for {row_key}: {e}")
                    correctness_by_model[model_name][row_key] = 0.0

    # Create a DataFrame with binary correctness values
    prediction_matrix = []

    for row_key in sorted(row_keys):
        row_data = {"row_key": row_key}
        for model in models:
            row_data[model] = correctness_by_model[model].get(row_key, np.nan)
        prediction_matrix.append(row_data)

    prediction_matrix_df = pd.DataFrame(prediction_matrix)

    # Calculate correlation matrix (pairwise for models)
    model_columns = prediction_matrix_df.columns[1:]  # Skip 'row_key'
    correlation_methods = {
        "pearson": lambda x, y: stats.pearsonr(x, y)[0],
        "spearman": lambda x, y: stats.spearmanr(x, y, nan_policy="omit")[0],
        "phi": lambda x, y: calculate_phi_coefficient(x, y),
    }

    correlation_results = {}

    for method_name, correlation_func in correlation_methods.items():
        print(f"Calculating {method_name} correlations...")
        correlation_matrix = np.zeros((len(model_columns), len(model_columns)))
        p_values_matrix = np.zeros((len(model_columns), len(model_columns)))

        for i, model1 in enumerate(model_columns):
            for j, model2 in enumerate(model_columns):
                if i == j:
                    # Self-correlation is always 1
                    correlation_matrix[i, j] = 1.0
                    p_values_matrix[i, j] = 0.0
                else:
                    # Drop rows where either model has NaN
                    valid_data = prediction_matrix_df[[model1, model2]].dropna()

                    if len(valid_data) > 1:
                        try:
                            if method_name == "phi":
                                correlation_matrix[i, j] = correlation_func(
                                    valid_data[model1], valid_data[model2]
                                )
                                # For phi coefficient, we don't calculate p-values
                                p_values_matrix[i, j] = np.nan
                            else:
                                corr, p_value = (
                                    stats.pearsonr(valid_data[model1], valid_data[model2])
                                    if method_name == "pearson"
                                    else stats.spearmanr(valid_data[model1], valid_data[model2])
                                )
                                correlation_matrix[i, j] = corr
                                p_values_matrix[i, j] = p_value
                        except Exception as e:
                            print(f"Error calculating {method_name} correlation: {e}")
                            correlation_matrix[i, j] = np.nan
                            p_values_matrix[i, j] = np.nan
                    else:
                        correlation_matrix[i, j] = np.nan
                        p_values_matrix[i, j] = np.nan

        correlation_df = pd.DataFrame(
            correlation_matrix, index=model_columns, columns=model_columns
        )
        p_values_df = pd.DataFrame(p_values_matrix, index=model_columns, columns=model_columns)

        correlation_results[method_name] = {"correlation": correlation_df, "p_values": p_values_df}

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # Save the prediction matrix
    prediction_matrix_df.to_csv(f"{output_prefix}_predictions.csv", index=False)
    print(f"Saved prediction matrix to {output_prefix}_predictions.csv")

    # Calculate and save pairwise success rates
    pairwise_success = analyze_pairwise_success(prediction_matrix_df, models)
    pairwise_success.to_csv(f"{output_prefix}_pairwise_success.csv")
    print(f"Saved pairwise success analysis to {output_prefix}_pairwise_success.csv")

    # Generate plots if requested
    if plot:
        for method_name, result in correlation_results.items():
            correlation_df = result["correlation"]

            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_df, dtype=bool))
            sns.heatmap(
                correlation_df,
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                mask=mask,
                square=True,
                fmt=".2f",
            )
            plt.title(f"{method_name.capitalize()} Correlation Between Models")
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_{method_name}_correlation.png", dpi=300)
            print(
                f"Saved {method_name} correlation plot to"
                f" {output_prefix}_{method_name}_correlation.png"
            )
            plt.close()

            # Save correlation matrices
            correlation_df.to_csv(f"{output_prefix}_{method_name}_correlation.csv")
            print(
                f"Saved {method_name} correlation matrix to"
                f" {output_prefix}_{method_name}_correlation.csv"
            )

            if method_name != "phi":  # We don't have p-values for phi
                p_values_df = result["p_values"]
                p_values_df.to_csv(f"{output_prefix}_{method_name}_pvalues.csv")

    # Return the correlation DataFrame for the chosen method
    return correlation_results, prediction_matrix_df


def calculate_phi_coefficient(x, y):
    """
    Calculate the phi coefficient (Matthews correlation coefficient) for binary data.
    This is especially appropriate for binary prediction outcomes.
    """
    # Create contingency table
    n_11 = np.sum((x == 1) & (y == 1))  # Both correct
    n_10 = np.sum((x == 1) & (y == 0))  # x correct, y incorrect
    n_01 = np.sum((x == 0) & (y == 1))  # x incorrect, y correct
    n_00 = np.sum((x == 0) & (y == 0))  # Both incorrect

    # Calculate phi coefficient
    if np.any([n_11 + n_10, n_11 + n_01, n_00 + n_10, n_00 + n_01]) == 0:
        return np.nan

    denominator = np.sqrt((n_11 + n_10) * (n_11 + n_01) * (n_00 + n_10) * (n_00 + n_01))
    if denominator == 0:
        return np.nan

    phi = (n_11 * n_00 - n_10 * n_01) / denominator
    return phi


def analyze_pairwise_success(prediction_matrix_df, models):
    """
    Analyze how often pairs of models succeed or fail together.

    Returns a DataFrame with counts of:
    - both_correct: Both models correct
    - both_incorrect: Both models incorrect
    - only_model1: Only the first model correct
    - only_model2: Only the second model correct
    - agreement: Both models agree (both correct or both incorrect)
    - disagreement: Models disagree
    - conditional_prob_correct: Probability model2 is correct given model1 is correct
    """
    model_pairs = list(itertools.combinations(models, 2))

    results = []
    for model1, model2 in model_pairs:
        # Get only rows where both models made predictions
        valid_data = prediction_matrix_df[[model1, model2]].dropna()

        if len(valid_data) > 0:
            both_correct = sum((valid_data[model1] == 1) & (valid_data[model2] == 1))
            both_incorrect = sum((valid_data[model1] == 0) & (valid_data[model2] == 0))
            only_model1 = sum((valid_data[model1] == 1) & (valid_data[model2] == 0))
            only_model2 = sum((valid_data[model1] == 0) & (valid_data[model2] == 1))

            agreement = both_correct + both_incorrect
            disagreement = only_model1 + only_model2

            # Conditional probability: P(model2 correct | model1 correct)
            model1_correct_count = both_correct + only_model1
            if model1_correct_count > 0:
                conditional_prob = both_correct / model1_correct_count
            else:
                conditional_prob = np.nan

            results.append(
                {
                    "model1": model1,
                    "model2": model2,
                    "both_correct": both_correct,
                    "both_incorrect": both_incorrect,
                    "only_model1": only_model1,
                    "only_model2": only_model2,
                    "agreement": agreement,
                    "disagreement": disagreement,
                    "agreement_rate": agreement / len(valid_data),
                    "conditional_prob_correct": conditional_prob,
                    "total_compared": len(valid_data),
                }
            )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Analyze correlation between model predictions")
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
        "--hours",
        type=float,
        default=None,
        help="Only consider files modified within the past X hours",
    )
    parser.add_argument(
        "--output", type=str, default="data/correlation", help="Prefix for output files"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating correlation plots")
    args = parser.parse_args()

    # Run the correlation analysis
    correlation_results, prediction_matrix = analyze_model_correlations(
        args.csv, args.results_dir, args.hours, args.output, not args.no_plot
    )

    # Print some high-level summary
    for method, result in correlation_results.items():
        correlation_df = result["correlation"]
        # Calculate average correlation (excluding self-correlations)
        mask = ~np.eye(correlation_df.shape[0], dtype=bool)
        avg_correlation = correlation_df.values[mask].mean()

        # Find highest and lowest correlations
        np.fill_diagonal(correlation_df.values, np.nan)
        max_corr = correlation_df.max().max()
        max_pair = np.where(correlation_df.values == max_corr)
        if len(max_pair[0]) > 0:
            max_model1 = correlation_df.index[max_pair[0][0]]
            max_model2 = correlation_df.columns[max_pair[1][0]]
        else:
            max_model1, max_model2 = "Unknown", "Unknown"

        min_corr = correlation_df.min().min()
        min_pair = np.where(correlation_df.values == min_corr)
        if len(min_pair[0]) > 0:
            min_model1 = correlation_df.index[min_pair[0][0]]
            min_model2 = correlation_df.columns[min_pair[1][0]]
        else:
            min_model1, min_model2 = "Unknown", "Unknown"

        print(f"\n{method.capitalize()} Correlation Summary:")
        print(f"Average correlation: {avg_correlation:.3f}")
        print(f"Highest correlation: {max_corr:.3f} between {max_model1} and {max_model2}")
        print(f"Lowest correlation: {min_corr:.3f} between {min_model1} and {min_model2}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
