# utils.py
import os
import pandas as pd
import re
import json
import subprocess
from datetime import datetime


def string_to_unicode_hex(input_string):
    return "_".join([f"{ord(char):04x}" for char in input_string])


def get_git_hash():
    """Get the current git commit hash"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown_git_hash"


def get_current_datetime():
    """Get the current date and time in a readable format"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def load_csv(csv_path):
    """Load the GD&T data CSV."""
    return pd.read_csv(csv_path)


def extract_assembly_and_page_from_filename(filename):
    """Extract assembly ID and page number from filename."""
    assembly_match = re.search(r"nist_ftc_(\d+)", filename)
    page_match = re.search(r"_pg(\d+)", filename)

    if not assembly_match or not page_match:
        return None, None

    assembly_id = int(assembly_match.group(1))
    page_id = int(page_match.group(1))

    return assembly_id, page_id


def save_df_with_results(df, result_column, output_path="data/results.csv"):
    """Save DataFrame with results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def get_element_details(csv_path, element_id, assembly_id=None, page_id=None):
    """Get details for a specific element ID from the CSV."""
    df = pd.read_csv(csv_path)

    query = df["Element ID"] == element_id
    if assembly_id is not None:
        query = query & (df["Assembly ID"] == assembly_id)
    if page_id is not None:
        query = query & (df["Page ID"] == page_id)

    element_row = df[query]

    if element_row.empty:
        return None

    return element_row.iloc[0].to_dict()
