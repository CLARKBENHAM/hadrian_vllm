# utils.py
import os
import pandas as pd
import re
import json
import subprocess
from datetime import datetime

import sys


def basic_numeric_hash(input_string, prime=31, modulus=2**32):
    hash_value = 0
    for char in input_string:
        hash_value = (hash_value * prime + ord(char)) % modulus
    return hash_value


def string_to_unicode_hex(input_string):
    return basic_numeric_hash([ord(char) for char in input_string])


def get_git_hash():
    """Get the current git commit hash"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown_git_hash"


def get_current_datetime():
    """Get the current date and time in a readable format"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def is_debug_mode():
    return sys.gettrace() is not None


def load_csv(csv_path):
    """Load the GD&T data CSV."""
    return pd.read_csv(csv_path)


def extract_assembly_and_page_from_filename(filename):
    """Extract assembly ID and page number from filename."""
    filename = os.path.basename(filename)  # ensure just name not path

    assembly_match = re.search(r"nist_ftc_(\d+)", filename)
    page_match = re.search(r"_pg(\d+)", filename)

    if not assembly_match or not page_match:
        # {assembly}_{page}_q{quarter}
        pattern = r"^(\d+)_(\d+)_q(\d+)\.png$"
        match = re.match(pattern, filename)
        if match:
            assembly_id, page_id, quarter = map(int, match.groups())
            return assembly_id, page_id
        return None, None

    assembly_id = int(assembly_match.group(1))
    page_id = int(page_match.group(1))

    return assembly_id, page_id


def save_df_with_results(df, result_column, output_path="data/results/results.csv"):
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
