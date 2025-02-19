# %%
# %%
import os

import re
import pandas as pd

import pdfplumber
import camelot
import pandas as pd
from difflib import SequenceMatcher

# A dictionary to map special GD&T or other symbols to more descriptive text.
SYMBOL_MAP = {
    "⟂": "Perpendicularity",
    "Ø": "Diameter",
    "Ⓜ️": "M (MMC)",  # e.g., symbol for Maximum Material Condition
    "⌴": "Square",  # or "Square shape" – depends on usage
    "↧": "Depth",  # or whatever the arrow symbol means in your context
    "⌳": "Taper",  # or "Taper ratio"
    "⌲": "Conicity",  # or "Conical shape"
    "SØ": "Spherical Diameter",
    # etc.
}

# The required columns from your Adobe export CSV.
TARGET_COLS = [
    "Feature ID",
    "Feature Description",
    "Specification",
    "Element ID",
    "Comments",
    "assembly id",
]

# A minimal set of headers to decide if a table is relevant.
NEED_COLS = TARGET_COLS[:-1]  # everything except "assembly id"


def normalize_text(cell_value):
    """
    Normalize extracted text by:
      1) Replacing certain symbols with dictionary values
      2) Stripping whitespace
    """
    if not isinstance(cell_value, str):
        return cell_value  # if it's not a string, return as is
    for symbol, text_replacement in SYMBOL_MAP.items():
        cell_value = cell_value.replace(symbol, text_replacement)
    return cell_value.strip()


def remove_unicode(text):
    """Remove non-ASCII characters from a string."""
    if not isinstance(text, str):
        return text
    return "".join(c for c in text if ord(c) < 128)


def remove_unicode_from_df(df):
    """Apply unicode removal element-wise in a DataFrame."""
    return df.map(remove_unicode)


def filter_and_fix_table(df):
    """
    Accept the DataFrame only if its columns exactly match TARGET_COLS.
    If "assembly id" is missing, add it (with empty strings) before checking.
    Otherwise, discard the table (return None).
    """
    # Normalize column names by stripping whitespac
    df.columns = [col.strip() if isinstance(col, str) else "" for col in df.columns]
    # Add missing "assembly id" column if necessary
    if "assembly id" not in df.columns:
        df["assembly id"] = ""
    # Now, if the set and count of columns matches TARGET_COLS, keep it.
    if set(df.columns) == set(TARGET_COLS) and len(df.columns) == len(TARGET_COLS):
        # Reorder columns exactly to TARGET_COLS
        return df[TARGET_COLS]
    else:
        return None


def extract_tables_pdfplumber(pdf_path):
    """
    Extract tables from a PDF using pdfplumber.
    Returns a list of DataFrames with the desired headers.
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                }
            )
            for table in raw_tables:
                if not table or not table[0]:
                    continue
                df = pd.DataFrame(table[1:], columns=table[0])
                # Normalize every cell in the DataFrame.
                df = df.map(normalize_text)
                fixed = filter_and_fix_table(df)
                if fixed is not None:
                    tables.append(fixed)
    return tables


def extract_tables_camelot(pdf_path, flavor="lattice"):
    """
    Extract tables from a PDF using Camelot.
    Returns a list of DataFrames with the desired headers.
    """
    tables = []
    tables_camelot = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
    for table in tables_camelot:
        df = table.df.copy()
        if df.empty or len(df) < 2:
            continue
        df.columns = df.iloc[0]
        df = df[1:]
        df = df.map(normalize_text)
        fixed = filter_and_fix_table(df)
        if fixed is not None:
            tables.append(fixed)
    return tables


def compare_dataframes(df1, df2, tolerance=1.0):
    """
    Compare two DataFrames cell-by-cell.
    Before comparing, both should have non-ASCII characters removed.
    'tolerance' is the similarity threshold (1.0 means exact match).
    This function prints out any mismatches.
    """
    rows = min(len(df1), len(df2))
    cols = min(len(df1.columns), len(df2.columns))

    for r in range(rows):
        for c in range(cols):
            val1 = str(df1.iat[r, c])
            val2 = str(df2.iat[r, c])
            ratio = SequenceMatcher(None, val1, val2).ratio()
            if ratio < tolerance:
                print(f"Mismatch at row {r}, col {df1.columns[c]}:")
                print(f"  Extracted: '{val1}'")
                print(f"  Reference: '{val2}' (Similarity: {ratio:.2f})")


if __name__ == "__main__":
    # List of PDF paths to process.
    pdf_paths = [
        "data/NIST-FTC-CTC-PMI-CAD-models/FTC Definitions/nist_ftc_06_asme1_rd_fsi.pdf",
        "data/NIST-FTC-CTC-PMI-CAD-models/FTC Definitions/nist_ftc_10_asme1_rb_fsi.pdf",
        "data/NIST-FTC-CTC-PMI-CAD-models/FTC Definitions/nist_ftc_09_asme1_rd_fsi.pdf",
        "data/NIST-FTC-CTC-PMI-CAD-models/FTC Definitions/nist_ftc_07_asme1_rd_fsi.pdf",
        "data/NIST-FTC-CTC-PMI-CAD-models/FTC Definitions/nist_ftc_08_asme1_rc1_fsi.pdf",
        "data/NIST-FTC-CTC-PMI-CAD-models/FTC Definitions/nist_ftc_11_asme1_rb_fsi.pdf",
    ]

    all_plumber = []
    all_camelot = []
    # Extract from each PDF using both methods.
    for path in pdf_paths:
        m = re.search(r"_(\d{2})_", path)
        assembly_id = m.group(1) if m else None
        plumber_df = pd.concat(extract_tables_pdfplumber(path))
        camelot_df = pd.concat(extract_tables_camelot(path, flavor="lattice"))
        print(path)
        compare_dataframes(plumber_df, camelot_df, tolerance=0.95)
        all_plumber += [plumber_df]
        all_camelot += [camelot_df]

    # %%
    # Load the reference Adobe-exported CSV.
    adobe_csv = "data/fsi_labels/adobe_exported_fsi.csv"
    adobe_df = pd.read_csv(adobe_csv)
    for all_tables in [all_camelot]:  # , all_plumber]:
        merged_df = pd.concat(all_tables, ignore_index=True)
        merged_csv = "data/merged_extracted.csv"
        merged_df.to_csv(merged_csv, index=False, encoding="utf-8")
        print(f"Merged {len(all_tables)} tables into {merged_csv}")

        # Remove non-ASCII characters from both DataFrames for a fair comparison.
        merged_clean = remove_unicode_from_df(merged_df)
        adobe_clean = remove_unicode_from_df(adobe_df)

        # Verification 1: Check dimensions.
        if merged_clean.shape != adobe_clean.shape:
            print("Dimension mismatch:")
            print("  Extracted:", merged_clean.shape)
            print("  Adobe:", adobe_clean.shape)
        else:
            print("Dimensions match. Proceeding with cell-by-cell comparison...")

        # Verification 2: Compare cell-by-cell.
        print("\nComparing merged extracted data to Adobe export (ignoring unicode differences):")
        compare_dataframes(merged_clean, adobe_clean, tolerance=1)
