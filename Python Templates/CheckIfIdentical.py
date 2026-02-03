# %%
"""
XLSX File Comparison Tool

Compares two Excel files by cell values across all sheets.
Reports exact locations where values differ.

Handles common pitfalls:
- Integer/float mismatches (2025 vs 2025.0)
- Empty cells and NaN values
- Files with different sheet names
- Files with different dimensions

Usage:
1. Set your file paths in the 'pairs' list
2. Run each cell in order
3. Review the output for differences
"""

import pandas as pd

# %%
def normalize_value(val):
    """
    Convert a cell value to a standard string format.
    Treats 2025 and 2025.0 as equal.
    """
    if val == "":
        return ""
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return str(val).strip()


def compare_xlsx(path1, path2):
    """
    Compare two xlsx files cell by cell.
    
    Returns a list of differences. Empty list means files match.
    Each difference is a dict with: sheet, row, col, file1_value, file2_value
    """
    xl1 = pd.read_excel(path1, sheet_name=None, header=None)
    xl2 = pd.read_excel(path2, sheet_name=None, header=None)
    
    all_diffs = []
    
    # Check for sheets that exist in only one file
    only_in_1 = set(xl1.keys()) - set(xl2.keys())
    only_in_2 = set(xl2.keys()) - set(xl1.keys())
    
    if only_in_1:
        all_diffs.append(f"Sheets only in file 1: {only_in_1}")
    if only_in_2:
        all_diffs.append(f"Sheets only in file 2: {only_in_2}")
    
    # Compare sheets that exist in both files
    common_sheets = set(xl1.keys()) & set(xl2.keys())
    
    for sheet in common_sheets:
        df1 = xl1[sheet].fillna("")
        df2 = xl2[sheet].fillna("")
        
        # Expand both to the same dimensions
        max_rows = max(df1.shape[0], df2.shape[0])
        max_cols = max(df1.shape[1], df2.shape[1])
        
        df1 = df1.reindex(index=range(max_rows), columns=range(max_cols)).fillna("")
        df2 = df2.reindex(index=range(max_rows), columns=range(max_cols)).fillna("")
        
        # Compare cell by cell
        for row in range(max_rows):
            for col in range(max_cols):
                val1 = normalize_value(df1.iloc[row, col])
                val2 = normalize_value(df2.iloc[row, col])
                
                if val1 != val2:
                    all_diffs.append({
                        "sheet": sheet,
                        "row": row + 1,
                        "col": col + 1,
                        "file1_value": df1.iloc[row, col],
                        "file2_value": df2.iloc[row, col]
                    })
    
    return all_diffs

# %%
# Define your file pairs here
# Each tuple: (path_to_file_1, path_to_file_2)

pairs = [
    (
        r"path\to\first_file.xlsx",
        r"path\to\second_file.xlsx"
    ),
    # Add more pairs as needed
]

# %%
# Run the comparison

for i, (p1, p2) in enumerate(pairs, 1):
    print(f"\n{'='*60}")
    print(f"PAIR {i}")
    print(f"{'='*60}")
    
    diffs = compare_xlsx(p1, p2)
    
    if not diffs:
        print("✓ Files are identical")
    else:
        print(f"✗ Found {len(diffs)} difference(s):\n")
        for d in diffs[:20]:
            if isinstance(d, str):
                print(f"  {d}")
            else:
                print(f"  Sheet: {d['sheet']}, Row: {d['row']}, Col: {d['col']}")
                print(f"    File 1: '{d['file1_value']}'")
                print(f"    File 2: '{d['file2_value']}'")
                print()
        
        if len(diffs) > 20:
            print(f"  ... and {len(diffs) - 20} more differences")
          
