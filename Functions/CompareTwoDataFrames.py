"""
    Compare two pandas DataFrames and report differences in shape, column names, data types, and content.

    This function provides a comprehensive comparison of two DataFrames, identifying differences
    in various aspects such as shape, column names, data types, and actual data content. It's 
    designed to handle different data types, including numeric, datetime, and string columns.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame to compare.
    df2 (pd.DataFrame): The second DataFrame to compare.
    name1 (str): Name to use for the first DataFrame in the output (default: 'DataFrame1').
    name2 (str): Name to use for the second DataFrame in the output (default: 'DataFrame2').
    float_tolerance (float): Tolerance for floating point comparisons (default: 1e-6).
                             Values are considered equal if they differ by less than this amount.
    date_only (bool): If True, compare only the date part of datetime columns (default: True).
    max_samples (int): Maximum number of sample differences to show for each column (default: 10).

    Returns:
    None: This function prints its output directly and doesn't return a value.

    The function performs the following comparisons:
    1. Shape of the DataFrames
    2. Column names
    3. Data types of common columns
    4. Content of common columns

    For content comparison:
    - Numeric columns are compared using numpy's allclose function with the specified float_tolerance.
    - Datetime columns are compared either by date only or full timestamp, depending on the date_only parameter.
    - String columns are compared after stripping whitespace and handling NaN values.

    The function prints its findings, including samples of differing values up to max_samples per column.
    
    """


# Define the function to compare Datesets 

def compare_dataframes(df1, df2, name1='NameDF1', name2='NameDF2', float_tolerance=1e-6):
    print(f"Comparing {name1} and {name2}:")
    
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # 1. Compare shapes
    if df1.shape == df2.shape:
        print(f"✓ Shapes match: {df1.shape}")
    else:
        print(f"✗ Shapes don't match: {name1} {df1.shape} vs {name2} {df2.shape}")
    
    # 2. Compare column names
    common_columns = list(set(df1.columns) & set(df2.columns))
    only_in_df1 = set(df1.columns) - set(df2.columns)
    only_in_df2 = set(df2.columns) - set(df1.columns)
    
    if not only_in_df1 and not only_in_df2:
        print("✓ Column names match")
    else:
        print("✗ Column names don't match:")
        if only_in_df1:
            print(f"   Columns in {name1} but not in {name2}: {only_in_df1}")
        if only_in_df2:
            print(f"   Columns in {name2} but not in {name1}: {only_in_df2}")
    
    # 3. Compare data types for common columns
    if common_columns:
        dtypes_df1 = df1[common_columns].dtypes
        dtypes_df2 = df2[common_columns].dtypes
        dtypes_match = dtypes_df1.equals(dtypes_df2)
        if dtypes_match:
            print("✓ Data types match for common columns")
        else:
            print("✗ Data types don't match for some common columns:")
            for col in common_columns:
                if dtypes_df1[col] != dtypes_df2[col]:
                    print(f"   Column '{col}': {name1} {dtypes_df1[col]} vs {name2} {dtypes_df2[col]}")
    else:
        print("No common columns to compare data types")
    
    # 4. Compare content for common columns
    if common_columns:
        differing_columns = []
        for col in common_columns:
            series1 = df1[col]
            series2 = df2[col]
            
            if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
                if not np.allclose(series1.fillna(0), series2.fillna(0), rtol=float_tolerance, equal_nan=True):
                    differing_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(series1) and pd.api.types.is_datetime64_any_dtype(series2):
                # For datetime, compare dates and treat NaT as equal
                equal_dates = (series1.dt.date == series2.dt.date) | (series1.isna() & series2.isna())
                if not equal_dates.all():
                    differing_columns.append(col)
            else:
                if not (series1.fillna('').astype(str).str.strip() == series2.fillna('').astype(str).str.strip()).all():
                    differing_columns.append(col)
        
        if not differing_columns:
            print("✓ Data content is identical for common columns")
        else:
            print("✗ Data content differs for common columns")
            print(f"   Columns with differences: {differing_columns}")
            
            for col in differing_columns[:10]:
                print(f"\n   Sample differences in column '{col}':")
                if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                    mask = ~np.isclose(df1[col].fillna(0), df2[col].fillna(0), rtol=float_tolerance, equal_nan=True)
                elif pd.api.types.is_datetime64_any_dtype(df1[col]) and pd.api.types.is_datetime64_any_dtype(df2[col]):
                    mask = (df1[col].dt.date != df2[col].dt.date) & (~(df1[col].isna() & df2[col].isna()))
                else:
                    mask = df1[col].fillna('').astype(str).str.strip() != df2[col].fillna('').astype(str).str.strip()
                
                comparison = pd.concat([df1.loc[mask, col], df2.loc[mask, col]], axis=1, keys=[name1, name2])
                print(comparison.head().to_string())
    else:
        print("No common columns to compare content")
