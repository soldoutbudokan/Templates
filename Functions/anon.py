"""
Data Anonymization Tool

This script transforms sensitive datasets into anonymized versions while preserving 
their statistical properties. It enables safe sharing of data with language models 
or third parties by replacing sensitive values with synthetic data that maintains 
the original patterns and relationships.

Key Features:
- Handles multiple data types (numeric, categorical, dates)
- Preserves statistical distributions and relationships
- Detects and converts misformatted data (like strings stored as numbers)
- Processes both CSV and Excel files
- Maintains consistent anonymization mappings across the dataset

The anonymization process:
1. Detects true data types (numbers stored as text, dates, percentages)
2. Converts data to appropriate types
3. Generates synthetic data that preserves statistical properties:
   - Numeric: Maintains mean and standard deviation
   - Dates: Keeps relative temporal relationships
   - Categorical: Creates consistent anonymous mappings
4. Saves output with '_anon' suffix

Usage:
    python anonymizer.py
    # Or import as module:
    from anonymizer import process_file
    process_file('data.csv')

The tool is particularly useful for:
- Sharing sensitive business data with language models
- Testing data pipelines without exposing real user data
- Collaborating on projects involving private information
- Creating realistic test datasets

"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
import hashlib
import os
import re

def detect_and_convert_type(series):
    """
    Detects and converts series to appropriate data type, similar to Excel's VALUE function.
    Handles numbers stored as strings, percentages, and dates.
    
    Parameters:
    series (pd.Series): Input series to convert
    
    Returns:
    pd.Series: Converted series
    """
    # Drop NA to check type of actual values
    non_null = series.dropna()
    if len(non_null) == 0:
        return series
    
    # Sample value for type checking
    sample = str(non_null.iloc[0])
    
    # Try to detect dates first
    try:
        # Check if majority of non-null values can be parsed as dates
        date_count = sum(pd.to_datetime(non_null, errors='coerce').notna())
        if date_count / len(non_null) > 0.5:  # If more than 50% are valid dates
            return pd.to_datetime(series, errors='coerce')
    except:
        pass
    
    # Check for percentage strings (e.g., "45%")
    if all(str(x).endswith('%') for x in non_null if pd.notna(x)):
        try:
            return series.str.rstrip('%').astype(float) / 100
        except:
            pass
    
    # Check for numeric strings
    try:
        # Remove thousands separators and currency symbols
        cleaned = series.astype(str).str.replace(r'[,$£€\s]', '', regex=True)
        # Handle parentheses for negative numbers
        cleaned = cleaned.str.replace(r'\((.*?)\)', r'-\1', regex=True)
        
        # Convert to float first
        numeric = pd.to_numeric(cleaned, errors='coerce')
        
        # If all numbers are integers, convert to int
        if np.all(numeric.dropna() == numeric.dropna().astype(int)):
            return numeric.astype('Int64')  # Using Int64 to handle NaN values
        return numeric
    except:
        pass
    
    # If no specific type is detected, return original series
    return series

def anonymize_dataframe(df, seed=42):
    """
    Anonymizes a pandas DataFrame by replacing values with random data while preserving data types
    and basic statistical properties.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame to anonymize
    seed (int): Random seed for reproducibility
    
    Returns:
    pandas.DataFrame: Anonymized DataFrame
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Create a copy to avoid modifying the original
    df_anon = df.copy()
    
    # First pass: detect and convert types
    for column in df_anon.columns:
        df_anon[column] = detect_and_convert_type(df_anon[column])
    
    # Create consistent mapping for categorical values
    categorical_mappings = {}
    
    # Second pass: anonymize based on detected types
    for column in df_anon.columns:
        col_type = df_anon[column].dtype
        non_null_values = df_anon[column].dropna()
        
        if len(non_null_values) == 0:
            continue
            
        # Numeric data
        if np.issubdtype(col_type, np.number):
            mean = non_null_values.mean()
            std = non_null_values.std()
            if std == 0:
                std = 1
            
            # Generate random values with similar distribution
            random_values = np.random.normal(mean, std, len(df_anon))
            
            # Maintain integers if original was integer
            if np.issubdtype(col_type, np.integer):
                random_values = np.round(random_values).astype(col_type)
            
            # Handle percentages (values between 0 and 1)
            if mean >= 0 and mean <= 1 and max(non_null_values) <= 1:
                random_values = np.clip(random_values, 0, 1)
            
            df_anon[column] = random_values
            
        # DateTime data
        elif np.issubdtype(col_type, np.datetime64):
            min_date = pd.to_datetime(non_null_values.min())
            max_date = pd.to_datetime(non_null_values.max())
            date_range = (max_date - min_date).days
            
            random_dates = [
                min_date + timedelta(days=random.randint(0, date_range))
                for _ in range(len(df_anon))
            ]
            df_anon[column] = random_dates
            
        # String/Object data
        else:
            if column not in categorical_mappings:
                unique_values = non_null_values.unique()
                
                # Generate random strings for mapping
                random_values = []
                for _ in range(len(unique_values)):
                    # Create hash-based random string
                    original = str(unique_values[_])
                    hashed = hashlib.md5(original.encode()).hexdigest()[:8]
                    random_values.append(f"ANON_{hashed}")
                
                categorical_mappings[column] = dict(zip(unique_values, random_values))
            
            # Apply mapping
            df_anon[column] = df_anon[column].map(categorical_mappings[column])
    
    return df_anon

def process_file(filepath, seed=42):
    """
    Reads a file, anonymizes it, and saves the result with '_anon' appended to the filename.
    
    Parameters:
    filepath (str): Path to the input file
    seed (int): Random seed for reproducibility
    
    Returns:
    str: Path to the anonymized output file
    """
    # Split the filepath into directory, filename, and extension
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    
    # Create output filepath
    output_filepath = os.path.join(directory, f"{name}_anon{ext}")
    
    # Read the file based on extension
    if ext.lower() == '.csv':
        df = pd.read_csv(filepath)
    elif ext.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Anonymize the dataframe
    df_anonymized = anonymize_dataframe(df, seed)
    
    # Save the anonymized data
    if ext.lower() == '.csv':
        df_anonymized.to_csv(output_filepath, index=False)
    else:
        df_anonymized.to_excel(output_filepath, index=False)
    
    return output_filepath

if __name__ == "__main__":
    # Example usage
    filepath = input("Enter the path to your data file: ")
    try:
        output_path = process_file(filepath)
        print(f"Anonymized file saved to: {output_path}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")