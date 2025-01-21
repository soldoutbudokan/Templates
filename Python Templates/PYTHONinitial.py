# %%
################################################################################
# Title: [Analysis Name]
# Author: Tirth Bhatt
# Date Created: [Date]
# Last Modified: [Date]
#
# Project: [Project]
# 
# Notes:
# - Key assumptions
# - Data sources
# - Dependencies
################################################################################

# %%
# Initial Setup
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Define paths
paths = {
    "project": "F:/Proj/DXXXXX-00 - XXXXXX",
    "analysis": "Analysis",
    "workstream": "Workstream"
}

# Construct base directory
base_dir = Path(paths["project"]) / paths["analysis"] / paths["workstream"]

# Define working directories
dirs = {
    "raw": base_dir / "raw",
    "intermediate": base_dir / "intermediate",
    "output": base_dir / "output",
    "code": base_dir / "code"
}

# Create directories if they don't exist
for dir_path in dirs.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# %%
# Data loading functions
def load_raw_data(filename: str) -> pd.DataFrame:
    """Load and perform initial cleaning of raw data"""
    try:
        df = pd.read_csv(dirs["raw"] / filename)
        print(f"Successfully loaded {filename}")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return None

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main data processing function"""
    if df is None:
        return None
    
    # Basic cleaning
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.drop_duplicates()
    df = df.dropna(subset=['key_variable'])
    
    return df

# %%
# Load and process data
raw_data = load_raw_data("data.csv")
cleaned_data = process_data(raw_data)

# Basic data inspection
print("\nData Summary:")
print(cleaned_data.describe())
print("\nMissing Values:")
print(cleaned_data.isnull().sum())

# %%
# Analysis functions
def run_regression(df: pd.DataFrame, y_var: str, x_vars: list) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run OLS regression with robust standard errors"""
    X = sm.add_constant(df[x_vars])
    y = df[y_var]
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC1')
    
    return results

def create_summary_stats(df: pd.DataFrame, vars_of_interest: list) -> pd.DataFrame:
    """Create summary statistics table"""
    summary = df[vars_of_interest].describe()
    summary.loc['N'] = df[vars_of_interest].count()
    
    return summary

# %%
# Main analysis
if cleaned_data is not None:
    # Run regression
    y_variable = 'dependent_var'
    x_variables = ['independent_var1', 'independent_var2']
    
    reg_results = run_regression(cleaned_data, y_variable, x_variables)
    print("\nRegression Results:")
    print(reg_results.summary())
    
    # Create summary statistics
    summary_vars = ['var1', 'var2', 'var3']
    summary_stats = create_summary_stats(cleaned_data, summary_vars)

# %%
# Create visualizations
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cleaned_data, x='independent_var1', y='dependent_var')
plt.title('Relationship between Variables')
plt.savefig(dirs["output"] / 'scatter_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# %%
# Save outputs
if cleaned_data is not None:
    # Save intermediate data
    cleaned_data.to_csv(dirs["intermediate"] / "cleaned_data.csv", index=False)
    cleaned_data.to_stata(dirs["intermediate"] / "cleaned_data.dta")  # Optional Stata format
    
    # Save analysis outputs
    summary_stats.to_csv(dirs["output"] / "summary_statistics.csv", index=True)
    with open(dirs["output"] / "regression_results.txt", 'w') as f:
        f.write(reg_results.summary().as_text())
    
    # Save final dataset
    final_data = cleaned_data.copy()
    final_data['date_created'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    final_data.to_csv(dirs["output"] / "final_results.csv", index=False)
    
    print("\nFiles saved:")
    print(f"Intermediate data: {dirs['intermediate']/'cleaned_data.csv'}")
    print(f"Final results: {dirs['output']/'final_results.csv'}")