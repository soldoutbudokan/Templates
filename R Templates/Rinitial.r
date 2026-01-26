# Load required packages
################################################################################
# Title: [Analysis Name]
# Author: soldoutbudokan
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

# Install pacman if not already installed
if (!require("pacman")) install.packages("pacman")

# Load (and install if necessary) required packages
pacman::p_load(
  tidyverse,
  ggplot2,
  lubridate,     # Date handling
  janitor,       # Clean column names
  readxl,
  tidytext
)

# Set core project paths
paths <- list(
  project = "F:/Proj/DXXXXX-00 - XXXXXX",
  analysis = "Analysis",
  workstream = "Workstream"
)

# Construct base directory
base_dir <- file.path(
  paths$project,
  paths$analysis,
  paths$workstream
)

# Define working directories
dirs <- list(
  raw = file.path(base_dir, "raw"),
  intermediate = file.path(base_dir, "intermediate"),
  output = file.path(base_dir, "output"),
  code = file.path(base_dir, "code")
)

# Create directories if needed
for (dir in dirs) {
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
}
dir_create <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

paths <- list(
  raw = file.path(base_dir, "raw"),
  intermediate = file.path(base_dir, "intermediate"),
  output = file.path(base_dir, "output")
)

walk(paths, dir_create)

# Load helper functions
source("functions/data_cleaning.R")

# Load and process data
raw_data <- read_csv(file.path(paths$raw, "data.csv")) %>%
  janitor::clean_names()

# Basic data cleaning
cleaned_data <- raw_data %>%
  # Remove duplicates
  distinct() %>%
  # Handle missing values
  drop_na(key_variable) %>%
  # Create new variables
  mutate(
    log_income = log(income),
    treatment = factor(treatment)
  )

# Summary statistics
summary_stats <- cleaned_data %>%
  select(income, education, age) %>%
  datasummary_skim(output = "latex")

# Save summary stats
write_latex(summary_stats, 
           file.path(paths$output, "summary_statistics.tex"))

# Run regressions
reg_models <- list(
  "Base" = feols(y ~ x1 | fe1, data = cleaned_data),
  "Controls" = feols(y ~ x1 + x2 + x3 | fe1 + fe2, 
                    cluster = "group_id",
                    data = cleaned_data)
)

# Create regression table
modelsummary(reg_models,
            output = file.path(paths$output, "regression_results.tex"),
            stars = TRUE,
            gof_map = c("nobs", "r.squared"))

# Create visualizations
treatment_plot <- ggplot(cleaned_data, 
                        aes(x = time, y = outcome, color = treatment)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Treatment Effects Over Time",
       x = "Time Period",
       y = "Outcome")

ggsave(file.path(paths$output, "treatment_plot.pdf"),
       plot = treatment_plot,
       width = 8,
       height = 6)

# Save intermediate data
write_csv(cleaned_data,
          file.path(paths$intermediate, "cleaned_data.csv"))

# Optional: Save in Stata format if needed for collaboration
write_dta(cleaned_data,
          file.path(paths$intermediate, "cleaned_data.dta"))

# Save final output datasets
final_data <- cleaned_data %>%
  select(key_variables) %>%
  mutate(date_created = Sys.Date())

# Main csv output
write_csv(final_data,
          file.path(paths$output, "final_results.csv"))

# Optional: Save in multiple formats if needed
write_rds(final_data, 
          file.path(paths$output, "final_results.rds"))  # R binary format
write_dta(final_data,
          file.path(paths$output, "final_results.dta"))  # Stata format

# Print confirmation message with file locations
cat("\nFiles saved:\n",
    "Intermediate data:", file.path(paths$intermediate, "cleaned_data.csv"), "\n",
    "Final results:", file.path(paths$output, "final_results.csv"), "\n")