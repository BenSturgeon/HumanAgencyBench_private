# %%
import pandas as pd
from pathlib import Path
import glob

# %%
# Adjust pandas display options for better readability in terminal
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', None) # Use this if columns still seem truncated

# %%
# Define the base directory and directories/patterns to exclude
base_dir = Path("output")
# We want directories directly under output, excluding specific ones
# Let's list the directories we *want* to include based on the structure provided earlier
# Or find all dirs, then filter
all_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
exclude_dirs = ['eval_output', 'generation_output'] # Exclude these directory names
target_dirs = [d for d in all_dirs if d.name not in exclude_dirs]

print(target_dirs)
# Comment out the line that restricts processing to a single directory
# target_dirs = [Path("output/acknowledge_limitations")]

print(f"Found {len(target_dirs)} evaluation directories to process:")
for d in target_dirs:
    print(f"- {d.name}")



# Construct file paths
all_csv_files = [d / "raw.csv" for d in target_dirs if (d / "raw.csv").is_file()]

print(f"\nFound {len(all_csv_files)} raw.csv files to process:")
for f in all_csv_files:
    print(f"- {f}")


# %%


# List to hold individual dataframes
df_list = []

# Process each CSV file
for file_path in all_csv_files:
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Extract the evaluation dimension from the parent directory name
        evaluation_dimension = file_path.parent.name

        # Add the evaluation dimension as a new column
        df['evaluation_dimension'] = evaluation_dimension

        # Keep only relevant columns (assuming 'subject_model' and 'score' exist)
        if 'subject_model' in df.columns and 'score' in df.columns:
            # Select and copy to avoid SettingWithCopyWarning later
            df_subset = df[['subject_model', 'score', 'evaluation_dimension']].copy()
            # --- Clean subject_model name ---
            # Apply cleaning only if subject_model is not empty/NaN
            df_subset['subject_model'] = df_subset['subject_model'].apply(
                lambda x: Path(x).name if pd.notna(x) and isinstance(x, str) and '/' in x else x
            )
            df_list.append(df_subset)
        else:
            print(f"Warning: 'subject_model' or 'score' column not found in {file_path}. Skipping this file.")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# %%
# Concatenate all dataframes into one
if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)
    print("\nCombined DataFrame created successfully.")
    print("First 5 rows:")
    print(combined_df.head())
    print("\nInfo:")
    combined_df.info()
else:
    print("\nNo dataframes were created. Please check the CSV files and paths.")


# %%
# Now you can analyze the combined_df
# Calculate average scores and display in a table
if df_list and 'combined_df' in locals() and not combined_df.empty: # Make sure combined_df was created
    print("\nCalculating average scores...")
    # Convert score to numeric, coercing errors (non-numeric scores become NaN)
    # This handles cases where 'score' might contain non-numeric values
    combined_df['score'] = pd.to_numeric(combined_df['score'], errors='coerce')

    # Report rows where score could not be converted
    invalid_score_rows = combined_df[combined_df['score'].isna()]
    if not invalid_score_rows.empty:
        print(f"\nWarning: Found {len(invalid_score_rows)} rows with non-numeric scores. These will be excluded from averaging.")
        # Optional: print details of affected rows/files if needed for debugging
        # print(invalid_score_rows[['evaluation_dimension', 'subject_model', 'score']])

    # Drop rows where score is NaN before grouping
    combined_df.dropna(subset=['score'], inplace=True)

    if not combined_df.empty:
        # Calculate the average score
        final_scores = combined_df.groupby(['evaluation_dimension', 'subject_model'])['score'].mean().reset_index()
        # Apply scaling *only* to the score column
        final_scores['score'] = final_scores['score'] * 10

        print("\nAverage Scores per Model per Dimension:")
        print(final_scores)

        # Pivot the table for a clearer view (Models as columns, Dimensions as rows)
        try:
            # Use the cleaned model names for columns
            pivot_table = final_scores.pivot(index='evaluation_dimension', columns='subject_model', values='score')
            print("\n\nPivoted Average Scores Table:")
            # Format the scores to 2 decimal places for cleaner output
            print(pivot_table.to_string(float_format="%.2f"))
        except ValueError as ve:
             print(f"\nCould not create pivot table due to duplicate entries for the same model and dimension: {ve}")
             print("This might happen if a model appears multiple times within the same raw.csv with the same score after potential coercion.")
             print("Displaying grouped data instead:")
             print(final_scores)
        except Exception as e:
            print(f"\nCould not create pivot table: {e}")
            print("Displaying grouped data instead:")
            print(final_scores)
    else:
        print("\nNo valid numeric scores found after cleaning. Cannot calculate averages.")

else:
    print("\nCannot perform analysis as the combined dataframe is empty or was not created.") 


# Calculate and display overall average scores

if 'final_scores' in locals() and not final_scores.empty:
    print("\nCalculating overall average scores...")
    
    # Average score per dimension (across all models)
    avg_score_per_dimension = final_scores.groupby('evaluation_dimension')['score'].mean().reset_index()
    print("\nOverall Average Score per Dimension:")
    print(avg_score_per_dimension.to_string(float_format="%.2f"))
    
    # Average score per model (across all dimensions)
    avg_score_per_model = final_scores.groupby('subject_model')['score'].mean().reset_index()
    print("\nOverall Average Score per Model:")
    print(avg_score_per_model.to_string(float_format="%.2f"))

else:
    print("\nCannot calculate overall averages because 'final_scores' DataFrame is not available or empty.")

# %%
# Generate LaTeX code for the pivot table

if 'pivot_table' in locals() and not pivot_table.empty:
    print("\nGenerating LaTeX code for the pivot table...")
    # Ensure model names (columns) and dimensions (index) are strings
    pivot_table.columns = pivot_table.columns.astype(str)
    pivot_table.index = pivot_table.index.astype(str)
    
    # Generate LaTeX string
    # escape=True helps handle special characters like underscores
    # clines='all;data' adds lines between rows which might be useful for readability
    latex_string = pivot_table.to_latex(
        float_format="%.2f", 
        caption="HAB Scores by Dimension and Model. Higher scores indicate greater alignment with the agency dimension.", # Placeholder caption
        label="tab:hab_results", # Placeholder label
        position="htbp", # Standard table positioning
        escape=True # Escape special LaTeX characters in index/columns
    )
    
    print("\n--- LaTeX Code Start ---")
    print(latex_string)
    print("--- LaTeX Code End ---")
    
else:
    print("\nCannot generate LaTeX code because 'pivot_table' is not available or empty.")

# %%
# Save the pivot table to a CSV file
if 'pivot_table' in locals() and not pivot_table.empty:
    try:
        csv_filename = "hab_scores_pivot_table.csv"
        # Save the pivot table to CSV, applying float formatting for consistency
        pivot_table.to_csv(csv_filename, float_format="%.2f")
        print(f"\nPivot table saved to {csv_filename}")
    except Exception as e:
        print(f"\nError saving pivot table to CSV: {e}")
else:
    print("\nCannot save pivot table to CSV because 'pivot_table' is not available or empty.")
