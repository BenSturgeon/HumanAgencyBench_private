import pandas as pd
import argparse
import os
from pathlib import Path

def create_baseline_sample(source_csv: str, output_csv: str, model_filter: str, sample_size: int):
    """
    Reads a raw results CSV, filters by subject model, samples responses, 
    and saves them as a baseline dataset.

    Args:
        source_csv: Path to the input raw.csv file.
        output_csv: Path to save the sampled baseline data.
        model_filter: The exact string for the subject_model to filter by.
        sample_size: The desired number of samples.
    """
    print(f"Reading source CSV: {source_csv}")
    source_path = Path(source_csv)
    output_path = Path(output_csv)

    if not source_path.is_file():
        print(f"Error: Source file not found at {source_csv}")
        return

    try:
        # Define columns to load to potentially save memory
        # We need 'prompt', 'subject_response', and 'subject_model'
        use_cols = ['prompt', 'subject_response', 'subject_model'] 
        
        # Consider using chunking if the file is extremely large, 
        # but try direct loading first
        df = pd.read_csv(source_path, usecols=use_cols)
        print(f"Successfully loaded {len(df)} rows.")

        print(f"Filtering for subject_model == '{model_filter}'")
        filtered_df = df[df['subject_model'] == model_filter].copy() # Use .copy() to avoid SettingWithCopyWarning
        print(f"Found {len(filtered_df)} rows matching the model filter.")

        if len(filtered_df) == 0:
            print("Error: No rows found matching the specified model filter. Cannot create baseline.")
            return

        if len(filtered_df) > sample_size:
            print(f"Sampling {sample_size} rows randomly.")
            sampled_df = filtered_df.sample(n=sample_size, random_state=42) # Use a fixed random state for reproducibility
        else:
            print(f"Found {len(filtered_df)} matching rows, which is less than or equal to the requested sample size ({sample_size}). Using all matching rows.")
            sampled_df = filtered_df

        # Select and rename columns for the baseline file
        baseline_df = sampled_df[['prompt', 'subject_response']].copy() # Explicit copy
        baseline_df.rename(columns={'subject_response': 'response'}, inplace=True)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving {len(baseline_df)} sampled pairs to: {output_csv}")
        baseline_df.to_csv(output_path, index=False)
        print("Baseline dataset created successfully.")

    except FileNotFoundError:
        print(f"Error: Source file not found at {source_csv}")
    except KeyError as e:
         print(f"Error: Missing expected column in source CSV: {e}. Required columns: {use_cols}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample prompt-response pairs from a raw pipeline output CSV to create a baseline dataset for sensitivity analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--source_csv", 
        type=str, 
        default="output/avoid_value_manipulation/raw.csv",
        help="Path to the source raw.csv file from a pipeline run."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="sensitivity_analysis/baseline_responses_MODEL.csv",
        help="Path to save the output baseline CSV file. Replace MODEL with a model identifier."
    )
    parser.add_argument(
        "--model_filter", 
        type=str, 
        required=True,
        help="The exact string identifier for the subject_model to filter by."
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=500,
        help="The desired number of prompt-response pairs to sample."
    )

    args = parser.parse_args()

    create_baseline_sample(
        source_csv=args.source_csv,
        output_csv=args.output_csv,
        model_filter=args.model_filter,
        sample_size=args.sample_size
    ) 