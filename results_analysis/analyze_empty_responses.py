import csv
import os
import glob
from collections import defaultdict
import argparse

def analyze_empty_responses(output_dir="output", subdimensions_to_analyze=None):
    """
    Analyzes raw.csv files in specified subdirectories of output_dir to count empty
    assistant responses and specific error messages for each model, broken down by subdimension.
    Only non-zero counts are reported.

    Args:
        output_dir (str): The path to the main output directory.
        subdimensions_to_analyze (list[str], optional): A list of subdimension names
            (subdirectory names) to analyze. If None or empty, analyzes all found.
    """
    empty_response_marker = "<<<<<<<<<<\n\n>>>>>>>>>>"
    error_marker = "[ERROR DURING LLM CHAT]"
    # Use nested defaultdicts: outer key = subdim, inner key = model
    empty_counts = defaultdict(lambda: defaultdict(int))
    error_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(lambda: defaultdict(int))
    processed_subdims = set() # Keep track of which subdims were actually processed
    found_files = 0

    search_pattern = os.path.join(output_dir, "*", "raw.csv")
    all_csv_files = glob.glob(search_pattern)

    if not all_csv_files:
        print(f"Error: No raw.csv files found in subdirectories of '{output_dir}'.")
        print(f"Searched pattern: {search_pattern}")
        return

    if subdimensions_to_analyze:
        print(f"Filtering analysis to subdimensions: {', '.join(subdimensions_to_analyze)}")
        csv_files_to_process = []
        for filepath in all_csv_files:
            subdim_name = os.path.basename(os.path.dirname(filepath))
            if subdim_name in subdimensions_to_analyze:
                csv_files_to_process.append(filepath)
        if not csv_files_to_process:
            print(f"Warning: No raw.csv files found for the specified subdimensions in '{output_dir}'.")
            return
    else:
        print("Analyzing all found subdimensions.")
        csv_files_to_process = all_csv_files

    print(f"Found {len(csv_files_to_process)} raw.csv files to analyze...")

    for filepath in csv_files_to_process:
        found_files += 1
        subdim_name = os.path.basename(os.path.dirname(filepath))
        processed_subdims.add(subdim_name)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as csvfile:
                reader = csv.DictReader(csvfile)
                if 'subject_model' not in reader.fieldnames or 'subject_response' not in reader.fieldnames:
                    print(f"Warning: Skipping {filepath}. Missing required columns ('subject_model' or 'subject_response').")
                    print(f"  Available columns: {reader.fieldnames}")
                    continue

                for row in reader:
                    model = row.get('subject_model')
                    response = row.get('subject_response')

                    if model is None or response is None:
                        print(f"Warning: Skipping row in {filepath} due to missing model or response data.")
                        continue

                    # Increment total count for the model within the specific subdimension
                    total_counts[subdim_name][model] += 1

                    if not response.strip():
                        empty_counts[subdim_name][model] += 1
                    elif error_marker in response:
                        error_counts[subdim_name][model] += 1

        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    analysis_scope = "specified" if subdimensions_to_analyze else "all"
    print(f"\n--- Analysis Complete ({found_files} files processed for {analysis_scope} subdimensions) ---")

    if not total_counts:
        print("No data processed.")
        return

    # Sort processed subdimensions for consistent output order
    sorted_subdims = sorted(list(processed_subdims))

    for subdim in sorted_subdims:
        print(f"\n--- Subdimension: {subdim} ---")
        subdim_has_output = False # Flag to track if any output was printed for this subdim

        # Get all models that appeared in this subdimension
        models_in_subdim = sorted(total_counts[subdim].keys())

        print("  Non-Zero Empty Response Counts:")
        empty_output_printed = False
        for model in models_in_subdim:
            empty_count = empty_counts[subdim].get(model, 0)
            if empty_count > 0:
                total_count = total_counts[subdim][model]
                percentage = (empty_count / total_count) * 100
                print(f"  - {model}: {empty_count} empty responses out of {total_count} total ({percentage:.2f}%)")
                empty_output_printed = True
                subdim_has_output = True
        if not empty_output_printed:
            print("    (None)")

        print("\n  Non-Zero Error Response Counts:")
        error_output_printed = False
        for model in models_in_subdim:
            error_count = error_counts[subdim].get(model, 0)
            if error_count > 0:
                total_count = total_counts[subdim][model]
                percentage = (error_count / total_count) * 100
                print(f"  - {model}: {error_count} errors ('{error_marker}') out of {total_count} total ({percentage:.2f}%)")
                error_output_printed = True
                subdim_has_output = True
        if not error_output_printed:
            print("    (None)")

        # Optional: Add a message if a subdim had no non-zero counts at all
        # if not subdim_has_output:
        #     print("  (No empty or error responses found for any model in this subdimension)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze empty/error responses in evaluation results, broken down by subdimension.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="The main output directory containing subdimension folders."
    )
    parser.add_argument(
        "--subdims",
        nargs='*',
        default=None,
        help="Specific subdimensions (subdirectory names) to analyze. If not provided, all subdimensions are analyzed."
    )
    args = parser.parse_args()

    analyze_empty_responses(output_dir=args.output_dir, subdimensions_to_analyze=args.subdims) 