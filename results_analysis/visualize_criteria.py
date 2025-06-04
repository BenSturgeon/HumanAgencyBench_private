import pandas as pd
# import plotly.express as px # Removed: No longer generating HTML charts
import os
from collections import Counter
import json # Added for parsing list-like strings
import copy
from results_analysis.prompt_parser import load_criteria_mappings # Import the new function
# import plotly.graph_objects as go # Removed: No longer generating HTML charts

def find_csv_files(base_path):
    """Finds all raw.csv files in the subdirectories of base_path."""
    csv_files = []
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        print(f"Error: Base path {base_path} does not exist or is not a directory.")
        return csv_files

    for category_name_from_dir in os.listdir(base_path):
        category_path = os.path.join(base_path, category_name_from_dir)
        if os.path.isdir(category_path):
            direct_raw_csv_path = os.path.join(category_path, 'raw.csv')
            if os.path.exists(direct_raw_csv_path) and os.path.isfile(direct_raw_csv_path):
                csv_files.append((category_name_from_dir, direct_raw_csv_path))
            
            for item in os.listdir(category_path):
                item_path = os.path.join(category_path, item)
                if os.path.isdir(item_path) and 'run' in item: 
                    raw_csv_path = os.path.join(item_path, 'raw.csv')
                    if os.path.exists(raw_csv_path) and os.path.isfile(raw_csv_path):
                        # Use the main category name, not with _run_X
                        csv_files.append((category_name_from_dir, raw_csv_path))
    return csv_files

def count_criteria(csv_files, criteria_column_name, criteria_mappings):
    """Counts occurrences of each criterion, mapping keys to descriptions."""
    category_criteria_counts = {}
    for category_name, csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            if criteria_column_name not in df.columns:
                print(f"Warning: Column '{criteria_column_name}' not found in {csv_path} for category '{category_name}'. Skipping.")
                continue
            
            df_cleaned = df.dropna(subset=[criteria_column_name])
            current_category_mappings = criteria_mappings.get(category_name)

            if not current_category_mappings:
                print(f"Warning: No criteria mappings found for category '{category_name}'. Will count raw values from {csv_path}.")
                raw_counts = Counter()
                for val in df_cleaned[criteria_column_name].astype(str):
                    raw_counts.update([f"UNMAPPED_CATEGORY_RAW_VALUE: {val} (Category: {category_name})"])
                if category_name not in category_criteria_counts:
                    category_criteria_counts[category_name] = Counter()
                category_criteria_counts[category_name].update(raw_counts)
                continue

            processed_criteria_for_category = Counter()
            for _, row in df_cleaned.iterrows():
                eval_response_value = row[criteria_column_name]
                criteria_keys_found = []

                if isinstance(eval_response_value, str):
                    try:
                        # Attempt to parse as a JSON object
                        data = json.loads(eval_response_value)
                        if isinstance(data, dict) and 'issues' in data and isinstance(data['issues'], list):
                            criteria_keys_found.extend(data['issues'])
                        elif isinstance(data, list): # Handles if eval_response_value was '["A","B"]' directly
                            criteria_keys_found.extend(data)
                        elif len(eval_response_value) == 1 and eval_response_value.isalpha():
                            criteria_keys_found.append(eval_response_value) # Single letter
                        else:
                            print(f"Warning: Unexpected JSON structure in '{criteria_column_name}' column: '{eval_response_value[:100]}'... in {csv_path}. Expected JSON object with 'issues' list or a direct list of issue codes.")
                    except json.JSONDecodeError:
                        # If not a valid JSON object string, it might be a single key (e.g. 'A') or some other format.
                        if len(eval_response_value) == 1 and eval_response_value.isalpha():
                            criteria_keys_found.append(eval_response_value)
                        elif eval_response_value.strip(): # Non-empty string that is not parseable JSON and not a single letter
                            print(f"Warning: Could not parse JSON from '{criteria_column_name}' column and not a single letter: '{eval_response_value[:100]}'... in {csv_path}.")
                            # Count this problematic string as a distinct unmapped item
                            processed_criteria_for_category.update([f"UNPARSEABLE_VALUE: {eval_response_value[:50]} (Category: {category_name})"])
                elif isinstance(eval_response_value, (list, set, tuple)):
                     criteria_keys_found.extend([str(k) for k in eval_response_value]) # Should ideally not happen if CSV is text
                else: # Numbers or other types
                    if pd.notna(eval_response_value):
                         criteria_keys_found.append(str(eval_response_value))

                for key in criteria_keys_found:
                    key_str = str(key).strip() # Ensure it's a string and cleaned
                    criterion_description = current_category_mappings.get(key_str)
                    if criterion_description:
                        processed_criteria_for_category.update([criterion_description])
                    else:
                        if key_str: # Avoid counting empty strings if they slip through
                            processed_criteria_for_category.update([f"UNMAPPED_KEY: {key_str} (Category: {category_name})"])
            
            if category_name not in category_criteria_counts:
                category_criteria_counts[category_name] = Counter()
            category_criteria_counts[category_name].update(processed_criteria_for_category)
            
        except pd.errors.EmptyDataError:
            print(f"Warning: {csv_path} for category '{category_name}' is empty. Skipping.")
        except Exception as e:
            print(f"Error processing {csv_path} for category '{category_name}': {e}")
    return category_criteria_counts

def main():
    # Determine project root assuming visualize_criteria.py is in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    prompts_src_dir = os.path.join(project_root, 'src', 'prompts')
    base_output_path = os.path.join(project_root, 'output', 'output_neurips_2025')
    # visualization_output_directory = os.path.join(project_root, 'visualizations') # Removed: No longer used

    print(f"Loading criteria mappings from: {prompts_src_dir}")
    criteria_mappings = load_criteria_mappings(prompts_dir=prompts_src_dir)
    
    if not criteria_mappings:
        print("Error: No criteria mappings loaded. Please check 'prompt_parser.py' and the structure of 'src/prompts'. Exiting.")
        return
    else:
        print(f"Successfully loaded mappings for {len(criteria_mappings)} categories.")

    # --- IMPORTANT ---
    # This is the column in your raw.csv files that contains the criteria KEYS (e.g., 'A', 'B', or ['A', 'C'])
    criteria_column_name = 'evaluator_response' 
    # --- IMPORTANT ---
    
    csv_files = find_csv_files(base_output_path)
    
    if not csv_files:
        print(f"No 'raw.csv' files found in the subdirectories of {base_output_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")

    category_counts = count_criteria(csv_files, criteria_column_name, criteria_mappings)
    
    # Print all values in terminal for debugging
    print("\n" + "="*80)
    print("CRITERIA COUNTS BY CATEGORY")
    print("="*80)
    
    for category_name, criteria_counts in category_counts.items():
        print(f"\n{category_name.upper().replace('_', ' ')}:")
        print("-" * 50)
        for criterion, count in criteria_counts.items():
            print(f"  {count:>6} | {criterion}")
        print(f"  Total: {sum(criteria_counts.values())}")
    
    print("\n" + "="*80)
    
    if not category_counts:
        print("No criteria counts were generated. Check warnings for issues.")

if __name__ == '__main__':
    main() 