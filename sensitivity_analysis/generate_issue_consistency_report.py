# generate_issue_consistency_report.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import glob
import os
import argparse
import re
from pathlib import Path
import html # For escaping text in HTML

def load_and_combine_data(input_pattern: str) -> pd.DataFrame:
    """Loads all CSVs matching the pattern, adds identifiers, and combines them."""
    print(f"[DEBUG] Searching with pattern: {input_pattern}") # Debug pattern
    all_files = glob.glob(input_pattern, recursive=True)
    if not all_files:
        print(f"Error: No files found matching pattern: {input_pattern}")
        return pd.DataFrame()

    print(f"[DEBUG] First 5 files found: {all_files[:5]}") # Debug files found

    df_list = []
    print(f"Found {len(all_files)} result files. Loading...")
    for i, f in enumerate(all_files):
        # --- Debugging Prints Ensure these are active ---
        if i < 5:
            print(f"\n[DEBUG] Processing file path: {f}")
        try:
            df = pd.read_csv(f)
            parts = Path(f).parts
            if i < 5: print(f"[DEBUG]   Path parts: {parts}")
            # --- Infer identifiers from path ---
            # REVISED LOGIC: Assume structure like .../input_dir/dimension_sample<N>/results.csv
            if len(parts) >= 2:
                pair_folder_name = parts[-2] # e.g., 'ask_clarifying_sample1'
                if i < 5: print(f"[DEBUG]   Pair folder name (parts[-2]): {pair_folder_name}")

                # Try to extract dimension prefix using specific suffix pattern
                match = re.match(r"^(.*?)_sample(\d+)$", pair_folder_name)
                if i < 5: print(f"[DEBUG]   Regex match result: {match}")

                if match:
                    dimension_name_raw = match.group(1) # Group 1 captures everything before _sample<N>
                    dimension_group = dimension_name_raw.replace('_', ' ').title() # e.g., 'Ask Clarifying'
                    pair_id = pair_folder_name # Keep full folder name as pair ID
                    if i < 5: print(f"[DEBUG]     -> Matched! Raw Dim: '{dimension_name_raw}'")
                else:
                    # Fallback if pattern doesn't match (e.g., no _sampleN suffix)
                    # Use the whole folder name as the dimension, cleaned up.
                    dimension_group = pair_folder_name.replace('_', ' ').title()
                    pair_id = pair_folder_name
                    if i < 5: print(f"[DEBUG]     -> No Match! Using folder name for Dim: '{dimension_group}'")
            else:
                # Fallback for very short paths
                pair_id = f"unknown_pair_{os.path.basename(f)}"
                dimension_group = "Unknown Dimension (Path Too Short)"

            if i < 5: print(f"[DEBUG]   Assigned Dimension Group: {dimension_group}")
            if i < 5: print(f"[DEBUG]   Assigned Pair ID: {pair_id}")

            df['source_file'] = os.path.basename(f)
            df['pair_id'] = pair_id
            df['dimension_group'] = dimension_group
            df['run_id'] = f"{dimension_group}_{pair_id}"

            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            # Fix pandas warning: Avoid chained assignment with inplace=True
            df['score'] = df['score'].fillna(-1)

            if 'reported_original_issues' not in df.columns:
                 print(f"Warning: Column 'reported_original_issues' not found in {f}. Skipping file.")
                 continue

            df_list.append(df)
        except Exception as e:
            print(f"Error loading or processing file {f}: {e}")

    if not df_list:
        print("Error: No data could be loaded successfully.")
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    unique_dims_found = combined_df['dimension_group'].unique()
    print(f"[DEBUG] Unique dimension groups identified after loading: {unique_dims_found}") # Add final check
    print(f"Successfully loaded {len(combined_df)} total repetitions from {combined_df['run_id'].nunique()} unique runs across {len(unique_dims_found)} dimensions.")
    return combined_df

def calculate_issue_stats_per_dimension(df: pd.DataFrame) -> dict:
    """Calculates issue frequencies and consistency stats per dimension."""
    if df.empty or 'reported_original_issues' not in df.columns or 'dimension_group' not in df.columns or 'run_id' not in df.columns:
        print("Error: DataFrame is empty or missing required columns for analysis.")
        return {}

    all_dimension_stats = {}
    unique_dimensions = df['dimension_group'].unique()

    for dim in unique_dimensions:
        print(f"Processing dimension: {dim}...")
        dim_df = df[df['dimension_group'] == dim].copy()
        run_ids_in_dim = dim_df['run_id'].unique()
        
        # --- Calculate per-run stats FIRST ---
        per_run_score_means = []
        per_run_score_stddevs = []
        for run_id in run_ids_in_dim:
            run_scores = dim_df[(dim_df['run_id'] == run_id) & (dim_df['score'] >= 0)]['score']
            if not run_scores.empty:
                per_run_score_means.append(run_scores.mean())
                per_run_score_stddevs.append(run_scores.std() if len(run_scores) > 1 else 0)
            else:
                 # Handle runs with no valid scores if necessary, maybe append NaN or skip?
                 # For now, let's assume most runs have valid scores.
                 pass 
                 
        # Calculate overall summary of per-run score stats
        avg_per_run_mean_score = np.mean(per_run_score_means) if per_run_score_means else np.nan
        avg_per_run_stddev_score = np.mean(per_run_score_stddevs) if per_run_score_stddevs else np.nan
        std_per_run_mean_score = np.std(per_run_score_means) if len(per_run_score_means) > 1 else 0
        
        run_score_summary = {
            'Avg Mean Score (per run)': round(avg_per_run_mean_score, 2),
            'Std Dev of Mean Scores (across runs)': round(std_per_run_mean_score, 2),
            'Avg Score Std Dev (per run)': round(avg_per_run_stddev_score, 2)
        }

        # --- Calculate per-run issue frequencies ---
        per_run_issue_freq = {} # {run_id: {issue: freq_count}}
        all_issues_in_dim = set()
        valid_reps_per_run = {} # {run_id: count}

        for run_id in run_ids_in_dim:
            run_df = dim_df[dim_df['run_id'] == run_id]
            run_issue_counts = Counter()
            valid_reps = 0
            for issues_str in run_df['reported_original_issues']:
                # Check for NaN or specific error strings before processing
                if pd.isna(issues_str) or issues_str in ["Invalid JSON", "LLM Call/Processing Error", "Error: 'issues' field missing", "Parsing failed", "Translation Error"]:
                    continue
                valid_reps += 1
                # Split issues, clean them up
                issues_list = [issue.strip() for issue in issues_str.split('||') if issue.strip() and not issue.startswith("Unknown Letter:")]
                if not issues_list and issues_str.strip() == "":
                     run_issue_counts["__No Issues Reported__"] += 1
                else:
                    run_issue_counts.update(issues_list)

            per_run_issue_freq[run_id] = dict(run_issue_counts) # Convert counter to dict
            all_issues_in_dim.update(run_issue_counts.keys())
            valid_reps_per_run[run_id] = valid_reps

        if not all_issues_in_dim:
            print(f"  No valid issues identified for dimension {dim}. Skipping.")
            continue

        # --- Aggregate stats across runs within the dimension ---
        issue_stats_list = []
        sorted_issues = sorted(list(all_issues_in_dim))

        for issue in sorted_issues:
            # Collect the frequency (count / total valid reps for that run) of this issue across all runs in the dimension
            frequencies_for_issue = [] # List of counts for this issue across runs
            total_count_for_issue = 0
            runs_reporting_issue = 0

            for run_id in run_ids_in_dim:
                count = per_run_issue_freq.get(run_id, {}).get(issue, 0)
                frequencies_for_issue.append(count)
                total_count_for_issue += count
                if count > 0:
                    runs_reporting_issue += 1

            # Calculate stats based on counts
            avg_freq_count = np.mean(frequencies_for_issue) if frequencies_for_issue else 0
            std_dev_freq_count = np.std(frequencies_for_issue) if len(frequencies_for_issue) > 1 else 0
            
            # Calculate Coefficient of Variation (handle division by zero)
            cv_freq_count = (std_dev_freq_count / avg_freq_count) if avg_freq_count > 0 else 0
            # Calculate % of runs reporting the issue
            percent_runs_reporting = (runs_reporting_issue / len(run_ids_in_dim)) * 100 if len(run_ids_in_dim) > 0 else 0

            issue_stats_list.append({
                "Underlying Issue": issue,
                "Total Count (Dimension)": total_count_for_issue,
                "Avg Freq per Run (Count)": round(avg_freq_count, 2),
                "Std Dev of Counts (per run)": round(std_dev_freq_count, 2), # Clarified label
                "CV of Counts (per run)": round(cv_freq_count, 2), # Added CV
                "# Runs Reporting Issue": runs_reporting_issue,
                "% Runs Reporting Issue": round(percent_runs_reporting, 1) # Added % Runs Reporting
            })

        stats_df = pd.DataFrame(issue_stats_list)
        stats_df["# Runs in Dimension"] = len(run_ids_in_dim) # Add total runs for context
        stats_df = stats_df.sort_values(by="Avg Freq per Run (Count)", ascending=False) # Sort

        all_dimension_stats[dim] = {
            'issue_stats': stats_df,
            'score_summary': run_score_summary 
        }
        print(f"  Finished processing dimension: {dim}")

    return all_dimension_stats

def generate_html_report(dimension_stats: dict, output_file: str):
    """Generates the HTML report from the calculated statistics."""
    html_parts = []
    html_parts.append("<!DOCTYPE html>") # Ensure proper doctype
    html_parts.append("<html><head><meta charset=\"UTF-8\"><title>Underlying Issue Consistency Report</title>") # Add charset
    # Add CSS and Plotly JS
    html_parts.append("""
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            .dimension-section { margin-bottom: 40px; border: 1px solid #ccc; padding: 20px; border-radius: 8px; overflow-x: auto; } /* Add overflow-x */
            h1, h2 { color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            table { border-collapse: collapse; width: 100%; margin-top: 15px; font-size: 0.9em; table-layout: auto; } /* Changed width and layout */
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; word-wrap: break-word; } /* Added word-wrap */
            th { background-color: #f2f2f2; white-space: nowrap; } /* Added nowrap */
            .plotly-graph-div { margin-top: 15px; }
            pre { background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }
        </style>
    </head><body>""")
    html_parts.append("<h1>Underlying Issue Identification Consistency Report</h1>")

    sorted_dimensions = sorted(dimension_stats.keys())

    for dim in sorted_dimensions:
        dim_data = dimension_stats[dim]
        stats_df = dim_data['issue_stats']
        score_summary = dim_data['score_summary']
        
        if stats_df.empty:
            continue
            
        html_parts.append(f"<div class='dimension-section'>")
        html_parts.append(f"<h2>Dimension: {html.escape(dim)}</h2>")

        # --- Display Score Summary Stats ---
        html_parts.append("<h3>Score Consistency Summary (Across Runs)</h3>")
        if not all(np.isnan(v) for v in score_summary.values()):
            score_summary_df = pd.DataFrame([score_summary])
            html_parts.append(score_summary_df.to_html(index=False, justify="left", na_rep="N/A"))
        else:
            html_parts.append("<p>No valid scores found to summarize consistency.</p>")
        
        # --- Generate Horizontal Bar Chart ---
        plot_df = stats_df[stats_df["Underlying Issue"] != "__No Issues Reported__"].copy()

        if not plot_df.empty:
            fig = px.bar(
                plot_df,
                y="Underlying Issue",
                x="Avg Freq per Run (Count)",
                error_x="Std Dev of Counts (per run)", # Use Std Dev for error bars
                orientation='h',
                title=f"Average Frequency of Identified Issues per Run (± Std Dev)",
                labels={
                    "Underlying Issue": "Issue Description",
                    "Avg Freq per Run (Count)": "Average Count per ~50 Repetitions" # Clarify y-axis unit
                },
                 height=max(400, len(plot_df) * 35) # Adjust height dynamically
            )
            # Set y-axis category order based on the sorted DataFrame's x value
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            html_parts.append(chart_html)
        else:
            html_parts.append("<p>No specific issues reported for this dimension.</p>")

        # --- Generate Statistics Table ---
        # Display all rows, including "No Issues"
        # Rename columns for clarity in the final table
        display_df = stats_df.rename(columns={
            "Std Dev of Counts (per run)": "Std Dev (Counts)",
            "CV of Counts (per run)": "CV (Counts)"
        })
        table_html = display_df.to_html(index=False, justify="left", escape=True, na_rep="N/A") # Ensure escaping
        html_parts.append("<h3>Issue Frequency & Consistency Statistics</h3>")
        html_parts.append(table_html)

        html_parts.append("</div>") # Close dimension-section div

    html_parts.append("</body></html>") # Close body and html tags

    # Write to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_parts))
        print(f"Successfully generated HTML report: {output_file}")
    except Exception as e:
        print(f"Error writing HTML report to {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an HTML report summarizing underlying issue consistency from sensitivity runs.")
    parser.add_argument("input_dir",
                        help="Directory containing the sensitivity run outputs (e.g., 'sensitivity_analysis/'). Use quotes if path contains spaces.")
    parser.add_argument("-o", "--output_html", default="issue_consistency_report.html",
                        help="Name for the output HTML report file (default: issue_consistency_report.html).")
    parser.add_argument("-p", "--pattern", default="**/single_pair_results_*.csv",
                        help="Glob pattern to find CSV files relative to input_dir (default: '**/single_pair_results_*.csv').")

    args = parser.parse_args()

    # Construct the full search pattern
    full_pattern = str(Path(args.input_dir) / args.pattern)

    # Load data
    combined_data = load_and_combine_data(full_pattern)

    if not combined_data.empty:
        # Calculate stats
        dimension_statistics = calculate_issue_stats_per_dimension(combined_data)

        # Generate report
        if dimension_statistics:
             generate_html_report(dimension_statistics, args.output_html)
        else:
             print("Skipping report generation as no statistics were calculated.")
    else:
        print("\nAnalysis skipped due to data loading issues.") 