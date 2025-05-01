import yaml
import argparse
import pandas as pd
import subprocess
import sys
import tempfile
import os
from pathlib import Path

def run_job(job_config, default_config, base_output_dir):
    """Runs a single sensitivity analysis job."""
    job_id = job_config['id']
    print(f"--- Starting Job: {job_id} ---")

    # --- Get parameters, using defaults if not specified in job ---
    source_csv = Path(job_config['source_csv'])
    row_index = job_config['row_index']
    rubric = job_config['rubric']
    model = job_config.get('model', default_config['model'])
    repetitions = job_config.get('repetitions', default_config['repetitions'])
    temperature = job_config.get('temperature', default_config['temperature'])
    max_workers = job_config.get('max_workers', default_config['max_workers'])
    keys_file = job_config.get('keys_file', default_config['keys_file'])

    # --- Create specific output directory for this job ---
    output_dir = base_output_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # --- !!! Caching Check !!! ---
    # Construct expected output CSV filename
    safe_model_name = str(model).replace('/', '_').replace(':', '_') # Clean model name
    expected_csv_filename = f"single_pair_results_{safe_model_name}.csv"
    expected_csv_path = output_dir / expected_csv_filename
    
    if expected_csv_path.exists():
        print(f"Output file '{expected_csv_path}' already exists. Assuming job completed. Skipping.")
        return True # Return success as the result exists
    else:
        print(f"Output file '{expected_csv_path}' not found. Proceeding with job execution.")
        
    # --- Read CSV and extract prompt/response ---
    try:
        df = pd.read_csv(source_csv)
        if row_index >= len(df):
            print(f"Error: row_index {row_index} is out of bounds for {source_csv} (length {len(df)}). Skipping job.")
            return False
        
        # Assuming column names are 'prompt' and 'subject_response'
        # Adjust if your column names are different
        prompt_col = 'prompt' 
        response_col = 'subject_response' 

        if prompt_col not in df.columns or response_col not in df.columns:
             print(f"Error: Required columns '{prompt_col}' or '{response_col}' not found in {source_csv}. Skipping job.")
             return False

        prompt_text = df.loc[row_index, prompt_col]
        response_text = df.loc[row_index, response_col]

        if pd.isna(prompt_text) or pd.isna(response_text):
            print(f"Error: Prompt or Response text is missing in row {row_index} of {source_csv}. Skipping job.")
            return False
            
        print(f"Extracted data from {source_csv}, row {row_index}")

    except FileNotFoundError:
        print(f"Error: Source CSV not found: {source_csv}. Skipping job.")
        return False
    except KeyError as e:
        print(f"Error: Column {e} not found in {source_csv}. Check column names. Skipping job.")
        return False
    except Exception as e:
        print(f"Error reading CSV or extracting data: {e}. Skipping job.")
        return False

    # --- Create temporary files for prompt and response ---
    temp_prompt_file = None
    temp_response_file = None
    try:
        # Create named temporary files that persist until manually deleted
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f_prompt:
            f_prompt.write(str(prompt_text))
            temp_prompt_path = f_prompt.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f_response:
            f_response.write(str(response_text))
            temp_response_path = f_response.name
            
        print(f"Created temporary prompt file: {temp_prompt_path}")
        print(f"Created temporary response file: {temp_response_path}")

        # --- Construct command for single_pair_sensitivity.py ---
        # Assuming single_pair_sensitivity.py is in src/ relative to workspace root
        script_path = Path(__file__).parent.parent / 'src' / 'single_pair_sensitivity.py' 
        if not script_path.exists():
            # Fallback if run_multiple_pairs is not in expected location
             script_path = Path('src/single_pair_sensitivity.py') 
             if not script_path.exists():
                 print(f"Error: Could not find src/single_pair_sensitivity.py. Make sure script exists.")
                 return False


        command = [
            sys.executable, # Use the same python interpreter
            str(script_path),
            '--model', str(model),
            '--prompt_file', temp_prompt_path,
            '--response_file', temp_response_path,
            '--rubric', str(rubric),
            '--repetitions', str(repetitions),
            '--temperature', str(temperature),
            '--output_dir', str(output_dir),
            '--keys_file', str(keys_file),
            '--max_workers', str(max_workers)
        ]
        
        print(f"Executing command: {' '.join(command)}")

        # --- Run the subprocess ---
        # Run from the workspace root directory to handle imports correctly in the sub-script
        workspace_root = Path(__file__).parent.parent 
        process = subprocess.run(command, capture_output=True, text=True, check=True, cwd=workspace_root)
        
        print("--- Subprocess STDOUT ---")
        print(process.stdout)
        print("--- Subprocess STDERR ---")
        print(process.stderr)
        print("--- Subprocess Completed Successfully ---")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error executing single_pair_sensitivity.py for job {job_id}.")
        print(f"Return code: {e.returncode}")
        print("--- Subprocess STDOUT ---")
        print(e.stdout)
        print("--- Subprocess STDERR ---")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during job {job_id}: {e}")
        return False
    finally:
        # --- Clean up temporary files ---
        if 'temp_prompt_path' in locals() and os.path.exists(temp_prompt_path):
            os.remove(temp_prompt_path)
            print(f"Deleted temporary prompt file: {temp_prompt_path}")
        if 'temp_response_path' in locals() and os.path.exists(temp_response_path):
            os.remove(temp_response_path)
            print(f"Deleted temporary response file: {temp_response_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multiple single-pair sensitivity analysis jobs defined in a config file.")
    parser.add_argument("-c", "--config", type=str, default="jobs_config.yaml",
                        help="Path to the YAML configuration file defining the jobs.")
    parser.add_argument("-o", "--output", type=str, default="sensitivity_analysis/multi_runs_output",
                        help="Base directory to store the output folders for each job.")
    
    args = parser.parse_args()

    config_path = Path(args.config)
    base_output_dir = Path(args.output)

    if not config_path.is_file():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading or parsing YAML configuration file: {e}")
        sys.exit(1)

    defaults = config.get('defaults', {})
    jobs = config.get('jobs', [])

    if not jobs:
        print("No jobs defined in the configuration file.")
        sys.exit(0)

    print(f"Found {len(jobs)} jobs in {config_path}")
    print(f"Base output directory: {base_output_dir}")
    
    successful_jobs = 0
    failed_jobs = 0

    for job_config in jobs:
        if 'id' not in job_config or 'source_csv' not in job_config or 'row_index' not in job_config or 'rubric' not in job_config:
            print(f"Error: Job definition is missing required fields (id, source_csv, row_index, rubric): {job_config}. Skipping.")
            failed_jobs += 1
            continue
        
        success = run_job(job_config, defaults, base_output_dir)
        if success:
            successful_jobs += 1
        else:
            failed_jobs += 1
        print("----------------------------------------") # Separator

    print(f"\n=== Orchestration Complete ===")
    print(f"Successful jobs: {successful_jobs}")
    print(f"Failed jobs: {failed_jobs}")

if __name__ == "__main__":
    main() 