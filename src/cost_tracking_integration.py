"""
Cost Tracking Integration

This module integrates the cost tracker with the existing LLM implementation
without requiring changes to the core codebase.

Usage:
    # At the start of your pipeline, simply import this module:
    import src.cost_tracking_integration

    # All LLM API calls will now be tracked automatically
"""

import os
import json
import time
import datetime
import uuid
from functools import wraps
from typing import Dict, Any, Optional, List, Union

# Import the original LLM classes
from src.llms import LLM, OpenAILLM, AnthropicLLM, GroqLLM, ReplicateLLM, GeminiLLM

# Import the cost tracker
from test_cost_tracker import CostTracker, MODEL_COSTS

# Create a single global instance of the cost tracker
LOG_DIR = os.environ.get('COST_LOG_DIR', 'cost_logs')
os.makedirs(LOG_DIR, exist_ok=True)
COST_LOG_FILE = os.path.join(LOG_DIR, 'api_costs.json')
cost_tracker = CostTracker(COST_LOG_FILE)

# Track models with missing cost information
UNKNOWN_MODELS = set()

# Generate a unique run ID for this session
CURRENT_RUN_ID = str(uuid.uuid4())
RUN_START_TIME = datetime.datetime.now().isoformat()
RUN_METADATA = {
    "run_id": CURRENT_RUN_ID,
    "start_time": RUN_START_TIME,
    "command_line": " ".join(os.sys.argv),
    "working_directory": os.getcwd(),
    "user": os.environ.get("USER", "unknown"),
}

# Track cached requests to avoid double-charging for input tokens
CACHED_REQUEST_IDS = set()

def is_cached_request(request_id, metadata):
    """Check if this is a cached request based on metadata"""
    if request_id in CACHED_REQUEST_IDS:
        return True
        
    # Check for cache indicators in metadata
    if metadata and isinstance(metadata, dict):
        if metadata.get('cached') == True or metadata.get('is_cached') == True:
            CACHED_REQUEST_IDS.add(request_id)
            return True
    
    return False


def track_chat_call(func):
    """
    Decorator to track the cost of chat API calls.
    This captures the input and output and sends it to the cost tracker.
    """
    @wraps(func)
    def wrapper(self, prompt, *args, **kwargs):
        # Get the messages from the appropriate place based on class type
        try:
            # For the main LLM class, get messages from the underlying LLM implementation
            if isinstance(self, LLM) and hasattr(self, 'llm') and hasattr(self.llm, 'messages'):
                pre_call_messages = self.llm.messages.copy()
            # For the specific LLM implementations that have messages attribute directly
            elif hasattr(self, 'messages'):
                pre_call_messages = self.messages.copy()
            else:
                # Fallback for any other case
                pre_call_messages = []
        except Exception:
            # If anything goes wrong, just use an empty list
            pre_call_messages = []
        
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Record the start time
        start_time = time.time()
        
        # Call the original function
        response = func(self, prompt, *args, **kwargs)
        
        # Calculate execution time
        exec_time = time.time() - start_time
        
        # If the function returned an error, skip cost tracking
        if isinstance(response, str) and response.startswith(("Error:", "Unexpected error:")):
            return response
            
        # Prepare messages for token counting
        messages_for_tracking = pre_call_messages.copy()
        if len(messages_for_tracking) == 0 or messages_for_tracking[-1].get('role') != 'user':
            messages_for_tracking.append({"role": "user", "content": prompt})
        
        # Extract text response if logprobs were returned
        response_text = response.get('content', response) if isinstance(response, dict) else response
        
        # Build metadata
        metadata = {
            "run_id": CURRENT_RUN_ID,
            "execution_time": exec_time,
            "function": func.__name__,
            "request_id": request_id,
        }
        
        # Add parameters used in the call
        for k, v in kwargs.items():
            if k in ['temperature', 'max_tokens', 'top_p', 'return_json', 'cached', 'use_cache']:
                metadata[k] = v
                
        # Get the call location if possible
        try:
            import traceback
            frames = traceback.extract_stack()
            if len(frames) >= 3:
                # Skip this function and the chat function
                caller = frames[-3]
                metadata["caller"] = f"{caller.filename.split('/')[-1]}:{caller.lineno}"
        except Exception:
            pass
            
        # Track the cost
        try:
            # Get the model name from the right place
            if isinstance(self, LLM) and hasattr(self, 'model'):
                model = self.model
            else:
                model = getattr(self, 'model', 'unknown')
                
            # Check if this model is in our cost database
            if model not in MODEL_COSTS and model not in UNKNOWN_MODELS:
                UNKNOWN_MODELS.add(model)
                print(f"⚠️ Warning: No cost information for model '{model}'. Using fallback approximation.")
                # Save a list of unknown models for reference
                with open(os.path.join(LOG_DIR, 'unknown_models.txt'), 'a') as f:
                    f.write(f"{model}\n")
                
            # Check if this is a cached request (for models that support caching)
            cached = False
            if kwargs.get('cached') or kwargs.get('use_cache') or is_cached_request(request_id, metadata):
                cached = True
                metadata['is_cached'] = True
                
            # Track the cost
            cost_tracker.track_chat_completion(
                model=model,
                messages=messages_for_tracking,
                response=response_text,
                metadata=metadata,
                cached_input=cached
            )
        except Exception as e:
            # If something goes wrong with cost tracking, log the error but don't disrupt execution
            print(f"⚠️ Warning: Cost tracking error: {e}")
        
        return response
    
    return wrapper


# Patch the chat method of each LLM class to track costs
OpenAILLM.chat = track_chat_call(OpenAILLM.chat)
AnthropicLLM.chat = track_chat_call(AnthropicLLM.chat)
GroqLLM.chat = track_chat_call(GroqLLM.chat)
ReplicateLLM.chat = track_chat_call(ReplicateLLM.chat)
GeminiLLM.chat = track_chat_call(GeminiLLM.chat)
LLM.chat = track_chat_call(LLM.chat)


# Path to the consolidated cost history file
COST_HISTORY_FILE = os.path.join(LOG_DIR, 'cost_history.json')


def load_cost_history():
    """Load the consolidated cost history from a single JSON file."""
    if os.path.exists(COST_HISTORY_FILE):
        try:
            with open(COST_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Initialize new history if file doesn't exist or can't be read
    return {
        "runs": [],
        "total_cost": 0.0,
        "costs_by_model": {},
        "costs_by_date": {},
        "model_costs": MODEL_COSTS  # Store current model costs for reference
    }


def save_cost_history(history):
    """Save the consolidated cost history to a single JSON file."""
    with open(COST_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def update_cost_history():
    """
    Update the consolidated cost history with the current run data.
    This creates a single entry in the history for this run.
    """
    # Load existing history
    history = load_cost_history()
    
    # Get current session data
    current_costs = cost_tracker.session_costs
    
    # Calculate the run duration
    run_end_time = datetime.datetime.now().isoformat()
    run_duration = (datetime.datetime.fromisoformat(run_end_time) - 
                   datetime.datetime.fromisoformat(RUN_START_TIME)).total_seconds()
    
    # Create a run summary
    run_summary = {
        "run_id": CURRENT_RUN_ID,
        "start_time": RUN_START_TIME,
        "end_time": run_end_time,
        "duration_seconds": run_duration,
        "total_cost": current_costs["total"],
        "costs_by_model": current_costs["by_model"],
        "call_count": len(current_costs["by_call"]),
        "metadata": RUN_METADATA
    }
    
    # Calculate the difference from previous state
    previous_total = history["total_cost"]
    previous_by_model = history.get("costs_by_model", {})
    
    # Add the differences to the run summary
    run_summary["diff"] = {
        "total_cost": current_costs["total"],
        "costs_by_model": {}
    }
    
    for model, cost in current_costs["by_model"].items():
        previous_cost = previous_by_model.get(model, 0.0)
        run_summary["diff"]["costs_by_model"][model] = cost
    
    # Add this run to the history
    history["runs"].append(run_summary)
    
    # Update the totals
    history["total_cost"] += current_costs["total"]
    
    # Update costs by model
    for model, cost in current_costs["by_model"].items():
        if model not in history["costs_by_model"]:
            history["costs_by_model"][model] = 0.0
        history["costs_by_model"][model] += cost
    
    # Update costs by date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if today not in history["costs_by_date"]:
        history["costs_by_date"][today] = 0.0
    history["costs_by_date"][today] += current_costs["total"]
    
    # Save the updated history
    save_cost_history(history)
    
    return run_summary


def get_cost_report(detailed=False) -> str:
    """Generate a cost report for all API calls made so far."""
    return cost_tracker.generate_report(detailed=detailed)


def get_total_cost() -> float:
    """Get the total cost of all API calls made so far."""
    return cost_tracker.cost_log["total_cost"]


def get_costs_by_model() -> Dict[str, float]:
    """Get a breakdown of costs by model."""
    return cost_tracker.cost_log["costs_by_model"]


def get_session_costs() -> Dict[str, Any]:
    """Get costs for the current session."""
    return cost_tracker.session_costs


def get_run_report() -> Dict[str, Any]:
    """Get a report for the current run, including diff from previous state."""
    return update_cost_history()


def get_cost_history() -> Dict[str, Any]:
    """Get the full cost history from all runs."""
    return load_cost_history()


def get_unknown_models() -> List[str]:
    """Get the list of models encountered without pricing information."""
    return list(UNKNOWN_MODELS)


# Initialize
try:
    # Update cost history immediately to register this run
    run_metadata = RUN_METADATA.copy()
    run_metadata["status"] = "started"
    
    # Load existing history
    history = load_cost_history()
    
    # Add this run to the history with initial data
    run_summary = {
        "run_id": CURRENT_RUN_ID,
        "start_time": RUN_START_TIME,
        "status": "started",
        "metadata": run_metadata
    }
    history["runs"].append(run_summary)
    
    # Save the updated history
    save_cost_history(history)
    
    # Check if all models in MODEL_COSTS.keys() that are missing
    print(f"✅ Cost tracking initialized. Logs will be saved to {COST_LOG_FILE}")
    print(f"📊 Consolidated cost history at {COST_HISTORY_FILE}")
except Exception as e:
    print(f"⚠️ Warning: Cost tracking initialization error: {e}") 