from __future__ import annotations

"""Utilities for batched LLM requests with transparent hash-cache lookup and
cost-tracking support.

The public function exported here is ``batch_model_response`` which replicates
all behaviours of the existing ``model_response`` helper but sends every cache
MISS to the provider in **one** batched request (currently only Anthropic
Claude models are supported).  Successful responses are written back to the
same ``hash_cache`` directory so subsequent runs can hit the cache.
"""

from typing import List, Tuple, Dict, Optional, Any
import os
import pickle
import hashlib
import time

from src.llms import LLM
from src.utils import hash_cache
from src.cost_tracker import MODEL_COSTS

# ---------------------------------------------------------------------------
# Internal helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

_CACHE_DIR = "hash_cache"


def _make_cache_filename(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    cache_nonce: Any = None,
) -> str:
    """Re-implement the hashing logic of @hash_cache for *model_response*."""

    # The original decorator builds the key as:  [func_name, args, kwargs, nonce]
    args_tuple = (
        prompt,
        system_prompt,
        model,
        temperature,
        top_p,
        max_tokens,
    )
    key_parts = [
        "model_response",  # func.__name__ in evaluate_model.py
        args_tuple,
        {},  # kwargs after stripping extras
        cache_nonce,
    ]

    md5 = hashlib.md5(pickle.dumps(key_parts)).hexdigest()
    return os.path.join(_CACHE_DIR, f"{md5}.pkl")


def _read_cached(prompt, system_prompt, model, temperature, top_p, max_tokens):
    path = _make_cache_filename(prompt, system_prompt, model, temperature, top_p, max_tokens)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                resp, _sys_prompt = pickle.load(fh)
                return resp
        except Exception:
            # Corrupt cache file – ignore
            return None
    return None


def _write_cached(prompt, system_prompt, model, temperature, top_p, max_tokens, response):
    path = _make_cache_filename(prompt, system_prompt, model, temperature, top_p, max_tokens)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump((response, system_prompt), fh)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def batch_model_response(
    prompts: List[str],
    system_prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[str]:
    """Return assistant responses for *prompts* with caching and batch API.

    It first loads any cached responses; remaining prompts are sent in a single
    provider batch (currently only Claude models).  Fresh results are written
    back to the cache *and* logged via CostTracker when available.
    """

    assert len(prompts) > 0, "prompts list is empty"

    # 1. Check cache --------------------------------------------------------
    responses: List[Optional[str]] = [None] * len(prompts)
    uncached_indices: List[int] = []

    for i, p in enumerate(prompts):
        cached = _read_cached(p, system_prompt, model, temperature, top_p, max_tokens)
        if cached is not None:
            responses[i] = cached
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return [r for r in responses]  # All hits

    # 2. Send batch for misses ---------------------------------------------
    llm = LLM(model, system_prompt)

    # Only Anthropic models currently expose chat_batch; fallback to per-prompt.
    from src.llms import AnthropicLLM
    if isinstance(llm.llm, AnthropicLLM):
        batch_prompts = [prompts[i] for i in uncached_indices]
        batch_responses = llm.llm.chat_batch(
            batch_prompts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    else:
        # Fallback: sequential calls (rare – other providers unbatched anyway)
        batch_prompts = [prompts[i] for i in uncached_indices]
        batch_responses = [
            llm.chat(p, temperature=temperature, top_p=top_p, max_tokens=max_tokens)["content"]
            if isinstance(llm.chat(p, temperature=temperature, top_p=top_p, max_tokens=max_tokens), dict)
            else llm.chat(p, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            for p in batch_prompts
        ]

    # Sanity check
    assert len(batch_responses) == len(uncached_indices), "Mismatch in batch response length"

    # 3. Fill responses list & write cache ---------------------------------
    for local_idx, global_idx in enumerate(uncached_indices):
        resp_text = batch_responses[local_idx]
        responses[global_idx] = resp_text
        _write_cached(
            prompts[global_idx],
            system_prompt,
            model,
            temperature,
            top_p,
            max_tokens,
            resp_text,
        )

        # 4. Cost tracking per prompt (optional) ---------------------------
        # Cost tracking – fetch tracker at runtime to ensure we get the
        # instance created by wrap_llms_for_cost_tracking() *after* imports.
        try:
            import importlib
            cti = importlib.import_module("src.cost_tracking_integration")
            tracker = getattr(cti, "cost_tracker_instance", None)
            if tracker is not None:
                messages_for_tracking = [{"role": "user", "content": prompts[global_idx]}]
                tracker.track_chat_completion(
                    model=model,
                    messages=messages_for_tracking,
                    response=resp_text,
                    metadata={"batched": True},
                    cached_input=False,
                    usage=None,
                )
        except Exception:
            pass

    return [r for r in responses] 