import os
from collections import Counter


# -----------------------------------------------------------------------------
# API KEYS
# -----------------------------------------------------------------------------

def get_openrouter_api_key() -> str:
    """
    Return the API key for OpenRouter usage.

    Env var precedence:
    - OPENROUTER_API_KEY (preferred)
    - OPENAI_API_KEY (fallback; common name for OpenAI-compatible clients)
    """
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "Missing API key. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY."
        )
    return key

def get_together_api_key() -> str:
    """Return Together API key from TOGETHER_API_KEY env var."""
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise EnvironmentError("Missing Together API key. Set TOGETHER_API_KEY.")
    return key

# -----------------------------------------------------------------------------
# Error tracking (per Python process, resettable)
# -----------------------------------------------------------------------------
_LLM_ERROR_COUNTS: Counter = Counter()

def reset_llm_error_counts() -> None:
    """Reset in-process counters for LLM call errors/retries/successes."""
    _LLM_ERROR_COUNTS.clear()

def get_llm_error_counts() -> dict:
    """Get a snapshot of in-process counters for LLM call errors/retries/successes."""
    return dict(_LLM_ERROR_COUNTS)

def _count_llm_event(name: str) -> None:
    # Internal helper: increment a counter key.
    _LLM_ERROR_COUNTS[name] += 1

# Messages: yes contains messages
# def call_llm(client, model, temperature, messages, logprobs=True, top_logprobs=5):
#     """Call LLM and get response"""
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#         logprobs=logprobs,
#         top_logprobs=top_logprobs
#     )
#     return response.choices[0].message.content

import time
import random
from openai import APIError, RateLimitError, APIConnectionError

def call_llm(client, model, temperature, messages, logprobs=True, top_logprobs=5, max_retries=3):
    """
    Call LLM with retry logic and error handling.
    
    Args:
        client: OpenAI client instance
        model: Model name
        temperature: Temperature setting
        messages: Conversation messages
        logprobs: Whether to return logprobs
        top_logprobs: Number of top logprobs
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        str: LLM response content
    
    Raises:
        Exception: If all retries fail or non-retryable error occurs
    """
    _count_llm_event("call_llm.calls")
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs
            )
            
            # Check if response is valid
            if not response.choices or not response.choices[0].message.content:
                _count_llm_event("call_llm.empty_response")
                raise ValueError("Empty response from LLM")
            
            _count_llm_event("call_llm.success")
            return response.choices[0].message.content
            
        except RateLimitError as e:
            _count_llm_event("error.RateLimitError")
            # Rate limit: wait longer, then retry
            wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
            if attempt < max_retries - 1:
                _count_llm_event("call_llm.retry")
                print(f"⚠️  Rate limit hit. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ Rate limit error after {max_retries} attempts: {e}")
                raise
        
        except APIConnectionError as e:
            _count_llm_event("error.APIConnectionError")
            # Network error: retry with backoff
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            if attempt < max_retries - 1:
                _count_llm_event("call_llm.retry")
                print(f"⚠️  Connection error. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ Connection error after {max_retries} attempts: {e}")
                raise
        
        except APIError as e:
            _count_llm_event("error.APIError")
            # Other API errors: check if retryable
            if e.status_code and e.status_code >= 500:
                # Server error: retry
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                if attempt < max_retries - 1:
                    _count_llm_event("call_llm.retry")
                    print(f"⚠️  Server error ({e.status_code}). Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Server error after {max_retries} attempts: {e}")
                    raise
            else:
                # Client error (4xx): don't retry
                print(f"❌ Client error ({e.status_code}): {e}")
                raise
        
        except Exception as e:
            _count_llm_event(f"error.{type(e).__name__}")
            # Unexpected errors: log and raise
            print(f"❌ Unexpected error in LLM call: {type(e).__name__}: {e}")
            raise
    
    # Should never reach here, but just in case
    raise Exception(f"Failed to get LLM response after {max_retries} attempts")

def print_simulation_header(game, num_turns, num_agents, memory_capacity, agent_biases):
    """Print simulation configuration header"""
    print("=" * 80)
    print(f"SIMULATION: {game.__class__.__name__}")
    print("=" * 80)
    print(f"Number of rounds: {num_turns}")
    print(f"Number of agents: {num_agents}")
    print(f"Memory capacity: {memory_capacity}")
    if agent_biases:
        print(f"Agent biases: {agent_biases}")
    print(f"Role swapping: ENABLED (agents alternate roles each round)")
    print("-" * 80)

def print_llm_error_summary() -> None:
    """
    Print a compact summary of LLM-related errors seen in this Python process
    (typically reset per run via reset_llm_error_counts()).
    """
    counts = get_llm_error_counts()
    # Only show interesting keys
    keys = sorted(k for k in counts.keys() if k.startswith("error.") or k.startswith("call_llm."))
    print("=" * 80)
    print("LLM ERROR SUMMARY")
    print("=" * 80)
    if not keys:
        print("No LLM calls recorded.")
        print("-" * 80)
        return
    for k in keys:
        print(f"{k}: {counts[k]}")
    print("-" * 80)
