import os
import json
import time
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Constants
REASONING_MODEL = "llama-3.3-70b-versatile"
EVALUATION_MODEL = "llama-3.1-8b-instant"

# Load environment variables from .env if present
def load_dotenv():
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_dotenv()

def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
    return key

def call_llm(system: str, user: str, model: str, temperature: float = 0.0) -> str:
    """Wrapper that calls the Groq Chat Completions API and logs raw prompt + response."""
    api_key = get_api_key()
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"} if "json" in system.lower() or "json" in user.lower() else None
    }
    
    # Track duration
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        resp_json = response.json()
        content = resp_json["choices"][0]["message"]["content"]
    except Exception as e:
        content = f"Error during LLM API call: {str(e)}"
        raise e
    finally:
        duration = time.time() - start_time
        # Log to logs/llm_calls.log
        os.makedirs("logs", exist_ok=True)
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "temperature": temperature,
            "system_prompt": system,
            "user_prompt": user,
            "response": content,
            "duration_seconds": duration
        }
        with open("logs/llm_calls.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    return content

# 1. Planner Decompose
def decompose(goal: str) -> dict:
    system = (
        "You are an expert research planner. Decompose the user's research goal into a sequence of concrete, atomic subtasks. "
        "Each subtask must represent a single search or fetch/extract step. "
        "Output JSON ONLY in this format:\n"
        "{\n"
        '  "subtasks": [\n'
        '    {"id": "s1", "description": "Search for X", "status": "pending"},\n'
        '    {"id": "s2", "description": "Fetch page Y to extract details", "status": "pending"}\n'
        "  ]\n"
        "}"
    )
    user = f"Goal: {goal}"
    res = call_llm(system, user, REASONING_MODEL, temperature=0.0)
    try:
        return json.loads(res)
    except json.JSONDecodeError:
        # Fallback empty list if parsing failed
        return {"subtasks": []}

# 2. ReAct Reasoner
def reason(memory_snapshot: dict, current_subtask: dict) -> dict:
    system = (
        "You are a ReAct research agent. You must determine the next action to perform. "
        "To ensure your reasoning is rigorous, you MUST state the following fields in order:\n"
        "1. belief: What you currently believe about the progress toward the goal.\n"
        "2. gap: What is still missing to complete the current subtask.\n"
        "3. why_action: Why the chosen action directly addresses the gap.\n"
        "4. action: The tool to call (must be exactly 'search', 'fetch', or 'flaky_fetch').\n"
        "5. action_input: A dictionary of inputs for the tool (e.g. {'query': '...'} for search, {'url': '...'} for fetch/flaky_fetch).\n\n"
        "You MUST output JSON ONLY in the following format:\n"
        "{\n"
        '  "belief": "current state of knowledge...",\n'
        '  "gap": "what is missing...",\n'
        '  "why_action": "justification for the action...",\n'
        '  "action": "search" | "fetch" | "flaky_fetch",\n'
        '  "action_input": {"query": "..."} | {"url": "..."}\n'
        "}"
    )
    user = (
        f"Goal: {memory_snapshot['goal']}\n"
        f"Active Subtask: {json.dumps(current_subtask)}\n"
        f"Prior facts: {json.dumps(memory_snapshot['facts'])}\n"
        f"Summary Log: {json.dumps(memory_snapshot['summary_log'])}\n"
        f"Recent History: {json.dumps(memory_snapshot['history'])}\n"
    )
    res = call_llm(system, user, REASONING_MODEL, temperature=0.0)
    try:
        return json.loads(res)
    except json.JSONDecodeError:
        return {
            "belief": "Failed to parse reasoning",
            "gap": "JSON parse error",
            "why_action": "Fallback due to parse error",
            "action": "search",
            "action_input": {"query": "parse_failure_fallback"}
        }

# 3. Step Self-Evaluator
def evaluate(goal: str, subtask_desc: str, record_data: dict, facts_summary: dict) -> dict:
    system = (
        "You are a critical agent self-evaluator. Your job is to verify if the latest tool result was successful "
        "and moves the agent closer to the goal. You must answer as JSON only.\n\n"
        "JSON Format:\n"
        "{\n"
        '  "verdict": "success" | "tool_failure" | "inconsistent" | "goal_drift",\n'
        '  "reasoning": "A one or two sentence explanation of the verdict."\n'
        "}\n\n"
        "Evaluation Rules:\n"
        "1. If the result contains an 'error' key or indicates an HTTP error (e.g., Access Denied, Forbidden, Timeout), verdict is 'tool_failure'.\n"
        "2. If the result contradicts an established fact in Prior Confirmed Facts, verdict is 'inconsistent'.\n"
        "3. If the result is well-formed but does not help resolve the subtask or deviates from the original goal, verdict is 'goal_drift'.\n"
        "4. Otherwise, verdict is 'success'."
    )
    
    user = (
        f"Original Goal: {goal}\n"
        f"Subtask Description: {subtask_desc}\n"
        f"Action: {record_data['action']}({json.dumps(record_data['action_input'])})\n"
        f"Action Result: {json.dumps(record_data['action_result'])}\n"
        f"Prior Confirmed Facts: {json.dumps(facts_summary)}"
    )
    
    res = call_llm(system, user, EVALUATION_MODEL, temperature=0.0)
    try:
        return json.loads(res)
    except json.JSONDecodeError:
        return {
            "verdict": "tool_failure",
            "reasoning": "Evaluator output failed to parse as JSON."
        }

# 4. Final Synthesizer
def synthesize(goal: str, facts: dict, summary_log: List[str], unresolved_subtasks: List[dict]) -> str:
    system = (
        "You are an expert synthesizer. Summarize the findings to answer the original goal. "
        "Integrate all confirmed facts. If some subtasks were unresolvable or had inconsistent facts, "
        "you MUST explicitly state the low-confidence aspects and flag the gaps. Do not hide failures."
    )
    user = (
        f"Goal: {goal}\n"
        f"Confirmed Facts: {json.dumps(facts)}\n"
        f"Summary Log: {json.dumps(summary_log)}\n"
        f"Unresolved Subtasks: {json.dumps(unresolved_subtasks)}"
    )
    return call_llm(system, user, REASONING_MODEL, temperature=0.3)
