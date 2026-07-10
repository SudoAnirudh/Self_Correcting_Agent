import os
import json
import time
from agent import orchestrator
from agent.tools import ToolRouter
from agent import llm

def grade_answer(goal: str, expected: str, actual: str) -> bool:
    """Grades the actual answer against the expected reference using Groq."""
    system = (
        "You are an objective evaluation grader. Compare the agent's actual output against the expected "
        "reference answer for the research goal. Determine if the actual output contains the correct facts "
        "and successfully answers the goal.\n\n"
        "Output JSON ONLY in this format:\n"
        "{\n"
        '  "correct": true | false,\n'
        '  "reasoning": "A short explanation of the grade."\n'
        "}"
    )
    user = (
        f"Goal: {goal}\n"
        f"Expected Reference: {expected}\n"
        f"Actual Output: {actual}"
    )
    try:
        res = llm.call_llm(system, user, llm.EVALUATION_MODEL, temperature=0.0)
        data = json.loads(res)
        return bool(data.get("correct", False))
    except Exception as e:
        print(f"Error grading answer: {e}")
        # Fallback keyword checking if LLM fails
        return expected.lower() in actual.lower()

def main():
    print("==================================================")
    print("Starting Antigravity Self-Correction Evaluation")
    print("==================================================")
    
    with open("eval/goals.json", "r", encoding="utf-8") as f:
        goals = json.load(f)
        
    results = []
    
    for i, item in enumerate(goals, 1):
        goal_id = item["id"]
        description = item["description"]
        expected = item["expected"]
        
        print(f"\n[{i}/10] Goal: '{description}'")
        
        # --- 1. RUN BASELINE ---
        print("  Running Baseline Mode...")
        # Clean / force mocks for reproducible eval
        tools_baseline = ToolRouter(seed=12345, force_mocks=True)
        start_t = time.time()
        mem_base, ans_base = orchestrator.run(description, tools_baseline, use_self_correction=False)
        dur_base = time.time() - start_t
        is_correct_base = grade_answer(description, expected, ans_base)
        
        # --- 2. RUN SELF-CORRECTING ---
        print("  Running Self-Correcting Mode...")
        tools_sc = ToolRouter(seed=12345, force_mocks=True)
        start_t = time.time()
        mem_sc, ans_sc = orchestrator.run(description, tools_sc, use_self_correction=True)
        dur_sc = time.time() - start_t
        is_correct_sc = grade_answer(description, expected, ans_sc)
        
        # Count unresolved
        unresolved_base = [t for t in mem_base.subtasks if t.status not in ("done", "unresolvable")]
        unresolved_sc = [t for t in mem_sc.subtasks if t.status not in ("done", "unresolvable")]
        
        goal_results = {
            "id": goal_id,
            "goal": description,
            "expected": expected,
            "baseline": {
                "answer": ans_base,
                "steps": len(mem_base.history),
                "duration_seconds": dur_base,
                "correct": is_correct_base,
                "recoveries": mem_base.global_recovery_attempts,
                "unresolved_subtasks": len(unresolved_base)
            },
            "self_correcting": {
                "answer": ans_sc,
                "steps": len(mem_sc.history),
                "duration_seconds": dur_sc,
                "correct": is_correct_sc,
                "recoveries": mem_sc.global_recovery_attempts,
                "unresolved_subtasks": len(unresolved_sc)
            }
        }
        results.append(goal_results)
        
        print(f"  Baseline:        {'PASS' if is_correct_base else 'FAIL'} | Steps: {goal_results['baseline']['steps']} | Time: {dur_base:.2f}s")
        print(f"  Self-Correcting: {'PASS' if is_correct_sc else 'FAIL'} | Steps: {goal_results['self_correcting']['steps']} | Recoveries: {goal_results['self_correcting']['recoveries']} | Time: {dur_sc:.2f}s")
        
    # Save results summary to logs/eval_results.json
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n==================================================")
    print("Evaluation Completed. Summary saved to logs/eval_results.json")
    print("==================================================")

if __name__ == "__main__":
    main()
