import os
import json
import time
import argparse
from agent import orchestrator
from agent.tools import ToolRouter
from agent import llm
from eval.chaos_matrix import ChaosToolRouter, ChaosConfig

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
        # Fallback keyword checking if LLM fails
        return expected.lower() in actual.lower()

def main():
    parser = argparse.ArgumentParser(description="Run Self-Correction Benchmark Evaluation")
    parser.add_argument("--chaos", action="store_true", help="Enable adversarial chaos fault injection matrix")
    args = parser.parse_args()

    print("==================================================")
    print(f"Starting Self-Correction Evaluation {'[CHAOS MODE ENABLED]' if args.chaos else ''}")
    print("==================================================")
    
    with open("eval/goals.json", "r", encoding="utf-8") as f:
        goals = json.load(f)
        
    results = []
    total_chaos_faults = 0
    
    for i, item in enumerate(goals, 1):
        goal_id = item["id"]
        description = item["description"]
        expected = item["expected"]
        
        print(f"\n[{i}/10] Goal: '{description}'")
        
        # --- 1. RUN BASELINE ---
        print("  Running Baseline Mode...")
        tools_baseline = ToolRouter(seed=12345, force_mocks=True)
        start_t = time.time()
        mem_base, ans_base = orchestrator.run(description, tools_baseline, use_self_correction=False)
        dur_base = time.time() - start_t
        is_correct_base = grade_answer(description, expected, ans_base)
        
        # --- 2. RUN SELF-CORRECTING ---
        print("  Running Self-Correcting Mode...")
        if args.chaos:
            tools_sc = ChaosToolRouter(seed=12345, force_mocks=True, chaos_config=ChaosConfig(chaos_rate=0.3, seed=12345))
        else:
            tools_sc = ToolRouter(seed=12345, force_mocks=True)

        start_t = time.time()
        mem_sc, ans_sc = orchestrator.run(description, tools_sc, use_self_correction=True)
        dur_sc = time.time() - start_t
        is_correct_sc = grade_answer(description, expected, ans_sc)
        
        if args.chaos:
            total_chaos_faults += tools_sc.injected_faults_count

        # Count unresolved
        unresolved_base = [t for t in mem_base.subtasks if t.status not in ("done", "completed", "unresolvable")]
        unresolved_sc = [t for t in mem_sc.subtasks if t.status not in ("done", "completed", "unresolvable")]
        
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
                "unresolved_subtasks": len(unresolved_sc),
                "chaos_faults_injected": getattr(tools_sc, "injected_faults_count", 0)
            }
        }
        results.append(goal_results)
        
        print(f"  Baseline:        {'PASS' if is_correct_base else 'FAIL'} | Steps: {goal_results['baseline']['steps']} | Time: {dur_base:.2f}s")
        print(f"  Self-Correcting: {'PASS' if is_correct_sc else 'FAIL'} | Steps: {goal_results['self_correcting']['steps']} | Recoveries: {goal_results['self_correcting']['recoveries']} | Time: {dur_sc:.2f}s")
        if args.chaos:
            print(f"    [Chaos] Injected Faults: {tools_sc.injected_faults_count} | Breakdown: {tools_sc.fault_breakdown}")
        
    # Save results summary to logs/eval_results.json
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n==================================================")
    print("Evaluation Completed. Summary saved to logs/eval_results.json")
    if args.chaos:
        print(f"Total Chaos Faults Injected across 10 goals: {total_chaos_faults}")
    print("==================================================")

if __name__ == "__main__":
    main()

