import sys
from agent import orchestrator
from agent.tools import ToolRouter

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m baseline.run <goal>")
        sys.exit(1)
        
    goal = sys.argv[1]
    # In baseline, we can use real or mock tool routing. Default to force_mocks=False for production/real runs
    tools = ToolRouter(seed=42, force_mocks=False)
    
    print(f"Running baseline agent for goal: {goal}")
    mem, answer = orchestrator.run(goal, tools, use_self_correction=False)
    
    print("\n=== FINAL ANSWER ===")
    print(answer)
    print("====================")

if __name__ == "__main__":
    main()
