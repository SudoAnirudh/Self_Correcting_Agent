import pytest
from unittest.mock import patch, MagicMock
from agent import orchestrator
from agent.tools import ToolRouter
from agent.memory import WorkingMemory

@patch("agent.llm.call_llm")
def test_baseline_orchestrator_run(mock_call_llm):
    # Setup mocks for all LLM calls in order of execution:
    # 1. decompose: returns a plan with two subtasks
    # 2. reason (subtask 1): returns search action
    # 3. extract_facts (subtask 1): returns facts extracted
    # 4. reason (subtask 2): returns fetch action
    # 5. extract_facts (subtask 2): returns facts extracted
    # 6. synthesize: returns final answer
    
    mock_responses = [
        # 1. decompose
        '{"subtasks": [{"id": "s1", "description": "Search for Eiffel tower height", "status": "pending"}, {"id": "s2", "description": "Fetch official height info", "status": "pending"}]}',
        # 2. reason (s1)
        '{"belief": "Initial search", "gap": "Need Eiffel tower height url", "why_action": "Do a search", "action": "search", "action_input": {"query": "eiffel tower height"}}',
        # 3. extract_facts (s1)
        '{"official_url": "https://official.eiffel.tower/height"}',
        # 4. reason (s2)
        '{"belief": "Found URL", "gap": "Need details from URL", "why_action": "Fetch the URL", "action": "fetch", "action_input": {"url": "https://official.eiffel.tower/height"}}',
        # 5. extract_facts (s2)
        '{"eiffel_tower_height": "330 meters"}',
        # 6. synthesize
        "The official height of the Eiffel Tower is 330 meters."
    ]
    
    def side_effect(system, user, model, temperature=0.0):
        return mock_responses.pop(0)
        
    mock_call_llm.side_effect = side_effect
    
    # Initialize a mock-enabled ToolRouter
    tools = ToolRouter(seed=42, force_mocks=True)
    
    mem, final_answer = orchestrator.run(
        goal="official height of the eiffel tower",
        tools=tools,
        use_self_correction=False
    )
    
    assert final_answer == "The official height of the Eiffel Tower is 330 meters."
    assert mem.facts["eiffel_tower_height"] == "330 meters"
    assert mem.subtasks[0].status == "done"
    assert mem.subtasks[1].status == "done"
    assert len(mem.history) == 2
