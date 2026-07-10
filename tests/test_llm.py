import pytest
import json
from unittest.mock import patch, MagicMock
from agent import llm

@patch("agent.llm.requests.post")
def test_decompose(mock_post):
    # Set up mock response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "subtasks": [
                            {"id": "s1", "description": "Search for capital of France", "status": "pending"}
                        ]
                    })
                }
            }
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp
    
    result = llm.decompose("What is the capital of France?")
    assert "subtasks" in result
    assert result["subtasks"][0]["id"] == "s1"
    
@patch("agent.llm.requests.post")
def test_reason(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "belief": "I know Paris is the capital.",
                        "gap": "Need details.",
                        "why_action": "Check wikipedia.",
                        "action": "fetch",
                        "action_input": {"url": "https://wikipedia.org"}
                    })
                }
            }
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp
    
    snapshot = {"goal": "Test", "facts": {}, "summary_log": [], "history": []}
    subtask = {"id": "s1", "description": "Fetch wiki"}
    
    result = llm.reason(snapshot, subtask)
    assert result["action"] == "fetch"
    assert result["action_input"] == {"url": "https://wikipedia.org"}

@patch("agent.llm.requests.post")
def test_evaluate(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "verdict": "success",
                        "reasoning": "Correct page fetched."
                    })
                }
            }
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp
    
    result = llm.evaluate(
        goal="Test goal",
        subtask_desc="Fetch wiki",
        record_data={"action": "fetch", "action_input": {"url": "https://wikipedia.org"}, "action_result": "Success"},
        facts_summary={}
    )
    assert result["verdict"] == "success"
    assert result["reasoning"] == "Correct page fetched."
