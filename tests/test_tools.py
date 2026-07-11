import pytest
import requests
from agent.tools import ToolRouter, SearchInput, SearchResult, FetchInput, FetchResult

def test_tool_router_validation_failures():
    router = ToolRouter(seed=42, force_mocks=True)
    
    # 1. Invalid input: missing query in search
    res = router.call("search", {"invalid_key": "some query"})
    assert res["error"] == "invalid_input"
    assert "query" in res["detail"]
    
    # 2. Invalid input: missing url in fetch
    res = router.call("fetch", {"link": "https://example.com"})
    assert res["error"] == "invalid_input"
    assert "url" in res["detail"]
    
    # 3. Unknown tool
    res = router.call("nonexistent_tool", {"url": "https://example.com"})
    assert res["error"] == "invalid_input"
    assert "Unknown tool name" in res["detail"]

def test_tool_router_exception_handling():
    router = ToolRouter(seed=42, force_mocks=False)
    
    # Dispatch tool that raises an exception by using an invalid/nonexistent URL
    # or just passing something that causes requests to throw ConnectionError.
    # Note: requests.get will raise an exception for invalid URLs like "invalid_url".
    res = router.call("fetch", {"url": "not-a-valid-url"})
    assert "error" in res
    assert res["error"] in ("tool_exception", "schema_mismatch")

def test_search_tool_happy_path():
    router = ToolRouter(seed=42, force_mocks=True)
    res = router.call("search", {"query": "eiffel tower height"})
    
    assert "results" in res
    assert len(res["results"]) > 0
    assert res["results"][0]["title"] == "Eiffel Tower Height - Official Site"
    assert "330 meters" in res["results"][0]["snippet"]

def test_fetch_tool_happy_path():
    router = ToolRouter(seed=42, force_mocks=True)
    url = "https://official.eiffel.tower/height"
    res = router.call("fetch", {"url": url})
    
    assert res["url"] == url
    assert "official height of the eiffel tower" in res["text"].lower()
    assert "fetched_at" in res

def test_flaky_fetch_modes():
    # We want to verify that flaky_fetch goes through all three paths.
    # We will instantiate several routers with different seeds or invoke it in a loop
    # with a non-force_mocks or force_mocks router.
    router = ToolRouter(seed=12345, force_mocks=True)
    url = "https://official.eiffel.tower/height"
    
    outcomes = {"timeout": 0, "garbage": 0, "success": 0}
    
    # Run 50 times to hit all random paths
    for _ in range(50):
        res = router.call("flaky_fetch", {"url": url})
        if "error" in res:
            if "Timeout" in res["detail"]:
                outcomes["timeout"] += 1
        elif "Error 403 Access Denied" in res["text"]:
            outcomes["garbage"] += 1
        elif "330 meters" in res["text"]:
            outcomes["success"] += 1
            
    # Verify that all three paths were exercised at least once
    assert outcomes["timeout"] > 0, f"No timeout occurred: {outcomes}"
    assert outcomes["garbage"] > 0, f"No garbage response occurred: {outcomes}"
    assert outcomes["success"] > 0, f"No successful fetch occurred: {outcomes}"
