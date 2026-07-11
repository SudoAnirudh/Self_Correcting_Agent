import time
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
import requests
from bs4 import BeautifulSoup
from agent.mock_data import MOCK_SEARCH, MOCK_FETCH

# Input & Output Schemas
class SearchInput(BaseModel):
    query: str

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

class SearchOutput(BaseModel):
    results: List[SearchResult]

class FetchInput(BaseModel):
    url: str

class FetchResult(BaseModel):
    url: str
    text: str
    fetched_at: str

# Helper matching functions
def match_mock_search(query: str) -> Optional[List[dict]]:
    query_lower = query.lower()
    # Try exact match or substring matches
    for key, val in MOCK_SEARCH.items():
        if key in query_lower or query_lower in key:
            return val
    return None

def match_mock_fetch(url: str) -> Optional[str]:
    # Exact match or find the URL that contains the mock URL
    for key, val in MOCK_FETCH.items():
        if key in url or url in key:
            return val
    return None

class ToolRouter:
    def __init__(self, seed: Optional[int] = None, force_mocks: bool = True):
        self.seed = seed if seed is not None else random.randint(0, 1000000)
        self.rng = random.Random(self.seed)
        self.force_mocks = force_mocks # force using mock database for determinism in evals

    def call(self, name: str, raw_input: dict) -> dict:
        """Entrypoint for all tool calls. Validates input, dispatches, validates output."""
        try:
            validated_input = self._validate_input(name, raw_input)
        except (ValidationError, ValueError) as e:
            return {"error": "invalid_input", "detail": str(e)}

        try:
            raw_result = self._dispatch(name, validated_input)
            # Check if result is a structured error dict from the tool itself
            if isinstance(raw_result, dict) and "error" in raw_result:
                return raw_result
        except Exception as e:
            return {"error": "tool_exception", "detail": f"{type(e).__name__}: {str(e)}"}

        try:
            return self._validate_output(name, raw_result)
        except (ValidationError, ValueError) as e:
            return {"error": "schema_mismatch", "raw": raw_result, "detail": str(e)}

    def _validate_input(self, name: str, raw_input: dict) -> Any:
        if name == "search":
            return SearchInput.model_validate(raw_input)
        elif name in ("fetch", "flaky_fetch"):
            return FetchInput.model_validate(raw_input)
        else:
            raise ValueError(f"Unknown tool name: {name}")

    def _validate_output(self, name: str, raw_result: Any) -> dict:
        if name == "search":
            return SearchOutput.model_validate(raw_result).model_dump()
        elif name in ("fetch", "flaky_fetch"):
            return FetchResult.model_validate(raw_result).model_dump()
        else:
            raise ValueError(f"Unknown tool name: {name}")

    def _dispatch(self, name: str, validated_input: Any) -> Any:
        if name == "search":
            return self._search_tool(validated_input.query)
        elif name == "fetch":
            return self._fetch_tool(validated_input.url)
        elif name == "flaky_fetch":
            return self._flaky_fetch_tool(validated_input.url)

    def _search_tool(self, query: str) -> dict:
        # Check mock database first
        mock_res = match_mock_search(query)
        if self.force_mocks and mock_res:
            return {"results": mock_res}

        # Real DuckDuckGo Search
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            # Quote query safely
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            results = []
            
            # Extract search hits
            for item in soup.find_all("div", class_="result")[:3]:
                title_el = item.find("a", class_="result__a")
                snippet_el = item.find("a", class_="result__snippet")
                if title_el and snippet_el:
                    results.append({
                        "title": title_el.get_text(strip=True),
                        "url": title_el["href"],
                        "snippet": snippet_el.get_text(strip=True)
                    })
            
            if results:
                return {"results": results}
        except Exception as e:
            # If real request fails, fall back to mock database if available
            if mock_res:
                return {"results": mock_res}
            raise e

        # If real request returned empty results, check mock database
        if mock_res:
            return {"results": mock_res}
        return {"results": []}

    def _fetch_tool(self, url: str) -> dict:
        # Check mock database first
        mock_text = match_mock_fetch(url)
        if self.force_mocks and mock_text:
            return {
                "url": url,
                "text": mock_text,
                "fetched_at": datetime.now(timezone.utc).isoformat()
            }

        # Real Web Fetch
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, "html.parser")
            for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                script.decompose()
            text = soup.get_text(separator=" ", strip=True)
            # Truncate text to limit context window bloating
            cleaned_text = text[:3000]
            
            return {
                "url": url,
                "text": cleaned_text,
                "fetched_at": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            # Fall back to mock if available
            if mock_text:
                return {
                    "url": url,
                    "text": mock_text,
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                }
            raise e

    def _flaky_fetch_tool(self, url: str) -> dict:
        r = self.rng.random()
        
        # 15% probability of timeout exception
        if r < 0.15:
            raise requests.exceptions.Timeout("Connection timed out (simulated flaky fetch exception)")
            
        # 30% probability of garbage success
        elif r < 0.45:
            return {
                "url": url,
                "text": "Error 403 Access Denied. You do not have permission to access / on this server. <html><body>Some truncated garbage text and noise pattern...</html>",
                "fetched_at": datetime.now(timezone.utc).isoformat()
            }
            
        # 55% probability of normal success
        else:
            return self._fetch_tool(url)
