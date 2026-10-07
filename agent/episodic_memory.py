import os
import json
import uuid
import math
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EpisodicRecord(BaseModel):
    episode_id: str = Field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:8]}")
    goal_pattern: str
    error_type: str
    failed_action: str
    successful_resolution: str
    learned_rule: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _tokenize(text: str) -> List[str]:
    """Simple tokenization helper for similarity matching."""
    import re
    return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 1]


def _cosine_similarity(text1: str, text2: str) -> float:
    """Calculates TF-IDF / term frequency cosine similarity between two text strings."""
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)
    if not tokens1 or not tokens2:
        return 0.0

    tf1 = Counter(tokens1)
    tf2 = Counter(tokens2)
    all_terms = set(tf1.keys()).union(set(tf2.keys()))

    dot_product = sum(tf1[t] * tf2[t] for t in all_terms)
    mag1 = math.sqrt(sum(v * v for v in tf1.values()))
    mag2 = math.sqrt(sum(v * v for v in tf2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


class EpisodicMemoryStore:
    """Cross-session vectorized reflection store for recording and querying past error resolutions."""

    def __init__(self, storage_path: str = "logs/episodic_memory.json"):
        self.storage_path = storage_path
        self.records: List[EpisodicRecord] = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.records = [EpisodicRecord.model_validate(item) for item in data]
            except Exception:
                self.records = []

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in self.records], f, indent=2)

    def add_episode(self, record: EpisodicRecord):
        self.records.append(record)
        self._save()

    def record_resolution(
        self,
        goal_pattern: str,
        error_type: str,
        failed_action: str,
        successful_resolution: str,
        learned_rule: str,
    ) -> EpisodicRecord:
        record = EpisodicRecord(
            goal_pattern=goal_pattern,
            error_type=error_type,
            failed_action=failed_action,
            successful_resolution=successful_resolution,
            learned_rule=learned_rule,
        )
        self.add_episode(record)
        return record

    def query_similar(self, query_text: str, top_k: int = 3) -> List[EpisodicRecord]:
        """Queries store for past recovery episodes matching query_text by token similarity."""
        if not self.records:
            return []

        scored_records = []
        for rec in self.records:
            corpus = f"{rec.goal_pattern} {rec.error_type} {rec.failed_action} {rec.learned_rule}"
            score = _cosine_similarity(query_text, corpus)
            scored_records.append((score, rec))

        # Sort descending by similarity score
        scored_records.sort(key=lambda x: x[0], reverse=True)

        # Return top_k records with score > 0 or all if matches
        results = [rec for score, rec in scored_records if score > 0.05][:top_k]
        if not results and self.records:
            # Fallback return recent top_k
            results = self.records[-top_k:]
        return results
