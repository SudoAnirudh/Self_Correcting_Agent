import random
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from agent.tools import ToolRouter


class ChaosFaultType(str, Enum):
    SCHEMA_MUTATION = "SCHEMA_MUTATION"
    TRANSIENT_NETWORK_ERROR = "TRANSIENT_NETWORK_ERROR"
    DEGRADED_PAYLOAD = "DEGRADED_PAYLOAD"
    NOISY_CONTEXT = "NOISY_CONTEXT"


class ChaosConfig(BaseModel):
    chaos_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    fault_types: List[ChaosFaultType] = Field(
        default_factory=lambda: [
            ChaosFaultType.SCHEMA_MUTATION,
            ChaosFaultType.TRANSIENT_NETWORK_ERROR,
            ChaosFaultType.DEGRADED_PAYLOAD,
            ChaosFaultType.NOISY_CONTEXT,
        ]
    )
    seed: Optional[int] = None


class ChaosToolRouter(ToolRouter):
    """ToolRouter wrapper that stochastically injects adversarial faults into tool responses."""

    def __init__(
        self,
        seed: Optional[int] = None,
        force_mocks: bool = True,
        chaos_config: Optional[ChaosConfig] = None,
    ):
        super().__init__(seed=seed, force_mocks=force_mocks)
        self.chaos_config = chaos_config or ChaosConfig(seed=seed)
        self.chaos_rng = random.Random(self.chaos_config.seed or self.seed)
        self.injected_faults_count: int = 0
        self.fault_breakdown: Dict[str, int] = {f.value: 0 for f in ChaosFaultType}
        self.deterministic_stage_1_recoveries: int = 0
        self.llm_stage_2_recoveries: int = 0

    def call(self, name: str, raw_input: dict) -> dict:
        # Determine whether to inject a chaos fault
        if self.chaos_rng.random() < self.chaos_config.chaos_rate:
            fault = self.chaos_rng.choice(self.chaos_config.fault_types)
            self.injected_faults_count += 1
            self.fault_breakdown[fault.value] += 1

            if fault == ChaosFaultType.TRANSIENT_NETWORK_ERROR:
                err_choice = self.chaos_rng.choice(["rate_limit_429", "server_error_500", "timeout"])
                if err_choice == "rate_limit_429":
                    return {"error": "http_429_rate_limit", "status_code": 429, "detail": "Rate limit exceeded (simulated chaos)"}
                elif err_choice == "server_error_500":
                    return {"error": "http_500_server_error", "status_code": 500, "detail": "Internal server error (simulated chaos)"}
                else:
                    return {"error": "timeout", "detail": "Connection timed out (simulated chaos)"}

            elif fault == ChaosFaultType.DEGRADED_PAYLOAD:
                degraded_choice = self.chaos_rng.choice(["empty_dict", "null_payload", "truncated_text"])
                if degraded_choice == "empty_dict":
                    return {}
                elif degraded_choice == "null_payload":
                    return {"status": None, "results": None}
                else:
                    return {"url": raw_input.get("url", "http://example.com"), "text": "Err"}

            elif fault == ChaosFaultType.SCHEMA_MUTATION:
                return {"unexpected_corrupted_key": 12345, "status": "corrupted"}

            elif fault == ChaosFaultType.NOISY_CONTEXT:
                # Perform normal tool execution then inject distractor/noisy content into result
                real_res = super().call(name, raw_input)
                if isinstance(real_res, dict):
                    res_copy = dict(real_res)
                    res_copy["distractor_noise"] = "CONTRADICTORY FACT: The sky is green and Paris is in Asia."
                    return res_copy

        return super().call(name, raw_input)
