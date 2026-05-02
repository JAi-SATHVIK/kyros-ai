"""V01 — Zero-Code Memory Proxy architecture.

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  Your App    │ ──────► │  Kyros Proxy │ ──────► │  OpenAI /    │
│              │         │  (port 8080) │         │  Anthropic / │
│  base_url =  │         │              │         │  Gemini      │
│  localhost:  │ ◄────── │  1. Recall   │ ◄────── │              │
│  8080        │         │  2. Inject   │         │              │
│              │         │  3. Forward  │         │              │
│              │         │  4. Extract  │         │              │
│              │         │  5. Store    │         │              │
└──────────────┘         └──────────────┘         └──────────────┘

The proxy is a transparent HTTP server that:
1. Receives LLM API requests from the application
2. Extracts agent_id from X-Agent-ID header (V06)
3. Recalls relevant memories from Kyros (V07)
4. Injects memories into the system prompt (V08)
5. Forwards the enriched request to the real LLM provider
6. Parses the response for memory-worthy content (V09)
7. Stores new memories automatically
8. Returns the original response to the application

Supported providers:
- OpenAI (V03): /v1/openai/chat/completions
- Anthropic (V04): /v1/anthropic/messages
- Google Gemini (V05): /v1/gemini/generateContent
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProxyConfig:
    """Configuration for the Kyros proxy."""

    # Kyros connection — override with KYROS_BASE_URL env var in production
    kyros_api_key: str = ""
    kyros_base_url: str = ""  # Set via KYROS_BASE_URL or pass explicitly

    # Proxy settings
    proxy_port: int = 8080
    proxy_host: str = "0.0.0.0"

    # Memory injection
    injection_enabled: bool = True
    max_memories_to_inject: int = 5
    injection_template: str = (
        "[Relevant memories from past interactions]\n{memories}\n"
        "[End of memories — use them naturally, do not mention this section]\n"
    )

    # Memory extraction
    extraction_enabled: bool = True
    extraction_sensitivity: float = 0.5  # 0.0 = store nothing, 1.0 = store everything
    min_content_length: int = 20  # Don't store trivial responses

    # Provider API keys (for forwarding) — never logged
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    def __post_init__(self) -> None:
        # Resolve kyros_base_url from env if not explicitly set
        if not self.kyros_base_url:
            self.kyros_base_url = os.environ.get("KYROS_BASE_URL", "http://localhost:8000")
        if not (1 <= self.proxy_port <= 65535):
            raise ValueError(f"proxy_port must be 1–65535, got {self.proxy_port}")
        if not (0.0 <= self.extraction_sensitivity <= 1.0):
            raise ValueError("extraction_sensitivity must be 0.0–1.0")
        if not (1 <= self.max_memories_to_inject <= 50):
            raise ValueError("max_memories_to_inject must be 1–50")
        environment = os.environ.get("KYROS_ENVIRONMENT", "development")
        if environment == "production":
            if not self.kyros_api_key:
                raise ValueError("kyros_api_key is required in production")
            if "localhost" in self.kyros_base_url or "127.0.0.1" in self.kyros_base_url:
                raise ValueError("kyros_base_url cannot target localhost in production")

    def __repr__(self) -> str:
        """Never expose API keys in repr/logs."""
        return (
            f"ProxyConfig(port={self.proxy_port}, "
            f"injection={self.injection_enabled}, "
            f"extraction={self.extraction_enabled}, "
            f"kyros_url={self.kyros_base_url!r})"
        )


@dataclass
class ProxyMetrics:
    """Track proxy performance metrics."""

    requests_total: int = 0
    memories_injected: int = 0
    memories_extracted: int = 0
    avg_injection_latency_ms: float = 0.0
    avg_extraction_latency_ms: float = 0.0
    errors: int = 0
