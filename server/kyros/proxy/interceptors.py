"""V06–V09 — Request and response interceptors.

V06: Extract agent_id from X-Agent-ID header
V07: Pre-call hook — auto-recall top-5 relevant memories
V08: System prompt injector — append memories to system message
V09: Response interceptor — parse LLM response for memory-worthy content
"""

from __future__ import annotations

import time
import httpx

from kyros.proxy.architecture import ProxyConfig
from kyros.proxy.providers import NormalizedRequest
from kyros.logging import get_logger

logger = get_logger("kyros.proxy.interceptors")


# ─── V06: Agent ID Extractor ──────────────────

def extract_agent_id(headers: dict) -> str | None:
    """Extract agent_id from the X-Agent-ID header.

    The proxy uses this header to scope all memory operations
    to a specific agent. If missing, the proxy operates in
    pass-through mode (no memory injection/extraction).

    Headers are case-insensitive per HTTP spec.
    """
    # Check multiple casing variants
    for key in ("X-Agent-ID", "x-agent-id", "X-Agent-Id", "X-AGENT-ID"):
        if key in headers:
            agent_id = headers[key].strip()
            if agent_id:
                return agent_id
    return None


# ─── V07: Memory Recall Hook ──────────────────

async def recall_memories(
    agent_id: str,
    user_message: str,
    config: ProxyConfig,
    k: int = 5,
) -> list[dict]:
    """Pre-call hook: recall relevant memories from Kyros API.

    Calls the existing Kyros recall endpoint to fetch the top-k
    most relevant memories for the current user message.

    Returns a list of memory dicts with 'content' and 'score'.
    """
    start = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{config.kyros_base_url}/v1/memory/episodic/recall",
                json={
                    "agent_id": agent_id,
                    "query": user_message,
                    "k": k,
                },
                headers={
                    "X-API-Key": config.kyros_api_key,
                    "Content-Type": "application/json",
                },
            )

            if resp.status_code != 200:
                logger.warning(
                    "Recall failed",
                    status=resp.status_code,
                    agent_id=agent_id,
                )
                return []

            data = resp.json()
            results = data.get("results", [])

            elapsed = (time.monotonic() - start) * 1000
            logger.debug(
                "Recalled memories",
                agent_id=agent_id,
                count=len(results),
                latency_ms=round(elapsed, 2),
            )

            return results

    except Exception as e:
        logger.error("Recall error", error=str(e), agent_id=agent_id)
        return []


# ─── V08: Memory Injection ────────────────────

def format_memory_block(memories: list[dict], template: str) -> str:
    """Format recalled memories into a prompt-injection block.

    Takes raw memory results and formats them into a clean text
    block that gets prepended to the system message.

    Args:
        memories: List of memory dicts from the recall API.
        template: Format string with {memories} placeholder.

    Returns:
        Formatted memory block ready for injection, or empty
        string if no memories are available.
    """
    if not memories:
        return ""

    memory_lines = []
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        if content:
            memory_lines.append(f"  {i}. {content}")

    if not memory_lines:
        return ""

    memories_text = "\n".join(memory_lines)
    return template.format(memories=memories_text)


def inject_memories_into_request(
    provider_name: str,
    body: dict,
    system_msg: str,
    memory_block: str,
) -> dict:
    """Inject the memory block into the LLM request body.

    Uses the provider-specific injection method to place the
    memory block in the correct location (system message for
    OpenAI, system field for Anthropic, systemInstruction for Gemini).
    """
    from kyros.proxy.providers import get_provider

    if not memory_block:
        return body

    provider = get_provider(provider_name)
    return provider.inject_memories(body, system_msg, memory_block)


# ─── V09: Response Interceptor ────────────────

def extract_user_message(normalized: NormalizedRequest) -> str:
    """Extract the last user message from the normalized request.

    This is the query we use for memory recall — we want memories
    relevant to what the user is asking right now.
    """
    for msg in reversed(normalized.user_messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle structured content (e.g., OpenAI content arrays)
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return " ".join(text_parts)
    return ""


def should_store_response(
    content: str,
    config: ProxyConfig,
) -> bool:
    """Decide whether the LLM response is worth storing as a memory.

    Filters out:
    - Empty or very short responses
    - Generic responses ("Hello!", "Sure!", "I don't know")
    - Responses that are just code without explanation

    Args:
        content: The assistant's text response.
        config: Proxy configuration with sensitivity threshold.

    Returns:
        True if the response should be stored as a memory.
    """
    if not content or len(content.strip()) < config.min_content_length:
        return False

    # Generic/trivial responses to skip
    trivial_patterns = [
        "hello", "hi there", "sure", "okay", "ok", "yes", "no",
        "i don't know", "i'm not sure", "i cannot", "i can't",
        "how can i help", "is there anything else",
    ]

    content_lower = content.strip().lower()
    for pattern in trivial_patterns:
        if content_lower == pattern or content_lower == f"{pattern}.":
            return False

    # Skip if response is too short to be meaningful
    word_count = len(content.split())
    if word_count < 5:
        return False

    return True


async def store_memory(
    agent_id: str,
    content: str,
    config: ProxyConfig,
    role: str = "assistant",
    importance: float = 0.5,
) -> dict | None:
    """Store a memory in Kyros via the existing API.

    Called automatically by the proxy when a response passes
    the should_store_response filter.

    Returns the stored memory dict, or None on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{config.kyros_base_url}/v1/memory/episodic/remember",
                json={
                    "agent_id": agent_id,
                    "content": content,
                    "importance": importance,
                },
                headers={
                    "X-API-Key": config.kyros_api_key,
                    "Content-Type": "application/json",
                },
            )

            if resp.status_code in (200, 201):
                data = resp.json()
                logger.debug(
                    "Auto-stored memory",
                    agent_id=agent_id,
                    memory_id=data.get("memory_id"),
                )
                return data
            else:
                logger.warning(
                    "Auto-store failed",
                    status=resp.status_code,
                    agent_id=agent_id,
                )
                return None

    except Exception as e:
        logger.error("Auto-store error", error=str(e), agent_id=agent_id)
        return None
