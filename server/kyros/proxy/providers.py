"""V03–V05 — Provider compatibility layers.

Translates between each LLM provider's API format and Kyros's
internal representation for memory injection and extraction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class NormalizedRequest:
    """Provider-agnostic representation of an LLM chat request."""

    provider: str               # "openai" | "anthropic" | "gemini"
    model: str                  # e.g. "gpt-4o", "claude-3-5-sonnet"
    system_message: str         # Current system prompt
    user_messages: list[dict]   # [{role, content}]
    raw_body: dict              # Original request body (for forwarding)
    raw_headers: dict           # Original headers (for forwarding)
    target_url: str             # The real provider URL to forward to


@dataclass
class NormalizedResponse:
    """Provider-agnostic representation of an LLM response."""

    provider: str
    model: str
    assistant_content: str      # The assistant's text reply
    raw_body: dict              # Original response JSON
    raw_status: int             # HTTP status code
    raw_headers: dict           # Response headers


# ─── V03: OpenAI Provider ──────────────────────

class OpenAIProvider:
    """Handles OpenAI /chat/completions format."""

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    @staticmethod
    def normalize_request(body: dict, headers: dict) -> NormalizedRequest:
        """Convert OpenAI request into normalized format."""
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        system_msg = ""
        user_msgs = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "system":
                system_msg = str(msg.get("content", ""))
            else:
                user_msgs.append(msg)

        return NormalizedRequest(
            provider="openai",
            model=str(body.get("model", os.environ.get("KYROS_OPENAI_MODEL", "gpt-4o"))),
            system_message=system_msg,
            user_messages=user_msgs,
            raw_body=body,
            raw_headers=dict(headers),
            target_url=OpenAIProvider.BASE_URL,
        )

    @staticmethod
    def inject_memories(body: dict, system_msg: str, memory_block: str) -> dict:
        """Inject memory block into the system message of an OpenAI request."""
        enriched_system = f"{memory_block}\n\n{system_msg}" if system_msg else memory_block

        new_messages = []
        has_system = False
        for msg in body.get("messages", []):
            if msg.get("role") == "system":
                new_messages.append({"role": "system", "content": enriched_system})
                has_system = True
            else:
                new_messages.append(msg)

        if not has_system:
            new_messages.insert(0, {"role": "system", "content": enriched_system})

        return {**body, "messages": new_messages}

    @staticmethod
    def extract_response(status: int, body: dict, headers: dict) -> NormalizedResponse:
        """Extract assistant content from OpenAI response."""
        content = ""
        try:
            choices = body.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
        except (IndexError, KeyError):
            content = ""

        return NormalizedResponse(
            provider="openai",
            model=body.get("model", "unknown"),
            assistant_content=content,
            raw_body=body,
            raw_status=status,
            raw_headers=dict(headers),
        )

    @staticmethod
    def build_forward_headers(headers: dict, api_key: str) -> dict:
        """Build headers for forwarding to OpenAI."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


# ─── V04: Anthropic Provider ──────────────────

class AnthropicProvider:
    """Handles Anthropic /messages format."""

    BASE_URL = "https://api.anthropic.com/v1/messages"

    @staticmethod
    def normalize_request(body: dict, headers: dict) -> NormalizedRequest:
        """Convert Anthropic request into normalized format."""
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        system_msg = body.get("system", "")
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        return NormalizedRequest(
            provider="anthropic",
            model=str(body.get("model", os.environ.get("KYROS_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))),
            system_message=system_msg if isinstance(system_msg, str) else "",
            user_messages=messages,
            raw_body=body,
            raw_headers=dict(headers),
            target_url=AnthropicProvider.BASE_URL,
        )

    @staticmethod
    def inject_memories(body: dict, system_msg: str, memory_block: str) -> dict:
        """Inject memory block into the system field of an Anthropic request."""
        enriched_system = f"{memory_block}\n\n{system_msg}" if system_msg else memory_block
        return {**body, "system": enriched_system}

    @staticmethod
    def extract_response(status: int, body: dict, headers: dict) -> NormalizedResponse:
        """Extract assistant content from Anthropic response."""
        content = ""
        try:
            content_blocks = body.get("content", [])
            text_blocks = [b["text"] for b in content_blocks if b.get("type") == "text"]
            content = "\n".join(text_blocks)
        except (IndexError, KeyError):
            content = ""

        return NormalizedResponse(
            provider="anthropic",
            model=body.get("model", "unknown"),
            assistant_content=content,
            raw_body=body,
            raw_status=status,
            raw_headers=dict(headers),
        )

    @staticmethod
    def build_forward_headers(headers: dict, api_key: str) -> dict:
        """Build headers for forwarding to Anthropic."""
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }


# ─── V05: Google Gemini Provider ──────────────

class GeminiProvider:
    """Handles Google Gemini generateContent format."""

    BASE_URL_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    @staticmethod
    def normalize_request(body: dict, headers: dict) -> NormalizedRequest:
        """Convert Gemini request into normalized format."""
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        model = str(body.get("model", os.environ.get("KYROS_GEMINI_MODEL", "gemini-1.5-flash")))

        system_msg = ""
        sys_instruction = body.get("systemInstruction", {})
        if isinstance(sys_instruction, dict):
            parts = sys_instruction.get("parts", [])
            system_msg = " ".join(
                p.get("text", "") for p in parts if isinstance(p, dict)
            )

        user_msgs = []
        for content_item in body.get("contents", []):
            if not isinstance(content_item, dict):
                continue
            role = content_item.get("role", "user")
            parts = content_item.get("parts", [])
            text = " ".join(
                p.get("text", "") for p in parts if isinstance(p, dict)
            )
            user_msgs.append({"role": role, "content": text})

        return NormalizedRequest(
            provider="gemini",
            model=model,
            system_message=system_msg,
            user_messages=user_msgs,
            raw_body=body,
            raw_headers=dict(headers),
            target_url=GeminiProvider.BASE_URL_TEMPLATE.format(model=model),
        )

    @staticmethod
    def inject_memories(body: dict, system_msg: str, memory_block: str) -> dict:
        """Inject memory block into systemInstruction of a Gemini request."""
        enriched_system = f"{memory_block}\n\n{system_msg}" if system_msg else memory_block
        return {
            **body,
            "systemInstruction": {
                "parts": [{"text": enriched_system}]
            },
        }

    @staticmethod
    def extract_response(status: int, body: dict, headers: dict) -> NormalizedResponse:
        """Extract assistant content from Gemini response."""
        content = ""
        try:
            candidates = body.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                content = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except (IndexError, KeyError):
            content = ""

        return NormalizedResponse(
            provider="gemini",
            model=body.get("modelVersion", "unknown"),
            assistant_content=content,
            raw_body=body,
            raw_status=status,
            raw_headers=dict(headers),
        )

    @staticmethod
    def build_forward_headers(headers: dict, api_key: str) -> dict:
        """Build headers for forwarding to Gemini (uses query param, not header)."""
        return {
            "Content-Type": "application/json",
        }

    @staticmethod
    def append_api_key_to_url(url: str, api_key: str) -> str:
        """Gemini uses API key as a query parameter."""
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}key={api_key}"


# ─── Provider Registry ────────────────────────

PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def get_provider(name: str):
    """Get a provider by name."""
    provider = PROVIDERS.get(name)
    if not provider:
        raise ValueError(f"Unknown provider: {name}. Supported: {list(PROVIDERS.keys())}")
    return provider
