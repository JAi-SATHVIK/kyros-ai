"""V02 — Kyros Zero-Code Memory Proxy Server.

A transparent HTTP proxy that intercepts LLM API calls, injects
relevant memories, and extracts new memories from responses.

Routes:
    POST /v1/openai/chat/completions    → OpenAI proxy
    POST /v1/anthropic/messages         → Anthropic proxy
    POST /v1/gemini/generateContent     → Gemini proxy
    GET  /proxy/health                  → Health check
    GET  /proxy/metrics                 → Proxy metrics

Usage:
    # As standalone server:
    python -m kyros.proxy.server

    # Programmatic:
    from kyros.proxy.server import create_proxy_app
    app = create_proxy_app(config)
    uvicorn.run(app, port=8080)
"""

from __future__ import annotations

import os
import secrets
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from kyros.proxy.architecture import ProxyConfig, ProxyMetrics
from kyros.proxy.providers import get_provider
from kyros.proxy.interceptors import (
    extract_agent_id,
    recall_memories,
    format_memory_block,
    inject_memories_into_request,
    extract_user_message,
    should_store_response,
    store_memory,
)
from kyros.logging import get_logger

logger = get_logger("kyros.proxy.server")
MAX_PROXY_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB


class ProxyConfigUpdate(BaseModel):
    """Strict schema for runtime config updates."""

    model_config = ConfigDict(extra="forbid")

    injection_enabled: bool | None = None
    extraction_enabled: bool | None = None
    extraction_sensitivity: float | None = None
    max_memories_to_inject: int | None = None
    min_content_length: int | None = None


def _require_admin_token(request: Request, admin_token: str) -> JSONResponse | None:
    """Protect mutable proxy config endpoints with an admin token."""
    if not admin_token:
        return JSONResponse(
            {"error": "Proxy config endpoint disabled: KYROS_PROXY_ADMIN_TOKEN is not set"},
            status_code=503,
        )
    presented = request.headers.get("X-Proxy-Admin-Token", "")
    if not secrets.compare_digest(presented, admin_token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return None


async def _read_json_body(request: Request) -> tuple[dict, JSONResponse | None]:
    """Parse and validate JSON body with a conservative payload-size guard."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_PROXY_BODY_BYTES:
                return {}, JSONResponse({"error": "Payload too large"}, status_code=413)
        except ValueError:
            return {}, JSONResponse({"error": "Invalid Content-Length header"}, status_code=400)

    try:
        body = await request.json()
    except Exception:
        return {}, JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "JSON body must be an object"}, status_code=400)
    return body, None


def create_proxy_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create the proxy FastAPI application.

    Args:
        config: Proxy configuration. If None, reads from env vars.

    Returns:
        Configured FastAPI app ready to serve as an LLM proxy.
    """
    if config is None:
        config = ProxyConfig(
            kyros_api_key=os.environ.get("KYROS_API_KEY", ""),
            kyros_base_url=os.environ.get("KYROS_BASE_URL", "http://localhost:8000"),
            proxy_port=int(os.environ.get("KYROS_PROXY_PORT", "8080")),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        )

    metrics = ProxyMetrics()
    admin_token = os.environ.get("KYROS_PROXY_ADMIN_TOKEN", "").strip()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        app_instance.state.upstream_client = httpx.AsyncClient(timeout=120.0)
        try:
            yield
        finally:
            await app_instance.state.upstream_client.aclose()

    app = FastAPI(
        title="Kyros Zero-Code Memory Proxy",
        description="Transparent LLM proxy that adds persistent memory to any AI agent.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def add_security_headers(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-XSS-Protection"] = "0"
        if os.environ.get("KYROS_ENVIRONMENT", "").strip() == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
        return response

    # ─── Health & Metrics ──────────────────────

    @app.get("/proxy/health")
    async def health():
        return {
            "status": "ok",
            "proxy_version": "0.1.0",
            "injection_enabled": config.injection_enabled,
            "extraction_enabled": config.extraction_enabled,
        }

    @app.get("/proxy/metrics")
    async def get_metrics():
        return {
            "requests_total": metrics.requests_total,
            "memories_injected": metrics.memories_injected,
            "memories_extracted": metrics.memories_extracted,
            "avg_injection_latency_ms": round(metrics.avg_injection_latency_ms, 2),
            "avg_extraction_latency_ms": round(metrics.avg_extraction_latency_ms, 2),
            "errors": metrics.errors,
        }

    # ─── V16: Runtime Config ──────────────────

    @app.get("/proxy/config")
    async def get_config(request: Request):
        """Get current proxy configuration."""
        auth_error = _require_admin_token(request, admin_token)
        if auth_error:
            return auth_error
        return {
            "injection_enabled": config.injection_enabled,
            "extraction_enabled": config.extraction_enabled,
            "extraction_sensitivity": config.extraction_sensitivity,
            "max_memories_to_inject": config.max_memories_to_inject,
            "min_content_length": config.min_content_length,
            "kyros_base_url": config.kyros_base_url,
        }

    @app.put("/proxy/config")
    async def update_config(request: Request):
        """Update proxy configuration at runtime (no restart needed)."""
        auth_error = _require_admin_token(request, admin_token)
        if auth_error:
            return auth_error
        body, body_error = await _read_json_body(request)
        if body_error:
            return body_error

        try:
            update = ProxyConfigUpdate.model_validate(body)
        except ValidationError as exc:
            return JSONResponse(
                {"error": "Invalid config payload", "detail": exc.errors()},
                status_code=422,
            )

        updated = []

        if update.injection_enabled is not None:
            config.injection_enabled = update.injection_enabled
            updated.append("injection_enabled")

        if update.extraction_enabled is not None:
            config.extraction_enabled = update.extraction_enabled
            updated.append("extraction_enabled")

        if update.extraction_sensitivity is not None:
            config.extraction_sensitivity = max(0.0, min(1.0, update.extraction_sensitivity))
            updated.append("extraction_sensitivity")

        if update.max_memories_to_inject is not None:
            config.max_memories_to_inject = max(1, min(20, update.max_memories_to_inject))
            updated.append("max_memories_to_inject")

        if update.min_content_length is not None:
            config.min_content_length = max(5, min(500, update.min_content_length))
            updated.append("min_content_length")

        logger.info("Config updated", fields=updated)

        return {
            "updated": updated,
            "config": {
                "injection_enabled": config.injection_enabled,
                "extraction_enabled": config.extraction_enabled,
                "extraction_sensitivity": config.extraction_sensitivity,
                "max_memories_to_inject": config.max_memories_to_inject,
                "min_content_length": config.min_content_length,
            },
        }

    # ─── Core Proxy Handler ───────────────────

    async def proxy_handler(
        request: Request,
        provider_name: str,
    ) -> JSONResponse:
        """Universal proxy handler for all LLM providers.

        Flow:
        1. Parse incoming request
        2. V06: Extract agent_id from headers
        3. V07: Recall relevant memories
        4. V08: Inject memories into system prompt
        5. Forward enriched request to real provider
        6. V09: Extract and store new memories from response
        7. Return original response to caller
        """
        request_start = time.monotonic()
        metrics.requests_total += 1

        # Parse request body
        body, body_error = await _read_json_body(request)
        if body_error:
            return body_error

        headers = dict(request.headers)
        provider = get_provider(provider_name)

        # V06: Extract agent_id
        agent_id = extract_agent_id(headers)

        # Normalize the request
        normalized = provider.normalize_request(body, headers)

        # ── Memory Injection (V07 + V08) ──
        enriched_body = body
        if agent_id and config.injection_enabled:
            inject_start = time.monotonic()

            # V07: Recall relevant memories
            user_msg = extract_user_message(normalized)
            if user_msg:
                memories = await recall_memories(
                    agent_id=agent_id,
                    user_message=user_msg,
                    config=config,
                    k=config.max_memories_to_inject,
                )

                # V08: Inject into system prompt
                if memories:
                    memory_block = format_memory_block(
                        memories, config.injection_template
                    )
                    enriched_body = inject_memories_into_request(
                        provider_name=provider_name,
                        body=body,
                        system_msg=normalized.system_message,
                        memory_block=memory_block,
                    )
                    metrics.memories_injected += len(memories)

                    logger.info(
                        "Injected memories",
                        agent_id=agent_id,
                        count=len(memories),
                        provider=provider_name,
                    )

            inject_elapsed = (time.monotonic() - inject_start) * 1000
            # Running average
            n = metrics.requests_total
            metrics.avg_injection_latency_ms = (
                (metrics.avg_injection_latency_ms * (n - 1) + inject_elapsed) / n
            )

        # ── Forward to Real Provider ──────
        try:
            # Build provider API key mapping
            api_keys = {
                "openai": config.openai_api_key,
                "anthropic": config.anthropic_api_key,
                "gemini": config.gemini_api_key,
            }
            api_key = api_keys.get(provider_name, "")

            forward_headers = provider.build_forward_headers(headers, api_key)
            target_url = normalized.target_url

            # Gemini uses query param for API key
            if provider_name == "gemini":
                target_url = provider.append_api_key_to_url(target_url, api_key)

            client: httpx.AsyncClient | None = getattr(request.app.state, "upstream_client", None)
            if client is None:
                async with httpx.AsyncClient(timeout=120.0) as fallback_client:
                    resp = await fallback_client.post(
                        target_url,
                        json=enriched_body,
                        headers=forward_headers,
                    )
            else:
                resp = await client.post(
                    target_url,
                    json=enriched_body,
                    headers=forward_headers,
                )

            resp_body = resp.json()
            resp_headers = dict(resp.headers)

        except httpx.TimeoutException:
            metrics.errors += 1
            return JSONResponse(
                {"error": "Upstream provider timed out"},
                status_code=504,
            )
        except Exception as e:
            metrics.errors += 1
            logger.error("Forward error", error=str(e), provider=provider_name)
            return JSONResponse(
                {"error": f"Failed to contact {provider_name}: {str(e)}"},
                status_code=502,
            )

        # ── Memory Extraction (V09) ───────
        if agent_id and config.extraction_enabled and resp.status_code == 200:
            extract_start = time.monotonic()

            normalized_resp = provider.extract_response(
                resp.status_code, resp_body, resp_headers
            )

            # Store user message as episodic memory
            user_msg = extract_user_message(normalized)
            if user_msg and should_store_response(user_msg, config):
                await store_memory(
                    agent_id=agent_id,
                    content=user_msg,
                    config=config,
                    role="user",
                    importance=0.6,
                )
                metrics.memories_extracted += 1

            # Store assistant response as episodic memory
            if should_store_response(normalized_resp.assistant_content, config):
                await store_memory(
                    agent_id=agent_id,
                    content=normalized_resp.assistant_content,
                    config=config,
                    role="assistant",
                    importance=0.5,
                )
                metrics.memories_extracted += 1

            extract_elapsed = (time.monotonic() - extract_start) * 1000
            n = metrics.requests_total
            metrics.avg_extraction_latency_ms = (
                (metrics.avg_extraction_latency_ms * (n - 1) + extract_elapsed) / n
            )

        total_ms = (time.monotonic() - request_start) * 1000
        logger.info(
            "Proxy request complete",
            provider=provider_name,
            agent_id=agent_id or "none",
            status=resp.status_code,
            total_ms=round(total_ms, 2),
        )

        # Return the original provider response untouched
        return JSONResponse(
            content=resp_body,
            status_code=resp.status_code,
        )

    # ─── Route Registration ───────────────────

    # V03: OpenAI — matches /v1/openai/chat/completions and /v1/chat/completions
    @app.post("/v1/openai/chat/completions")
    async def proxy_openai(request: Request):
        return await proxy_handler(request, "openai")

    @app.post("/v1/chat/completions")
    async def proxy_openai_compat(request: Request):
        """OpenAI-compatible route (drop-in for base_url change)."""
        return await proxy_handler(request, "openai")

    # V04: Anthropic
    @app.post("/v1/anthropic/messages")
    async def proxy_anthropic(request: Request):
        return await proxy_handler(request, "anthropic")

    # V05: Gemini
    @app.post("/v1/gemini/generateContent")
    async def proxy_gemini(request: Request):
        return await proxy_handler(request, "gemini")

    return app


# ─── CLI Entry Point ──────────────────────────

def start(
    api_key: str | None = None,
    port: int = 8080,
    host: str = "0.0.0.0",
    kyros_url: str | None = None,
) -> None:
    """Start the proxy server programmatically.

    Args:
        api_key: Kyros API key. Falls back to KYROS_API_KEY env var.
        port: Port to listen on. Falls back to KYROS_PROXY_PORT env var.
        host: Host to bind to.
        kyros_url: Kyros API base URL. Falls back to KYROS_BASE_URL env var.

    Usage:
        import kyros.proxy.server as proxy
        proxy.start(api_key="mk_live_...", port=8080)
    """
    resolved_key = api_key or os.environ.get("KYROS_API_KEY", "")
    resolved_url = kyros_url or os.environ.get("KYROS_BASE_URL", "http://localhost:8000")
    resolved_port = port or int(os.environ.get("KYROS_PROXY_PORT", "8080"))
    config = ProxyConfig(
        kyros_api_key=resolved_key,
        kyros_base_url=resolved_url,
        proxy_port=resolved_port,
        proxy_host=host,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
    )

    app = create_proxy_app(config)

    logger.info(
        "Starting Kyros Memory Proxy",
        port=resolved_port,
        kyros_url=resolved_url,
        injection=config.injection_enabled,
        extraction=config.extraction_enabled,
    )

    uvicorn.run(app, host=host, port=resolved_port, log_level="info")


if __name__ == "__main__":
    start()
