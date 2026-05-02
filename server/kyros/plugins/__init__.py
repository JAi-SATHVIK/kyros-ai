"""
Kyros Plugin Architecture.

The Community core exports a lightweight plugin registry. Enterprise packages
register themselves here. If no enterprise package is installed, all enterprise
hooks are graceful no-ops — the Community Edition continues working perfectly.

This is the Open Core boundary.
"""

from __future__ import annotations
from typing import Callable, Any
from kyros.logging import get_logger

logger = get_logger("kyros.plugins")


class PluginRegistry:
    """Central registry for Enterprise plugin hooks."""
    
    def __init__(self):
        self._hooks: dict[str, list[Callable]] = {}
        self._enterprise_active = False
        
    def register(self, hook_name: str, fn: Callable) -> None:
        """Enterprise packages call this to register their hooks."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(fn)
        logger.info(f"Enterprise plugin registered: {hook_name}")
        self._enterprise_active = True
        
    async def call(self, hook_name: str, **kwargs) -> list[Any]:
        """The Community core calls hooks at extension points.
        
        Supports both sync and async hook functions.
        """
        if hook_name not in self._hooks:
            return []  # No enterprise plugin — graceful no-op
        results = []
        for fn in self._hooks[hook_name]:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(fn):
                    results.append(await fn(**kwargs))
                else:
                    results.append(fn(**kwargs))
            except Exception as e:
                logger.error("Plugin hook failed", hook=hook_name, error=str(e))
        return results
    
    @property
    def is_enterprise_active(self) -> bool:
        return self._enterprise_active


# Singleton registry
registry = PluginRegistry()


# ─── Extension Points exposed by Community Core ────────────────────────────

# Called after every successful fact store
HOOK_AFTER_FACT_STORE = "after_fact_store"

# Called after every successful episodic memory store
HOOK_AFTER_MEMORY_STORE = "after_memory_store"

# Called when generating embeddings (Enterprise can override with global model)
HOOK_EMBED = "embed"

# Called periodically (e.g., 24h) to trigger federated aggregation
HOOK_FEDERATED_ROUND = "federated_round"

# Called when belief propagation runs (Enterprise extends to cross-tenant)
HOOK_BELIEF_PROPAGATION = "belief_propagation"

# Called when an enterprise license key is validated
HOOK_LICENSE_CHECK = "license_check"
