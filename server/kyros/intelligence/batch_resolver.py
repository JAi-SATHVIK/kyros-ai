import json
import time
import asyncio
from uuid import UUID
from typing import Any
from redis.asyncio import Redis

from kyros.logging import get_logger
from kyros.intelligence.entity_resolver import resolve_and_update_entities
from kyros.intelligence.causal import extract_and_store_causal_edges

logger = get_logger("kyros.intelligence.batch_resolver")

async def enqueue_and_resolve_batch(
    redis_client: Redis,
    tenant_id: UUID,
    agent_id: UUID,
    memory_id: UUID,
    content: str,
    role: str | None,
    recent_memories: list[dict],
    force: bool = False
) -> None:
    """Queues a memory for batch entity resolution and causal extraction.
    
    If force=True, the queue size reaches 5, or the first item has been in the queue
    for > 30 seconds, flushes the queue and executes the batch LLM calls.
    """
    queue_key = f"kyros:batch:{agent_id}:memories"
    first_added_key = f"kyros:batch:{agent_id}:first_added_at"
    
    # Store memory details
    item = {
        "id": str(memory_id),
        "content": content,
        "role": role or "unknown"
    }
    await redis_client.rpush(queue_key, json.dumps(item))
    
    # Track when the first item was added to the batch
    first_added = await redis_client.get(first_added_key)
    now_ts = time.time()
    
    if not first_added:
        await redis_client.set(first_added_key, str(now_ts))
        first_added = str(now_ts)
        
        # Spawn a background task to flush the queue after a timeout (30 seconds)
        async def delayed_flush():
            await asyncio.sleep(30.0)
            await flush_if_old(redis_client, tenant_id, agent_id, recent_memories)
            
        asyncio.create_task(delayed_flush())
    
    len_queue = await redis_client.llen(queue_key)
    time_elapsed = now_ts - float(first_added)
    
    # Trigger flush if threshold reached, timeout met, or forced
    if len_queue >= 5 or time_elapsed >= 30.0 or force:
        await flush_queue(redis_client, tenant_id, agent_id, recent_memories, force)

async def flush_if_old(
    redis_client: Redis,
    tenant_id: UUID,
    agent_id: UUID,
    recent_memories: list[dict]
) -> None:
    """Checks if the batch has expired and flushes it if necessary."""
    queue_key = f"kyros:batch:{agent_id}:memories"
    first_added_key = f"kyros:batch:{agent_id}:first_added_at"
    
    first_added = await redis_client.get(first_added_key)
    if not first_added:
        return
        
    time_elapsed = time.time() - float(first_added)
    if time_elapsed >= 28.0:  # Allow small buffer for execution timing
        await flush_queue(redis_client, tenant_id, agent_id, recent_memories, force=True)

async def flush_queue(
    redis_client: Redis,
    tenant_id: UUID,
    agent_id: UUID,
    recent_memories: list[dict],
    force: bool
) -> None:
    """Consolidates and flushes the queue with transaction recovery fail-safes."""
    queue_key = f"kyros:batch:{agent_id}:memories"
    first_added_key = f"kyros:batch:{agent_id}:first_added_at"
    
    # Fetch timestamp for recovery reference
    first_added = await redis_client.get(first_added_key)
    
    # Get and clear queue atomically
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.lrange(queue_key, 0, -1)
        pipe.delete(queue_key)
        pipe.delete(first_added_key)
        raw_items, _, _ = await pipe.execute()
        
    if not raw_items:
        return
        
    items = [json.loads(x) for x in raw_items]
    
    logger.info(
        "Flushing memory batch for entity resolution",
        agent_id=str(agent_id),
        batch_size=len(items)
    )
    
    # 1. Consolidated Entity Resolution
    combined_text = "\n".join(
        f"[Speaker: {item['role']}] {item['content']}" for item in items
    )
    
    try:
        await resolve_and_update_entities(agent_id, combined_text)
    except Exception as e:
        # TRANSACTION SAFETY: Re-queue raw items on failure so no data is lost
        async with redis_client.pipeline(transaction=True) as pipe:
            for raw in reversed(raw_items):
                pipe.lpush(queue_key, raw)
            if first_added:
                pipe.set(first_added_key, first_added)
            await pipe.execute()
            
        logger.error("Failed batch entity resolution, items re-queued successfully", error=str(e))
        return
        
    # 2. Causal Edge Extraction
    if recent_memories:
        last_item = items[-1]
        try:
            await extract_and_store_causal_edges(
                tenant_id,
                agent_id,
                UUID(last_item["id"]),
                last_item["content"],
                recent_memories
            )
        except Exception as e:
            logger.error("Failed batch causal edge extraction", error=str(e))
