"""
Memory Management API Endpoints

Handles episodic, semantic, and procedural memory operations for AI agents.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel

from kyros.middleware.auth import get_current_user
from kyros.schemas.memory import (
    EpisodicMemory, SemanticMemory, ProceduralMemory,
    EpisodicMemoryCreate, SemanticMemoryCreate, ProceduralMemoryCreate,
    MemoryQuery, MemoryQueryResult
)
from kyros.services.memory import MemoryService
from kyros.services.database import DatabaseService
from kyros.storage.database import get_database_service

router = APIRouter()


class MemoryResponse(BaseModel):
    """Base memory response model."""
    id: str
    agent_id: str
    content: str
    metadata: dict
    confidence: float
    created_at: str
    updated_at: str


class EpisodicMemoryResponse(MemoryResponse):
    """Episodic memory response model."""
    event_type: str
    context: dict
    temporal_markers: dict


class SemanticMemoryResponse(MemoryResponse):
    """Semantic memory response model."""
    subject: str
    predicate: str
    object: str
    belief_strength: float


class ProceduralMemoryResponse(MemoryResponse):
    """Procedural memory response model."""
    procedure_name: str
    steps: List[dict]
    success_rate: float
    usage_count: int


class MemoryQueryResponse(BaseModel):
    """Memory query response model."""
    results: List[MemoryResponse]
    total: int
    query_time_ms: float


@router.post("/episodic", response_model=EpisodicMemoryResponse, status_code=status.HTTP_201_CREATED)
async def store_episodic_memory(
    memory_data: EpisodicMemoryCreate,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Store a new episodic memory for an agent.
    
    Episodic memories capture specific events and experiences with temporal context.
    
    Args:
        memory_data: Episodic memory creation parameters
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Returns:
        EpisodicMemoryResponse: Created episodic memory
        
    Raises:
        HTTPException: If storage fails or agent access denied
    """
    memory_service = MemoryService(db_service)
    
    try:
        memory = await memory_service.store_episodic_memory(
            user_id=current_user["id"],
            memory_data=memory_data,
        )
        
        return EpisodicMemoryResponse(
            id=str(memory.id),
            agent_id=str(memory.agent_id),
            content=memory.content,
            metadata=memory.metadata,
            confidence=memory.confidence,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
            event_type=memory.event_type,
            context=memory.context,
            temporal_markers=memory.temporal_markers,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store episodic memory: {str(e)}"
        )


@router.post("/semantic", response_model=SemanticMemoryResponse, status_code=status.HTTP_201_CREATED)
async def store_semantic_memory(
    memory_data: SemanticMemoryCreate,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Store a new semantic memory for an agent.
    
    Semantic memories capture factual knowledge as subject-predicate-object triples.
    
    Args:
        memory_data: Semantic memory creation parameters
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Returns:
        SemanticMemoryResponse: Created semantic memory
        
    Raises:
        HTTPException: If storage fails or agent access denied
    """
    memory_service = MemoryService(db_service)
    
    try:
        memory = await memory_service.store_semantic_memory(
            user_id=current_user["id"],
            memory_data=memory_data,
        )
        
        return SemanticMemoryResponse(
            id=str(memory.id),
            agent_id=str(memory.agent_id),
            content=memory.content,
            metadata=memory.metadata,
            confidence=memory.confidence,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
            subject=memory.subject,
            predicate=memory.predicate,
            object=memory.object,
            belief_strength=memory.belief_strength,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store semantic memory: {str(e)}"
        )


@router.post("/procedural", response_model=ProceduralMemoryResponse, status_code=status.HTTP_201_CREATED)
async def store_procedural_memory(
    memory_data: ProceduralMemoryCreate,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Store a new procedural memory for an agent.
    
    Procedural memories capture learned behaviors and step-by-step processes.
    
    Args:
        memory_data: Procedural memory creation parameters
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Returns:
        ProceduralMemoryResponse: Created procedural memory
        
    Raises:
        HTTPException: If storage fails or agent access denied
    """
    memory_service = MemoryService(db_service)
    
    try:
        memory = await memory_service.store_procedural_memory(
            user_id=current_user["id"],
            memory_data=memory_data,
        )
        
        return ProceduralMemoryResponse(
            id=str(memory.id),
            agent_id=str(memory.agent_id),
            content=memory.content,
            metadata=memory.metadata,
            confidence=memory.confidence,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
            procedure_name=memory.procedure_name,
            steps=memory.steps,
            success_rate=memory.success_rate,
            usage_count=memory.usage_count,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store procedural memory: {str(e)}"
        )


@router.post("/query", response_model=MemoryQueryResponse)
async def query_memories(
    query_data: MemoryQuery,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Query memories using semantic search.
    
    Supports querying across all memory types with optional filtering by type,
    confidence threshold, and temporal constraints.
    
    Args:
        query_data: Memory query parameters
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Returns:
        MemoryQueryResponse: Query results with relevance scores
        
    Raises:
        HTTPException: If query fails or agent access denied
    """
    memory_service = MemoryService(db_service)
    
    try:
        results, query_time = await memory_service.query_memories(
            user_id=current_user["id"],
            query_data=query_data,
        )
        
        return MemoryQueryResponse(
            results=[
                MemoryResponse(
                    id=str(result.id),
                    agent_id=str(result.agent_id),
                    content=result.content,
                    metadata=result.metadata,
                    confidence=result.confidence,
                    created_at=result.created_at.isoformat(),
                    updated_at=result.updated_at.isoformat(),
                )
                for result in results
            ],
            total=len(results),
            query_time_ms=query_time,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query memories: {str(e)}"
        )


@router.get("/{agent_id}/episodic", response_model=List[EpisodicMemoryResponse])
async def get_episodic_memories(
    agent_id: UUID,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Get episodic memories for a specific agent.
    
    Args:
        agent_id: Agent UUID
        limit: Maximum number of memories to return
        offset: Number of memories to skip
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Returns:
        List[EpisodicMemoryResponse]: Agent's episodic memories
        
    Raises:
        HTTPException: If agent not found or access denied
    """
    memory_service = MemoryService(db_service)
    
    try:
        memories = await memory_service.get_episodic_memories(
            agent_id=agent_id,
            user_id=current_user["id"],
            limit=limit,
            offset=offset,
        )
        
        return [
            EpisodicMemoryResponse(
                id=str(memory.id),
                agent_id=str(memory.agent_id),
                content=memory.content,
                metadata=memory.metadata,
                confidence=memory.confidence,
                created_at=memory.created_at.isoformat(),
                updated_at=memory.updated_at.isoformat(),
                event_type=memory.event_type,
                context=memory.context,
                temporal_markers=memory.temporal_markers,
            )
            for memory in memories
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get episodic memories: {str(e)}"
        )


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: UUID,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Delete a specific memory.
    
    Args:
        memory_id: Memory UUID
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Raises:
        HTTPException: If memory not found, access denied, or deletion fails
    """
    memory_service = MemoryService(db_service)
    
    try:
        success = await memory_service.delete_memory(
            memory_id=memory_id,
            user_id=current_user["id"],
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}"
        )