"""
Agent Management API Endpoints

Handles agent creation, configuration, and lifecycle management.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from kyros.middleware.auth import get_current_user
from kyros.schemas.agent import Agent, AgentCreate, AgentUpdate
from kyros.services.agent import AgentService
from kyros.services.database import DatabaseService
from kyros.storage.database import get_database_service

router = APIRouter()


class AgentResponse(BaseModel):
    """Agent response model."""
    id: str
    name: str
    description: Optional[str]
    config: dict
    created_at: str
    updated_at: str
    is_active: bool


class AgentListResponse(BaseModel):
    """Agent list response model."""
    agents: List[AgentResponse]
    total: int
    page: int
    page_size: int


@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: AgentCreate,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Create a new agent for the authenticated user.
    
    Args:
        agent_data: Agent creation parameters
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Returns:
        AgentResponse: Created agent information
        
    Raises:
        HTTPException: If creation fails
    """
    agent_service = AgentService(db_service)
    
    try:
        agent = await agent_service.create_agent(
            user_id=current_user["id"],
            agent_data=agent_data,
        )
        
        return AgentResponse(
            id=str(agent.id),
            name=agent.name,
            description=agent.description,
            config=agent.config,
            created_at=agent.created_at.isoformat(),
            updated_at=agent.updated_at.isoformat(),
            is_active=agent.is_active,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {str(e)}"
        )


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    page: int = 1,
    page_size: int = 20,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    List all agents for the authenticated user.
    
    Args:
        page: Page number (1-based)
        page_size: Number of agents per page
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Returns:
        AgentListResponse: Paginated list of agents
        
    Raises:
        HTTPException: If listing fails
    """
    agent_service = AgentService(db_service)
    
    try:
        agents, total = await agent_service.list_agents(
            user_id=current_user["id"],
            page=page,
            page_size=page_size,
        )
        
        return AgentListResponse(
            agents=[
                AgentResponse(
                    id=str(agent.id),
                    name=agent.name,
                    description=agent.description,
                    config=agent.config,
                    created_at=agent.created_at.isoformat(),
                    updated_at=agent.updated_at.isoformat(),
                    is_active=agent.is_active,
                )
                for agent in agents
            ],
            total=total,
            page=page,
            page_size=page_size,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: UUID,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Get a specific agent by ID.
    
    Args:
        agent_id: Agent UUID
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Returns:
        AgentResponse: Agent information
        
    Raises:
        HTTPException: If agent not found or access denied
    """
    agent_service = AgentService(db_service)
    
    try:
        agent = await agent_service.get_agent(
            agent_id=agent_id,
            user_id=current_user["id"],
        )
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        return AgentResponse(
            id=str(agent.id),
            name=agent.name,
            description=agent.description,
            config=agent.config,
            created_at=agent.created_at.isoformat(),
            updated_at=agent.updated_at.isoformat(),
            is_active=agent.is_active,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent: {str(e)}"
        )


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: UUID,
    agent_data: AgentUpdate,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Update an existing agent.
    
    Args:
        agent_id: Agent UUID
        agent_data: Agent update parameters
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Returns:
        AgentResponse: Updated agent information
        
    Raises:
        HTTPException: If agent not found, access denied, or update fails
    """
    agent_service = AgentService(db_service)
    
    try:
        agent = await agent_service.update_agent(
            agent_id=agent_id,
            user_id=current_user["id"],
            agent_data=agent_data,
        )
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        return AgentResponse(
            id=str(agent.id),
            name=agent.name,
            description=agent.description,
            config=agent.config,
            created_at=agent.created_at.isoformat(),
            updated_at=agent.updated_at.isoformat(),
            is_active=agent.is_active,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent: {str(e)}"
        )


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: UUID,
    current_user: dict = Depends(get_current_user),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Delete an agent and all associated memories.
    
    Args:
        agent_id: Agent UUID
        current_user: Current authenticated user
        db_service: Database service dependency
        
    Raises:
        HTTPException: If agent not found, access denied, or deletion fails
    """
    agent_service = AgentService(db_service)
    
    try:
        success = await agent_service.delete_agent(
            agent_id=agent_id,
            user_id=current_user["id"],
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {str(e)}"
        )