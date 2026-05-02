"""
Health Check API Endpoints

Provides health and status endpoints for monitoring and load balancing.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from kyros.services.database import DatabaseService
from kyros.services.redis import RedisService
from kyros.storage.database import get_database_service
from kyros.storage.redis import get_redis_service

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    service: str
    version: str
    database: str
    redis: str


@router.get("/", response_model=HealthResponse)
async def health_check(
    db_service: DatabaseService = Depends(get_database_service),
    redis_service: RedisService = Depends(get_redis_service),
):
    """
    Comprehensive health check endpoint.
    
    Checks the status of all critical services including database and Redis.
    """
    # Check database connectivity
    try:
        await db_service.health_check()
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    # Check Redis connectivity
    try:
        await redis_service.health_check()
        redis_status = "healthy"
    except Exception:
        redis_status = "unhealthy"
    
    # Overall status
    overall_status = "healthy" if db_status == "healthy" and redis_status == "healthy" else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        service="kyros-memory-os",
        version="0.1.0",
        database=db_status,
        redis=redis_status,
    )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the service is ready to accept traffic.
    """
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if the service is alive and running.
    """
    return {"status": "alive"}