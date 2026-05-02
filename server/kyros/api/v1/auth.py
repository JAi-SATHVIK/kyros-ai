"""
Authentication API Endpoints

Handles user authentication, API key management, and authorization.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

from kyros.services.auth import AuthService
from kyros.services.database import DatabaseService
from kyros.storage.database import get_database_service
from kyros.config import get_settings

router = APIRouter()
security = HTTPBearer()


class LoginRequest(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str
    expires_in: int


class ApiKeyRequest(BaseModel):
    """API key creation request model."""
    name: str
    description: Optional[str] = None
    expires_at: Optional[datetime] = None


class ApiKeyResponse(BaseModel):
    """API key response model."""
    id: str
    name: str
    key: str
    description: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    created_at: datetime
    is_active: bool


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Authenticate user and return access token.
    
    Args:
        request: Login credentials
        db_service: Database service dependency
        
    Returns:
        LoginResponse: Access token and metadata
        
    Raises:
        HTTPException: If credentials are invalid
    """
    auth_service = AuthService(db_service)
    
    try:
        user = await auth_service.authenticate_user(request.email, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        settings = get_settings()
        access_token = auth_service.create_access_token(
            data={"sub": user.id},
            expires_delta=timedelta(minutes=settings.jwt_expiry_minutes)
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_expiry_minutes * 60,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Get current authenticated user information.
    
    Args:
        credentials: JWT token from Authorization header
        db_service: Database service dependency
        
    Returns:
        UserResponse: Current user information
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    auth_service = AuthService(db_service)
    
    try:
        user = await auth_service.get_current_user(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return UserResponse(
            id=user.id,
            email=user.email,
            created_at=user.created_at,
            is_active=user.is_active,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    request: ApiKeyRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Create a new API key for the authenticated user.
    
    Args:
        request: API key creation parameters
        credentials: JWT token from Authorization header
        db_service: Database service dependency
        
    Returns:
        ApiKeyResponse: Created API key information
        
    Raises:
        HTTPException: If user is not authenticated or creation fails
    """
    auth_service = AuthService(db_service)
    
    # Verify user authentication
    user = await auth_service.get_current_user(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        api_key = await auth_service.create_api_key(
            user_id=user.id,
            name=request.name,
            description=request.description,
            expires_at=request.expires_at,
        )
        
        return ApiKeyResponse(
            id=api_key.id,
            name=api_key.name,
            key=api_key.key,
            description=api_key.description,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )


@router.get("/api-keys")
async def list_api_keys(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    List all API keys for the authenticated user.
    
    Args:
        credentials: JWT token from Authorization header
        db_service: Database service dependency
        
    Returns:
        List of API keys (without the actual key values)
        
    Raises:
        HTTPException: If user is not authenticated
    """
    auth_service = AuthService(db_service)
    
    # Verify user authentication
    user = await auth_service.get_current_user(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        api_keys = await auth_service.list_api_keys(user.id)
        
        return [
            {
                "id": key.id,
                "name": key.name,
                "description": key.description,
                "created_at": key.created_at,
                "expires_at": key.expires_at,
                "is_active": key.is_active,
            }
            for key in api_keys
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys"
        )