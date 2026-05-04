"""
Pydantic schemas for API validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class RatingRequest(BaseModel):
    """Schema for rating submission."""
    user_id: int = Field(..., description="User ID")
    movie_id: int = Field(..., description="Movie ID")
    rating: float = Field(..., ge=1, le=5, description="Rating value 1-5")


class RatingResponse(BaseModel):
    """Schema for rating response."""
    user_id: int
    movie_id: int
    rating: float
    timestamp: datetime


class RecommendationRequest(BaseModel):
    """Schema for recommendation request."""
    user_id: int = Field(..., description="User ID")
    n_recommendations: int = Field(default=5, ge=1, le=20, description="Number of recommendations")
    model_name: Optional[str] = Field(default="hybrid", description="Model to use")


class RecommendationResponse(BaseModel):
    """Schema for recommendation response."""
    user_id: int
    recommended_movies: List[int]
    model_name: str
    confidence_scores: Optional[List[float]] = None
    timestamp: datetime


class UserCreateRequest(BaseModel):
    """Schema for user creation."""
    user_id: int = Field(..., description="User ID")
    username: Optional[str] = None
    email: Optional[str] = None


class UserResponse(BaseModel):
    """Schema for user response."""
    user_id: int
    username: str
    email: Optional[str]
    created_at: datetime


class PredictionRequest(BaseModel):
    """Schema for rating prediction."""
    user_id: int = Field(..., description="User ID")
    movie_id: int = Field(..., description="Movie ID")


class PredictionResponse(BaseModel):
    """Schema for rating prediction response."""
    user_id: int
    movie_id: int
    predicted_rating: float
    model_name: str


class HealthResponse(BaseModel):
    """Schema for health check."""
    status: str
    message: str
    database_connected: bool


class StatisticsResponse(BaseModel):
    """Schema for statistics."""
    total_users: int
    total_ratings: int
    total_recommendations: int
    avg_ratings_per_user: float
