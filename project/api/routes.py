"""
API routes for recommendation system.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from typing import Optional

from api.schemas import (
    RatingRequest, RatingResponse, RecommendationRequest, RecommendationResponse,
    UserCreateRequest, UserResponse, PredictionRequest, PredictionResponse,
    HealthResponse, StatisticsResponse
)
from db.db import MongoDBHandler, InMemoryDatabase
from models.recommend import RecommenderFactory

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state (would normally use dependency injection)
_db: Optional[MongoDBHandler] = None
_models: dict = {}


def get_db() -> MongoDBHandler:
    """Get database instance."""
    return _db


def set_db(db: MongoDBHandler) -> None:
    """Set database instance."""
    global _db
    _db = db


def set_models(models: dict) -> None:
    """Set loaded models."""
    global _models
    _models = models


# Health & Status Endpoints

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and database connection."""
    db = get_db()
    return HealthResponse(
        status="healthy",
        message="API is running",
        database_connected=db.is_connected() if db else False
    )


@router.get("/stats", response_model=StatisticsResponse)
async def get_statistics():
    """Get system statistics."""
    db = get_db()
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    stats = db.get_stats()
    
    # Calculate average ratings per user
    total_users = stats.get('users', 1)
    total_ratings = stats.get('ratings', 0)
    avg_ratings = total_ratings / total_users if total_users > 0 else 0
    
    return StatisticsResponse(
        total_users=total_users,
        total_ratings=total_ratings,
        total_recommendations=stats.get('recommendations', 0),
        avg_ratings_per_user=avg_ratings
    )


# User Endpoints

@router.post("/users/create", response_model=UserResponse)
async def create_user(request: UserCreateRequest):
    """Create a new user."""
    db = get_db()
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    success = db.create_user(
        user_id=request.user_id,
        username=request.username,
        email=request.email
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Error creating user")
    
    user = db.get_user(request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        user_id=user['user_id'],
        username=user['username'],
        email=user.get('email'),
        created_at=user['created_at']
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """Get user information."""
    db = get_db()
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        user_id=user['user_id'],
        username=user['username'],
        email=user.get('email'),
        created_at=user['created_at']
    )


# Rating Endpoints

@router.post("/ratings", response_model=RatingResponse)
async def submit_rating(request: RatingRequest):
    """Submit a user rating."""
    db = get_db()
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    # Ensure user exists
    user = db.get_user(request.user_id)
    if not user:
        db.create_user(request.user_id)
    
    # Add rating
    success = db.add_rating(
        user_id=request.user_id,
        movie_id=request.movie_id,
        rating=request.rating
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Error submitting rating")
    
    return RatingResponse(
        user_id=request.user_id,
        movie_id=request.movie_id,
        rating=request.rating,
        timestamp=datetime.utcnow()
    )


@router.get("/users/{user_id}/ratings")
async def get_user_ratings(user_id: int):
    """Get all ratings by a user."""
    db = get_db()
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    ratings = db.get_user_ratings(user_id)
    return {"user_id": user_id, "ratings": ratings}


@router.get("/movies/{movie_id}/ratings")
async def get_movie_ratings(movie_id: int):
    """Get all ratings for a movie."""
    db = get_db()
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    ratings = db.get_movie_ratings(movie_id)
    return {"movie_id": movie_id, "ratings": ratings}


# Recommendation Endpoints

@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a user."""
    db = get_db()
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    # Select model
    model = _models.get(request.model_name)
    if not model:
        available_models = list(_models.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' not found. Available: {available_models}"
        )
    
    try:
        # Generate recommendations
        recommendations = model.recommend(
            request.user_id,
            n_recommendations=request.n_recommendations
        )
        
        if not recommendations:
            # Return empty recommendations
            recommendations = []
        
        # Save to database
        db.save_recommendation(
            user_id=request.user_id,
            recommended_movies=recommendations,
            model_name=request.model_name
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommended_movies=recommendations,
            model_name=request.model_name,
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/recommendations/{model_name}")
async def get_stored_recommendations(user_id: int, model_name: str):
    """Get stored recommendations for a user."""
    db = get_db()
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    recommendations = db.get_recommendations(user_id, model_name)
    
    if not recommendations:
        raise HTTPException(status_code=404, detail="Recommendations not found")
    
    return recommendations


# Prediction Endpoints

@router.post("/predict", response_model=PredictionResponse)
async def predict_rating(request: PredictionRequest):
    """Predict rating for user-movie pair."""
    model = _models.get('collaborative')
    
    if not model:
        raise HTTPException(status_code=400, detail="Collaborative model not available")
    
    try:
        # Predict rating
        predicted_rating = model.predict_rating(request.user_id, request.movie_id)
        
        return PredictionResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            predicted_rating=predicted_rating,
            model_name="collaborative"
        )
    
    except Exception as e:
        logger.error(f"Error predicting rating: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_available_models():
    """List available recommendation models."""
    return {"models": list(_models.keys())}
