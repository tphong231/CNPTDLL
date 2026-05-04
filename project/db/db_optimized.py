"""
Optimized Database Module with Caching.
Reduces MongoDB queries and fetches only necessary fields.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("PyMongo not available. Database features will be limited.")


class MongoDBHandlerOptimized:
    """
    Optimized MongoDB handler with caching to reduce queries.
    Features: Result caching, minimal field fetching, batch operations.
    """
    
    def __init__(self, mongo_uri: str = None, db_name: str = "recommendation_system",
                 timeout: int = 5000, cache_ttl: int = 300):
        """
        Initialize optimized MongoDB connection with caching.
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
            timeout: Connection timeout in ms
            cache_ttl: Cache time-to-live in seconds (5 min default)
        """
        if not MONGODB_AVAILABLE:
            logger.error("MongoDB not available. Install pymongo: pip install pymongo")
            self.client = None
            self.db = None
            return
        
        import os
        self.mongo_uri = mongo_uri or os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        
        # Cache storage
        self._cache = {}
        self._cache_timestamps = {}
        
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=timeout)
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            logger.info(f"Connected to MongoDB: {db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.client = None
            self.db = None
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected."""
        return self.db is not None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        elapsed = time.time() - self._cache_timestamps[cache_key]
        return elapsed < self.cache_ttl
    
    def _get_cached(self, cache_key: str):
        """Get value from cache if valid."""
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit: {cache_key}")
            return self._cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, value):
        """Store value in cache with timestamp."""
        self._cache[cache_key] = value
        self._cache_timestamps[cache_key] = time.time()
    
    def _clear_cache_for_key(self, pattern: str):
        """Clear cache entries matching pattern."""
        keys_to_delete = [k for k in self._cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self._cache[key]
            del self._cache_timestamps[key]
    
    # User Operations
    
    def create_user(self, user_id: int, username: str = None, 
                   email: str = None, metadata: Dict = None) -> bool:
        """
        Create a new user.
        
        Args:
            user_id: Unique user ID
            username: Optional username
            email: Optional email
            metadata: Optional metadata dict
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            logger.warning("Database not connected.")
            return False
        
        try:
            users_collection = self.db['users']
            user_doc = {
                'user_id': user_id,
                'username': username or f'user_{user_id}',
                'email': email,
                'created_at': datetime.utcnow(),
                'metadata': metadata or {}
            }
            
            users_collection.update_one(
                {'user_id': user_id},
                {'$set': user_doc},
                upsert=True
            )
            
            # Invalidate user cache
            self._clear_cache_for_key(f'user_{user_id}')
            
            logger.info(f"User {user_id} created/updated")
            return True
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return False
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID (with caching)."""
        if not self.is_connected():
            return None
        
        # Check cache first
        cache_key = f'user_{user_id}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            users_collection = self.db['users']
            # Only fetch necessary fields
            user = users_collection.find_one(
                {'user_id': user_id},
                {'_id': 0, 'user_id': 1, 'username': 1, 'email': 1}
            )
            
            if user:
                self._set_cache(cache_key, user)
            
            return user
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            return None
    
    # Rating Operations (with caching)
    
    def add_rating(self, user_id: int, movie_id: int, rating: float) -> bool:
        """
        Add or update a user's rating (with cache invalidation).
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            rating: Rating value (1-5)
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            logger.warning("Database not connected.")
            return False
        
        try:
            ratings_collection = self.db['ratings']
            rating_doc = {
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': datetime.utcnow()
            }
            
            ratings_collection.update_one(
                {'user_id': user_id, 'movie_id': movie_id},
                {'$set': rating_doc},
                upsert=True
            )
            
            # Invalidate related caches
            self._clear_cache_for_key(f'ratings_user_{user_id}')
            self._clear_cache_for_key('all_ratings')
            
            logger.info(f"Rating added: user {user_id}, movie {movie_id}, rating {rating}")
            return True
        except Exception as e:
            logger.error(f"Error adding rating: {str(e)}")
            return False
    
    def get_user_ratings(self, user_id: int) -> List[Dict]:
        """Get all ratings by a user (with caching)."""
        if not self.is_connected():
            return []
        
        # Check cache first
        cache_key = f'ratings_user_{user_id}'
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            ratings_collection = self.db['ratings']
            # Only fetch necessary fields
            ratings = list(ratings_collection.find(
                {'user_id': user_id},
                {'_id': 0, 'user_id': 1, 'movie_id': 1, 'rating': 1, 'timestamp': 1}
            ))
            
            if ratings:
                self._set_cache(cache_key, ratings)
            
            return ratings
        except Exception as e:
            logger.error(f"Error getting user ratings: {str(e)}")
            return []
    
    def get_movie_ratings(self, movie_id: int) -> List[Dict]:
        """Get all ratings for a movie (with caching)."""
        if not self.is_connected():
            return []
        
        cache_key = f'ratings_movie_{movie_id}'
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            ratings_collection = self.db['ratings']
            ratings = list(ratings_collection.find(
                {'movie_id': movie_id},
                {'_id': 0, 'user_id': 1, 'movie_id': 1, 'rating': 1}
            ))
            
            if ratings:
                self._set_cache(cache_key, ratings)
            
            return ratings
        except Exception as e:
            logger.error(f"Error getting movie ratings: {str(e)}")
            return []
    
    def get_all_ratings(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all ratings (with caching and optional limit).
        Much faster than without caching.
        """
        if not self.is_connected():
            return []
        
        cache_key = f'all_ratings_{limit or "unlimited"}'
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            ratings_collection = self.db['ratings']
            query = ratings_collection.find(
                {},
                {'_id': 0, 'user_id': 1, 'movie_id': 1, 'rating': 1}
            )
            
            if limit:
                query = query.limit(limit)
            
            ratings = list(query)
            
            if ratings:
                self._set_cache(cache_key, ratings)
            
            return ratings
        except Exception as e:
            logger.error(f"Error getting all ratings: {str(e)}")
            return []
    
    # Recommendation Operations
    
    def save_recommendation(self, user_id: int, recommended_movies: List[int],
                          model_name: str = 'default', confidence: List[float] = None) -> bool:
        """
        Save recommendations for a user.
        
        Args:
            user_id: User ID
            recommended_movies: List of recommended movie IDs
            model_name: Name of the model used
            confidence: Optional confidence scores
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            return False
        
        try:
            recommendations_collection = self.db['recommendations']
            rec_doc = {
                'user_id': user_id,
                'movies': recommended_movies,
                'model': model_name,
                'confidence': confidence or [],
                'created_at': datetime.utcnow()
            }
            
            recommendations_collection.update_one(
                {'user_id': user_id},
                {'$set': rec_doc},
                upsert=True
            )
            
            logger.info(f"Recommendations saved for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get database statistics (with caching).
        
        Returns:
            Dict with users, ratings, and recommendations counts
        """
        if not self.is_connected():
            return {'users': 0, 'ratings': 0, 'recommendations': 0}
        
        cache_key = 'db_stats'
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            stats = {
                'users': self.db['users'].count_documents({}),
                'ratings': self.db['ratings'].count_documents({}),
                'recommendations': self.db['recommendations'].count_documents({})
            }
            
            self._set_cache(cache_key, stats)
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {'users': 0, 'ratings': 0, 'recommendations': 0}
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache = {}
        self._cache_timestamps = {}
        logger.info("Database cache cleared")
