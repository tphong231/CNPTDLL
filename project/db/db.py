
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import MongoDB
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("PyMongo not available. Database features will be limited.")


class MongoDBHandler:
    """
    Handles all MongoDB operations for the recommendation system.
    Stores users, ratings, and recommendations.
    """
    
    def __init__(self, mongo_uri: str = None, db_name: str = "recommendation_system",
                 timeout: int = 5000):
        """
        Initialize MongoDB connection.
        
        Args:
            mongo_uri: MongoDB connection string (default from env)
            db_name: Database name
            timeout: Connection timeout in ms
        """
        if not MONGODB_AVAILABLE:
            logger.error("MongoDB not available. Install pymongo: pip install pymongo")
            self.client = None
            self.db = None
            return
        
        self.mongo_uri = mongo_uri or os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self.timeout = timeout
        
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=timeout)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            logger.info(f"Connected to MongoDB: {db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            logger.info("Running in offline mode. Database features disabled.")
            self.client = None
            self.db = None
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected."""
        return self.db is not None
    
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
            True if successful, False otherwise
        """
        if not self.is_connected():
            logger.warning("Database not connected. Skipping user creation.")
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
            logger.info(f"User {user_id} created/updated")
            return True
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return False
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        if not self.is_connected():
            return None
        
        try:
            users_collection = self.db['users']
            return users_collection.find_one({'user_id': user_id})
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            return None
    
    # Rating Operations
    
    def add_rating(self, user_id: int, movie_id: int, rating: float) -> bool:
        """
        Add or update a user's rating for a movie.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            rating: Rating value (1-5)
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            logger.warning("Database not connected. Skipping rating storage.")
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
            logger.info(f"Rating added: user {user_id}, movie {movie_id}, rating {rating}")
            return True
        except Exception as e:
            logger.error(f"Error adding rating: {str(e)}")
            return False
    
    def get_user_ratings(self, user_id: int) -> List[Dict]:
        """Get all ratings by a user."""
        if not self.is_connected():
            return []
        
        try:
            ratings_collection = self.db['ratings']
            return list(ratings_collection.find({'user_id': user_id}))
        except Exception as e:
            logger.error(f"Error getting user ratings: {str(e)}")
            return []
    
    def get_movie_ratings(self, movie_id: int) -> List[Dict]:
        """Get all ratings for a movie."""
        if not self.is_connected():
            return []
        
        try:
            ratings_collection = self.db['ratings']
            return list(ratings_collection.find({'movie_id': movie_id}))
        except Exception as e:
            logger.error(f"Error getting movie ratings: {str(e)}")
            return []
    
    def get_all_ratings(self) -> List[Dict]:
        """Get all ratings."""
        if not self.is_connected():
            return []
        
        try:
            ratings_collection = self.db['ratings']
            return list(ratings_collection.find({}))
        except Exception as e:
            logger.error(f"Error getting ratings: {str(e)}")
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
            confidence: Optional list of confidence scores
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            logger.warning("Database not connected. Skipping recommendation storage.")
            return False
        
        try:
            recommendations_collection = self.db['recommendations']
            rec_doc = {
                'user_id': user_id,
                'recommended_movies': recommended_movies,
                'confidence_scores': confidence or [0.0] * len(recommended_movies),
                'model_name': model_name,
                'timestamp': datetime.utcnow()
            }
            
            recommendations_collection.update_one(
                {'user_id': user_id, 'model_name': model_name},
                {'$set': rec_doc},
                upsert=True
            )
            logger.info(f"Recommendations saved for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
            return False
    
    def get_recommendations(self, user_id: int, model_name: str = None) -> Optional[Dict]:
        """Get recommendations for a user."""
        if not self.is_connected():
            return None
        
        try:
            recommendations_collection = self.db['recommendations']
            query = {'user_id': user_id}
            if model_name:
                query['model_name'] = model_name
            
            return recommendations_collection.find_one(query)
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return None
    
    # Utility Operations
    
    def clear_collection(self, collection_name: str) -> bool:
        """Clear all documents from a collection."""
        if not self.is_connected():
            return False
        
        try:
            self.db[collection_name].delete_many({})
            logger.info(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        if not self.is_connected():
            return {}
        
        try:
            stats = {
                'users': self.db['users'].count_documents({}),
                'ratings': self.db['ratings'].count_documents({}),
                'recommendations': self.db['recommendations'].count_documents({})
            }
            logger.info(f"Database stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def create_indexes(self) -> None:
        """Create indexes for optimized queries."""
        if not self.is_connected():
            return
        
        try:
            # User indexes
            self.db['users'].create_index('user_id', unique=True)
            
            # Rating indexes
            self.db['ratings'].create_index([('user_id', 1), ('movie_id', 1)], unique=True)
            self.db['ratings'].create_index('user_id')
            self.db['ratings'].create_index('movie_id')
            
            # Recommendation indexes
            self.db['recommendations'].create_index([('user_id', 1), ('model_name', 1)], unique=True)
            self.db['recommendations'].create_index('user_id')
            
            logger.info("Database indexes created")
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


class InMemoryDatabase:
    """
    In-memory database fallback for when MongoDB is not available.
    Useful for development and testing.
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self.users: Dict[int, Dict] = {}
        self.ratings: List[Dict] = []
        self.recommendations: Dict[int, Dict] = {}
        logger.info("InMemoryDatabase initialized")
    
    def is_connected(self) -> bool:
        """Always connected."""
        return True
    
    def create_user(self, user_id: int, username: str = None,
                   email: str = None, metadata: Dict = None) -> bool:
        """Create/update user."""
        self.users[user_id] = {
            'user_id': user_id,
            'username': username or f'user_{user_id}',
            'email': email,
            'created_at': datetime.utcnow(),
            'metadata': metadata or {}
        }
        return True
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def add_rating(self, user_id: int, movie_id: int, rating: float) -> bool:
        """Add/update rating."""
        # Remove existing rating if present
        self.ratings = [r for r in self.ratings 
                       if not (r['user_id'] == user_id and r['movie_id'] == movie_id)]
        
        self.ratings.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': datetime.utcnow()
        })
        return True
    
    def get_user_ratings(self, user_id: int) -> List[Dict]:
        """Get user's ratings."""
        return [r for r in self.ratings if r['user_id'] == user_id]
    
    def get_movie_ratings(self, movie_id: int) -> List[Dict]:
        """Get ratings for a movie."""
        return [r for r in self.ratings if r['movie_id'] == movie_id]
    
    def get_all_ratings(self) -> List[Dict]:
        """Get all ratings."""
        return self.ratings.copy()
    
    def save_recommendation(self, user_id: int, recommended_movies: List[int],
                          model_name: str = 'default', confidence: List[float] = None) -> bool:
        """Save recommendations."""
        key = f"{user_id}_{model_name}"
        self.recommendations[key] = {
            'user_id': user_id,
            'recommended_movies': recommended_movies,
            'confidence_scores': confidence or [0.0] * len(recommended_movies),
            'model_name': model_name,
            'timestamp': datetime.utcnow()
        }
        return True
    
    def get_recommendations(self, user_id: int, model_name: str = None) -> Optional[Dict]:
        """Get recommendations."""
        if model_name:
            key = f"{user_id}_{model_name}"
            return self.recommendations.get(key)
        
        # Return latest recommendation if no model specified
        for key, rec in self.recommendations.items():
            if rec['user_id'] == user_id:
                return rec
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics."""
        return {
            'users': len(self.users),
            'ratings': len(self.ratings),
            'recommendations': len(self.recommendations)
        }
    
    def close(self) -> None:
        """Close (no-op for in-memory)."""
        pass
