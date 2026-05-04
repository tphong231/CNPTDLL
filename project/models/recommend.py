"""
Recommendation models implementation.
Includes content-based, collaborative filtering, and hybrid approaches.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import Surprise
try:
    from surprise import SVD, Dataset, Reader  # type: ignore[import]
    from surprise.model_selection import train_test_split  # type: ignore[import]
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    logger.warning("Surprise library not available. SVD recommendations will be limited.")

# Try to import TensorFlow for NCF
try:
    from models.recommend_ncf import NeuralCollaborativeFiltering
    NCF_AVAILABLE = True
except ImportError:
    NCF_AVAILABLE = False
    logger.warning("NCF model not available. Neural CF will fallback to collaborative filtering.")

class BaseRecommender(ABC):
    """
    Abstract base class for recommendation models.
    """
    
    def __init__(self, name: str = "BaseRecommender"):
        """
        Initialize recommender.
        
        Args:
            name: Model name
        """
        self.name = name
        self.is_trained = False
        logger.info(f"Initialized {name}")
    
    @abstractmethod
    def train(self, ratings_df: pd.DataFrame) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """Generate recommendations for a user."""
        pass
    
    def save_model(self, filepath: str) -> bool:
        """Save model to disk."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    @staticmethod
    def load_model(filepath: str) -> Optional['BaseRecommender']:
        """Load model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None


class UnavailableRecommender(BaseRecommender):
    """
    Stand-in recommender when a required dependency cannot be loaded.
    """

    def __init__(self, name: str = "UnavailableRecommender", reason: str = None):
        super().__init__(name)
        self.reason = reason

    def train(self, ratings_df, *args, **kwargs) -> None:
        logger.warning(
            f"{self.name} cannot train because it is unavailable. Reason: {self.reason}"
        )

    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        logger.warning(
            f"{self.name} cannot recommend because it is unavailable. Reason: {self.reason}"
        )
        return []

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        logger.warning(
            f"{self.name} cannot predict because it is unavailable. Reason: {self.reason}"
        )
        return 0.0


class ContentBasedRecommender(BaseRecommender):
    """
    Content-based recommendation using item features.
    Recommends items similar to those the user has rated highly.
    """
    
    def __init__(self, min_common_features: int = 1):
        """
        Initialize content-based recommender.
        
        Args:
            min_common_features: Minimum shared features for similarity
        """
        super().__init__("ContentBased")
        self.user_item_matrix = None
        self.item_features = None
        self.user_preferences = None
        self.min_common_features = min_common_features
    
    def train(self, ratings_df: pd.DataFrame, item_features: pd.DataFrame = None) -> None:
        """
        Train content-based model.
        
        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating]
            item_features: DataFrame with item features (optional)
        """
        logger.info("Training ContentBasedRecommender")
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        # Store item features
        self.item_features = item_features
        
        # Calculate user preferences for items they've rated
        self.user_preferences = self.user_item_matrix.copy()
        
        self.is_trained = True
        logger.info(f"ContentBasedRecommender trained. Users: {len(self.user_preferences)}")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """
        Generate recommendations based on user's rated items.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended movie IDs
        """
        if not self.is_trained:
            logger.warning("Model not trained. Returning empty list.")
            return []
        
        if user_id not in self.user_preferences.index:
            logger.warning(f"User {user_id} not in training data. Using popular items fallback.")
            return self._get_popular_items(n_recommendations)
        
        # Get items user has already rated
        user_rated = self.user_preferences.loc[user_id]
        user_rated = user_rated[user_rated > 0].index.tolist()
        
        # Get candidate items (not rated by user)
        all_items = self.user_preferences.columns.tolist()
        candidate_items = [item for item in all_items if item not in user_rated]
        
        if not candidate_items:
            return []
        
        # Score candidates based on similarity to liked items
        scores = {}
        for candidate in candidate_items:
            score = 0.0
            for liked_item in user_rated:
                # Simple similarity: both non-zero means similarity
                if liked_item != candidate:
                    score += self.user_preferences.loc[user_id, liked_item] / len(user_rated)
            scores[candidate] = score
        
        # Sort by score and return top N
        sorted_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item for item, _ in sorted_recommendations[:n_recommendations]]
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    
    def _get_popular_items(self, n: int = 5) -> List[int]:
        """Get most popular items across all users."""
        if self.user_item_matrix is None:
            return []
        
        popularity = (self.user_item_matrix > 0).sum()
        return popularity.nlargest(n).index.tolist()


class CollaborativeFilteringRecommender(BaseRecommender):
    """
    Collaborative filtering using SVD from Surprise library.
    Predicts ratings and recommends highly predicted items.
    """
    
    def __init__(self, n_factors: int = 50, n_epochs: int = 20, 
                 learning_rate: float = 0.005, regularization: float = 0.02):
        """
        Initialize SVD-based recommender.
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            learning_rate: Learning rate for SGD
            regularization: Regularization parameter
        """
        if not SURPRISE_AVAILABLE:
            raise ImportError("Surprise library required. Install with: pip install scikit-surprise")
        
        super().__init__("CollaborativeFilteringSVD")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.model = None
        self.trainset = None
        self.user_item_matrix = None
    
    def train(self, ratings_df: pd.DataFrame) -> None:
        """
        Train SVD model using Surprise.
        
        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating]
        """
        logger.info("Training CollaborativeFilteringRecommender (SVD)")
        
        try:
            # Create Surprise dataset
            reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
            dataset = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
            self.trainset = dataset.build_full_trainset()
            
            # Initialize and train SVD model
            self.model = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.learning_rate,
                reg_all=self.regularization,
                random_state=42
            )
            self.model.fit(self.trainset)
            
            # Store user-item matrix for fallback
            self.user_item_matrix = ratings_df.pivot_table(
                index='user_id',
                columns='movie_id',
                values='rating',
                fill_value=0
            )
            
            self.is_trained = True
            logger.info(f"CollaborativeFilteringRecommender trained with {len(self.trainset.ur)} users")
        except Exception as e:
            logger.error(f"Error training SVD model: {str(e)}")
            self.is_trained = False
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """
        Generate recommendations using predicted ratings.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended movie IDs
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Returning empty list.")
            return []
        
        try:
            # Handle new users
            if user_id not in self.trainset.to_raw_uid.keys():
                logger.warning(f"User {user_id} not in training data. Using popular items.")
                return self._get_popular_items(n_recommendations)
            
            # Get all movies
            all_movies = self.trainset.to_raw_iid.values()
            
            # Get movies the user has already rated
            user_raw_id = self.trainset.to_inner_uid(user_id)
            rated_movies = [self.trainset.to_raw_iid(iid) 
                           for iid, _ in self.trainset.ur[user_raw_id]]
            
            # Predict ratings for unrated movies
            predictions = {}
            for movie_id in all_movies:
                if movie_id not in rated_movies:
                    try:
                        pred = self.model.predict(user_id, movie_id)
                        predictions[movie_id] = pred.est
                    except:
                        continue
            
            # Sort by predicted rating and return top N
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            recommendations = [movie for movie, _ in sorted_predictions[:n_recommendations]]
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a user-movie pair."""
        if not self.is_trained or self.model is None:
            return 0.0
        
        try:
            prediction = self.model.predict(user_id, movie_id)
            return prediction.est
        except:
            return 0.0
    
    def _get_popular_items(self, n: int = 5) -> List[int]:
        """Get most popular items."""
        if self.user_item_matrix is None:
            return []
        
        popularity = (self.user_item_matrix > 0).sum()
        return popularity.nlargest(n).index.tolist()


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender combining content-based and collaborative filtering.
    """
    
    def __init__(self, content_weight: float = 0.4, collaborative_weight: float = 0.6):
        """
        Initialize hybrid recommender.
        
        Args:
            content_weight: Weight for content-based recommendations
            collaborative_weight: Weight for collaborative filtering
        """
        super().__init__("Hybrid")
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = None
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        
        if SURPRISE_AVAILABLE:
            self.collaborative_recommender = CollaborativeFilteringRecommender()
    
    def train(self, ratings_df: pd.DataFrame, item_features: pd.DataFrame = None) -> None:
        """
        Train both recommenders.
        
        Args:
            ratings_df: DataFrame with ratings
            item_features: DataFrame with item features (optional)
        """
        logger.info("Training HybridRecommender")
        
        # Train content-based
        self.content_recommender.train(ratings_df, item_features)
        
        # Train collaborative filtering if available
        if self.collaborative_recommender:
            try:
                self.collaborative_recommender.train(ratings_df)
            except Exception as e:
                logger.warning(f"Could not train collaborative recommender: {str(e)}")
                self.collaborative_recommender = None
        
        self.is_trained = True
        logger.info("HybridRecommender trained")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """
        Generate hybrid recommendations.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended movie IDs
        """
        if not self.is_trained:
            logger.warning("Model not trained.")
            return []
        
        recommendations = {}
        
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend(user_id, n_recommendations * 2)
        for i, movie_id in enumerate(content_recs):
            score = (1.0 - i / len(content_recs)) * self.content_weight
            recommendations[movie_id] = recommendations.get(movie_id, 0) + score
        
        # Get collaborative recommendations
        if self.collaborative_recommender:
            collab_recs = self.collaborative_recommender.recommend(user_id, n_recommendations * 2)
            for i, movie_id in enumerate(collab_recs):
                score = (1.0 - i / len(collab_recs)) * self.collaborative_weight
                recommendations[movie_id] = recommendations.get(movie_id, 0) + score
        
        # Sort and return top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        final_recommendations = [movie for movie, _ in sorted_recs[:n_recommendations]]
        
        return final_recommendations


class RecommenderFactory:
    """
    Factory for creating recommender instances.
    """
    
    @staticmethod
    def create_recommender(model_type: str = 'hybrid', **kwargs) -> BaseRecommender:
        """
        Create a recommender instance.
        
        Args:
            model_type: Type of recommender ('content_based', 'collaborative', 'hybrid', 'ncf')
            **kwargs: Additional arguments for the recommender
            
        Returns:
            Recommender instance
        """
        if model_type == 'content_based':
            return ContentBasedRecommender(**kwargs)
        elif model_type == 'collaborative':
            if SURPRISE_AVAILABLE:
                return CollaborativeFilteringRecommender(**kwargs)
            logger.warning(
                "Collaborative recommender unavailable because scikit-surprise is missing. "
                "Using unavailable fallback model."
            )
            return UnavailableRecommender(
                name="CollaborativeFilteringUnavailable",
                reason="Missing scikit-surprise dependency"
            )
        elif model_type == 'ncf' or model_type == 'neural_cf':
            if NCF_AVAILABLE:
                return NeuralCollaborativeFiltering(**kwargs)
            logger.warning(
                "Neural CF model unavailable. Using collaborative filtering fallback."
            )
            return CollaborativeFilteringRecommender(**kwargs) if SURPRISE_AVAILABLE else UnavailableRecommender(
                name="NeuralCFUnavailable",
                reason="Missing dependencies"
            )
        elif model_type == 'hybrid':
            return HybridRecommender(**kwargs)
        else:
            logger.warning(f"Unknown model type: {model_type}. Using hybrid.")
            return HybridRecommender(**kwargs)
