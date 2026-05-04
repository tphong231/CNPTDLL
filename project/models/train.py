"""
Model training script for the recommendation system.
Prepares data, trains models, and evaluates performance.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
import os
from typing import Dict, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocess import DataPreprocessor
from utils.metrics import EvaluationReport, RecommendationMetrics
from utils.visualization import RecommendationVisualizer
from models.recommend import RecommenderFactory
from db.db import MongoDBHandler, InMemoryDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RecommendationTrainer:
    """
    End-to-end training pipeline for recommendation system.
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models/saved_models",
                 use_dummy: bool = False):
        """
        Initialize trainer.
        
        Args:
            data_dir: Data directory path
            models_dir: Model save directory
            use_dummy: Whether to use dummy data
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.use_dummy = use_dummy
        
        # Create directories
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = DataPreprocessor(data_dir)
        self.evaluator = EvaluationReport()
        self.visualizer = RecommendationVisualizer()
        self.db = self._init_database()
        
        logger.info("RecommendationTrainer initialized")
    
    def _init_database(self) -> MongoDBHandler:
        """Initialize database (MongoDB or in-memory fallback)."""
        try:
            db = MongoDBHandler()
            if db.is_connected():
                db.create_indexes()
                return db
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {str(e)}. Using in-memory DB.")
        
        return InMemoryDatabase()
    
    def load_data(self, use_movielens: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load or create dataset.
        
        Args:
            use_movielens: Whether to use MovieLens dataset
            
        Returns:
            Tuple of (ratings_df, movies_df)
        """
        logger.info("Loading data...")
        
        if use_movielens:
            logger.info("Loading MovieLens dataset...")
            ratings_df, movies_df = self.preprocessor.load_movielens_dataset()
        elif self.use_dummy:
            logger.info("Creating dummy dataset...")
            ratings_df = self.preprocessor.create_dummy_dataset(
                n_users=100,
                n_movies=50,
                n_ratings=1000
            )
            movies_df = self.preprocessor.create_movie_metadata(n_movies=50)
        else:
            # Try to load from data/raw directory
            try:
                ratings_file = Path(self.data_dir) / "raw" / "ratings.csv"
                movies_file = Path(self.data_dir) / "raw" / "movies.csv"
                
                ratings_df = self.preprocessor.load_dataset(str(ratings_file))
                movies_df = self.preprocessor.load_dataset(str(movies_file))
                logger.info("Loaded custom dataset from files")
            except FileNotFoundError:
                logger.warning("No dataset files found. Using dummy data.")
                ratings_df = self.preprocessor.create_dummy_dataset()
                movies_df = self.preprocessor.create_movie_metadata()
        
        logger.info(f"Data loaded - Ratings: {ratings_df.shape}, Movies: {movies_df.shape}")
        return ratings_df, movies_df
        else:
            # Try to load from file
            ratings_path = Path(self.data_dir) / "raw" / "ratings.csv"
            movies_path = Path(self.data_dir) / "raw" / "movies.csv"
            
            if not ratings_path.exists() or not movies_path.exists():
                logger.info("Dataset files not found. Creating dummy data...")
                ratings_df = self.preprocessor.create_dummy_dataset()
                movies_df = self.preprocessor.create_movie_metadata()
                
                # Save for future use
                ratings_df.to_csv(ratings_path, index=False)
                movies_df.to_csv(movies_path, index=False)
            else:
                ratings_df = self.preprocessor.load_dataset(str(ratings_path))
                movies_df = self.preprocessor.load_dataset(str(movies_path))
        
        return ratings_df, movies_df
    
    def preprocess_data(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the ratings data.
        
        Args:
            ratings_df: Raw ratings DataFrame
            
        Returns:
            Cleaned ratings DataFrame
        """
        logger.info("Preprocessing data...")
        
        cleaned_df = self.preprocessor.clean_data(ratings_df)
        
        # Save processed data
        self.preprocessor.save_processed_data(cleaned_df, 'ratings_processed.csv')
        
        # Create user-item matrix
        user_item_matrix = self.preprocessor.create_user_item_matrix(cleaned_df)
        self.preprocessor.save_matrix(user_item_matrix, 'user_item_matrix.csv')
        
        logger.info(f"Data preprocessing complete. Final shape: {cleaned_df.shape}")
        
        return cleaned_df
    
    def train_models(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None) \
            -> Dict[str, object]:
        """
        Train recommendation models.
        
        Args:
            ratings_df: Processed ratings DataFrame
            movies_df: Movies metadata DataFrame
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training models...")
        
        models = {}
        
        # Calculate user-genre preferences if available
        user_genre_prefs = None
        if movies_df is not None:
            try:
                user_genre_prefs = self.preprocessor.get_user_genre_preferences(
                    ratings_df, movies_df
                )
            except Exception as e:
                logger.warning(f"Could not calculate genre preferences: {str(e)}")
        
        # Train content-based recommender
        try:
            logger.info("Training Content-Based recommender...")
            content_model = RecommenderFactory.create_recommender('content_based')
            content_model.train(ratings_df, user_genre_prefs)
            models['content_based'] = content_model
            
            # Save model
            model_path = Path(self.models_dir) / 'content_based_model.pkl'
            content_model.save_model(str(model_path))
        except Exception as e:
            logger.error(f"Error training content-based model: {str(e)}")
        
        # Train collaborative filtering recommender
        try:
            logger.info("Training Collaborative Filtering recommender...")
            cf_model = RecommenderFactory.create_recommender(
                'collaborative',
                n_factors=int(os.getenv('N_FACTORS', 50)),
                n_epochs=int(os.getenv('N_EPOCHS', 20))
            )
            cf_model.train(ratings_df)
            models['collaborative'] = cf_model
            
            # Save model
            model_path = Path(self.models_dir) / 'collaborative_model.pkl'
            cf_model.save_model(str(model_path))
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {str(e)}")
        
        # Note: NCF (Neural Collaborative Filtering) requires TensorFlow
        # For now, NCF falls back to collaborative filtering above
        try:
            logger.info("Training Hybrid recommender...")
            hybrid_model = RecommenderFactory.create_recommender('hybrid')
            hybrid_model.train(ratings_df, user_genre_prefs)
            models['hybrid'] = hybrid_model
            
            # Save model
            model_path = Path(self.models_dir) / 'hybrid_model.pkl'
            hybrid_model.save_model(str(model_path))
        except Exception as e:
            logger.error(f"Error training hybrid model: {str(e)}")
        
        logger.info(f"Training complete. {len(models)} models trained.")
        
        return models
    
    def evaluate_models(self, models: Dict[str, object], ratings_df: pd.DataFrame) -> None:
        """
        Evaluate trained models.
        
        Args:
            models: Dictionary of trained models
            ratings_df: Test ratings data
        """
        logger.info("Evaluating models...")
        
        # Split data for evaluation
        sample_users = ratings_df['user_id'].unique()[:20]
        test_ratings = ratings_df[ratings_df['user_id'].isin(sample_users)]
        
        for model_name, model in models.items():
            try:
                logger.info(f"\nEvaluating {model_name}...")
                
                # Generate recommendations
                recommendations = {}
                relevant_items = {}
                
                for user_id in sample_users:
                    user_ratings = test_ratings[test_ratings['user_id'] == user_id]
                    
                    if len(user_ratings) > 0:
                        recommendations[user_id] = model.recommend(user_id, n_recommendations=10)
                        relevant_items[user_id] = user_ratings['movie_id'].tolist()
                
                # Evaluate ranking metrics
                if recommendations and relevant_items:
                    ranking_metrics = self.evaluator.evaluate_ranking(
                        recommendations, relevant_items, k_values=[5, 10]
                    )
                    
                    logger.info(f"{model_name} Ranking Metrics:")
                    for k_val, metrics in ranking_metrics.items():
                        logger.info(f"  {k_val}: {metrics}")
                        
                        # Visualize metrics
                        try:
                            self.visualizer.plot_metrics_heatmap(
                                ranking_metrics,
                                title=f"{model_name} - Ranking Metrics",
                                save_path=f"logs/{model_name}_metrics.png"
                            )
                        except:
                            pass  # Visualization is optional
            
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
    
    def generate_final_recommendations(self, models: Dict[str, object], 
                                      n_users: int = 10, n_recs: int = 5) -> None:
        """
        Generate and store final recommendations.
        
        Args:
            models: Trained models dictionary
            n_users: Number of users to generate recommendations for
            n_recs: Number of recommendations per user
        """
        logger.info("Generating final recommendations...")
        
        for model_name, model in models.items():
            try:
                logger.info(f"\nGenerating recommendations with {model_name}...")
                
                # Get sample users
                all_users = model.user_item_matrix.index.tolist()[:n_users] \
                    if hasattr(model, 'user_item_matrix') else range(1, n_users + 1)
                
                for user_id in all_users:
                    recommendations = model.recommend(user_id, n_recommendations=n_recs)
                    
                    # Store recommendations in database
                    self.db.save_recommendation(
                        user_id=user_id,
                        recommended_movies=recommendations,
                        model_name=model_name
                    )
                
                logger.info(f"Stored recommendations for {len(all_users)} users using {model_name}")
            
            except Exception as e:
                logger.error(f"Error generating recommendations with {model_name}: {str(e)}")
    
    def train(self, use_movielens: bool = False) -> Dict[str, object]:
        """
        Complete training pipeline with MovieLens support.
        
        Args:
            use_movielens: Whether to use MovieLens dataset
            
        Returns:
            Dictionary containing trained models and evaluation results
        """
        logger.info("Starting training pipeline...")
        
        try:
            # Load data
            ratings_df, movies_df = self.load_data(use_movielens)
            
            # Preprocess
            ratings_df = self.preprocess_data(ratings_df)
            
            # Train models
            models = self.train_models(ratings_df, movies_df)
            
            if not models:
                logger.error("No models were successfully trained.")
                return {}
            
            # Evaluate
            self.evaluate_models(models, ratings_df)
            
            # Generate recommendations
            self.generate_final_recommendations(models)
            
            # Print database stats
            stats = self.db.get_stats()
            logger.info(f"Database stats: {stats}")
            
            logger.info("Training pipeline complete!")
            
            return {
                'models': models,
                'data_stats': {
                    'ratings_count': len(ratings_df),
                    'users_count': ratings_df['user_id'].nunique(),
                    'movies_count': ratings_df['movie_id'].nunique(),
                    'dataset_type': 'MovieLens' if use_movielens else 'Dummy'
                }
            }
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return {}
        finally:
            self.db.close()
    
    def run_pipeline(self) -> None:
        """Execute complete training pipeline."""
        logger.info("Starting recommendation system training pipeline...")
        
        try:
            # Load data
            ratings_df, movies_df = self.load_data()
            
            # Preprocess
            ratings_df = self.preprocess_data(ratings_df)
            
            # Train models
            models = self.train_models(ratings_df, movies_df)
            
            if not models:
                logger.error("No models were successfully trained.")
                return
            
            # Evaluate
            self.evaluate_models(models, ratings_df)
            
            # Generate recommendations
            self.generate_final_recommendations(models)
            
            # Print database stats
            stats = self.db.get_stats()
            logger.info(f"Database stats: {stats}")
            
            logger.info("Training pipeline complete!")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        finally:
            self.db.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Train recommendation models')
    parser.add_argument('--dummy', action='store_true', help='Use dummy data')
    parser.add_argument('--movielens', action='store_true', help='Use MovieLens dataset')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--models-dir', default='models/saved_models', help='Models directory')
    
    args = parser.parse_args()
    
    trainer = RecommendationTrainer(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        use_dummy=args.dummy
    )
    
    if args.movielens:
        logger.info("Using MovieLens dataset for training...")
        result = trainer.train(use_movielens=True)
    else:
        trainer.run_pipeline()


if __name__ == '__main__':
    main()
