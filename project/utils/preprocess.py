"""
Data preprocessing utilities for the recommendation system.
Handles data loading, cleaning, and transformation.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing tasks including loading, cleaning,
    and creating user-item matrices.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        logger.info(f"DataPreprocessor initialized with data_dir: {data_dir}")
    
    def create_dummy_dataset(self, n_users: int = 100, n_movies: int = 50,
                            n_ratings: int = 1000, seed: int = 42) -> pd.DataFrame:
        """
        Create a dummy dataset for testing and development.
        
        Args:
            n_users: Number of users
            n_movies: Number of movies
            n_ratings: Number of ratings
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with columns: user_id, movie_id, rating, timestamp
        """
        np.random.seed(seed)
        logger.info(f"Creating dummy dataset with {n_users} users, {n_movies} movies, {n_ratings} ratings")
        
        ratings = []
        for _ in range(n_ratings):
            user_id = np.random.randint(1, n_users + 1)
            movie_id = np.random.randint(1, n_movies + 1)
            rating = np.random.randint(1, 6)  # Ratings 1-5
            timestamp = np.random.randint(1000000000, 1700000000)
            ratings.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        df = pd.DataFrame(ratings)
        df = df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
        logger.info(f"Dummy dataset created with shape: {df.shape}")
        
        return df
    
    def create_movie_metadata(self, n_movies: int = 50) -> pd.DataFrame:
        """
        Create dummy movie metadata.
        
        Args:
            n_movies: Number of movies
            
        Returns:
            DataFrame with movie information
        """
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'SciFi', 'Romance']
        movies = []
        
        for movie_id in range(1, n_movies + 1):
            selected_genres = np.random.choice(genres, size=np.random.randint(1, 4), replace=False)
            movies.append({
                'movie_id': movie_id,
                'title': f'Movie {movie_id}',
                'genres': '|'.join(selected_genres),
                'release_year': np.random.randint(1990, 2024)
            })
        
        return pd.DataFrame(movies)
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            filepath: Path to the dataset file (CSV)
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading dataset from {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def load_movielens_dataset(self, data_dir: str = "data/raw/movielens/ml-latest-small") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens dataset (real movie recommendation data).
        
        Args:
            data_dir: Directory containing MovieLens files
            
        Returns:
            Tuple of (ratings_df, movies_df)
        """
        logger.info(f"Loading MovieLens dataset from {data_dir}")
        
        try:
            # Load ratings
            ratings_path = Path(data_dir) / "ratings.csv"
            ratings_df = pd.read_csv(ratings_path)
            logger.info(f"Loaded ratings: {ratings_df.shape}")
            
            # Load movies
            movies_path = Path(data_dir) / "movies.csv"
            movies_df = pd.read_csv(movies_path)
            logger.info(f"Loaded movies: {movies_df.shape}")
            
            # Convert timestamp to datetime
            ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
            
            # Extract genres
            movies_df['genres_list'] = movies_df['genres'].str.split('|')
            
            logger.info("MovieLens dataset loaded successfully!")
            return ratings_df, movies_df
            
        except FileNotFoundError as e:
            logger.error(f"MovieLens files not found in {data_dir}: {str(e)}")
            logger.info("Falling back to dummy dataset...")
            return self.create_dummy_dataset(), self.create_movie_metadata()
        except Exception as e:
            logger.error(f"Error loading MovieLens dataset: {str(e)}")
            logger.info("Falling back to dummy dataset...")
            return self.create_dummy_dataset(), self.create_movie_metadata()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and duplicates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning")
        
        # Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
        logger.info(f"Removed duplicates. Shape: {initial_shape} -> {df.shape}")
        
        # Remove rows with missing values
        df = df.dropna()
        logger.info(f"Removed null values. Shape: {df.shape}")
        
        # Ensure rating is in valid range
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
        logger.info(f"Filtered invalid ratings. Shape: {df.shape}")
        
        return df
    
    def create_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a user-item matrix from ratings data.
        
        Args:
            df: DataFrame with user_id, movie_id, and rating columns
            
        Returns:
            User-item matrix (users as rows, movies as columns)
        """
        logger.info("Creating user-item matrix")
        
        user_item_matrix = df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        logger.info(f"User-item matrix created. Shape: {user_item_matrix.shape}")
        logger.info(f"Sparsity: {1 - (df.shape[0] / (user_item_matrix.shape[0] * user_item_matrix.shape[1])):.2%}")
        
        return user_item_matrix
    
    def get_user_genre_preferences(self, ratings_df: pd.DataFrame, 
                                   movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate user preferences for each genre based on their ratings.
        
        Args:
            ratings_df: DataFrame with ratings
            movies_df: DataFrame with movie metadata
            
        Returns:
            DataFrame with user genre preferences
        """
        logger.info("Calculating user genre preferences")
        
        # Merge ratings with movie metadata
        merged = ratings_df.merge(movies_df, on='movie_id')
        
        # Explode genres (handle pipe-separated genres)
        merged['genre'] = merged['genres'].str.split('|')
        merged = merged.explode('genre')
        merged['genre'] = merged['genre'].str.strip()
        
        # Calculate average rating by user and genre
        user_genre_prefs = merged.groupby(['user_id', 'genre'])['rating'].mean().unstack(fill_value=0)
        
        logger.info(f"User genre preferences shape: {user_genre_prefs.shape}")
        
        return user_genre_prefs
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.processed_data_dir / filename
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def save_matrix(self, matrix: pd.DataFrame, filename: str) -> None:
        """
        Save matrix to file.
        
        Args:
            matrix: Matrix to save
            filename: Output filename
        """
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.processed_data_dir / filename
        
        try:
            matrix.to_csv(filepath)
            logger.info(f"Matrix saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving matrix: {str(e)}")
            raise
