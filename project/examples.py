"""
Example usage of the recommendation system.
Demonstrates how to use models, database, and evaluation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocess import DataPreprocessor
from utils.metrics import EvaluationReport, RecommendationMetrics
from utils.visualization import RecommendationVisualizer
from models.recommend import RecommenderFactory
from db.db import InMemoryDatabase


def example_1_create_dummy_data():
    """Example 1: Create and explore dummy dataset."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Create Dummy Dataset")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    
    # Create dummy data
    ratings_df = preprocessor.create_dummy_dataset(
        n_users=50,
        n_movies=30,
        n_ratings=500
    )
    
    print(f"\nDataset shape: {ratings_df.shape}")
    print(f"\nFirst 5 rows:")
    print(ratings_df.head())
    
    print(f"\nRating statistics:")
    print(ratings_df['rating'].describe())
    
    # Create movies metadata
    movies_df = preprocessor.create_movie_metadata(n_movies=30)
    print(f"\nMovies metadata:")
    print(movies_df.head())
    
    return ratings_df, movies_df


def example_2_train_models(ratings_df: pd.DataFrame):
    """Example 2: Train recommendation models."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Train Recommendation Models")
    print("="*60)
    
    models = {}
    
    # Train content-based
    print("\nTraining Content-Based Recommender...")
    content_model = RecommenderFactory.create_recommender('content_based')
    content_model.train(ratings_df)
    models['content_based'] = content_model
    print("✓ Content-based trained")
    
    # Train collaborative filtering
    print("\nTraining Collaborative Filtering (SVD)...")
    try:
        cf_model = RecommenderFactory.create_recommender('collaborative')
        cf_model.train(ratings_df)
        models['collaborative'] = cf_model
        print("✓ Collaborative filtering trained")
    except Exception as e:
        print(f"✗ Could not train CF: {str(e)}")
    
    # Train hybrid
    print("\nTraining Hybrid Recommender...")
    hybrid_model = RecommenderFactory.create_recommender('hybrid')
    hybrid_model.train(ratings_df)
    models['hybrid'] = hybrid_model
    print("✓ Hybrid trained")
    
    return models


def example_3_generate_recommendations(models: dict):
    """Example 3: Generate recommendations from models."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Generate Recommendations")
    print("="*60)
    
    user_id = 1
    n_recs = 5
    
    for model_name, model in models.items():
        print(f"\n{model_name.upper()} Recommendations for User {user_id}:")
        recommendations = model.recommend(user_id, n_recommendations=n_recs)
        
        if recommendations:
            for i, movie_id in enumerate(recommendations, 1):
                print(f"  {i}. Movie {movie_id}")
        else:
            print("  No recommendations available")


def example_4_evaluate_models(models: dict, ratings_df: pd.DataFrame):
    """Example 4: Evaluate model performance."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Evaluate Models")
    print("="*60)
    
    # Sample test users
    test_users = ratings_df['user_id'].unique()[:10]
    
    for model_name, model in models.items():
        print(f"\n{model_name.upper()} Evaluation:")
        
        recommendations = {}
        relevant_items = {}
        
        for user_id in test_users:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            if len(user_ratings) > 0:
                recommendations[user_id] = model.recommend(user_id, n_recommendations=5)
                relevant_items[user_id] = user_ratings['movie_id'].tolist()
        
        # Evaluate
        evaluator = EvaluationReport()
        results = evaluator.evaluate_ranking(
            recommendations,
            relevant_items,
            k_values=[5]
        )
        
        print(f"  Metrics @ 5:")
        for metric, value in results.get('@5', {}).items():
            print(f"    {metric.capitalize()}: {value:.4f}")


def example_5_database_operations():
    """Example 5: Database operations."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Database Operations")
    print("="*60)
    
    db = InMemoryDatabase()
    
    # Create users
    print("\nCreating users...")
    db.create_user(user_id=1, username="alice", email="alice@example.com")
    db.create_user(user_id=2, username="bob", email="bob@example.com")
    print("✓ Users created")
    
    # Add ratings
    print("\nAdding ratings...")
    db.add_rating(user_id=1, movie_id=101, rating=5)
    db.add_rating(user_id=1, movie_id=102, rating=4)
    db.add_rating(user_id=2, movie_id=101, rating=3)
    print("✓ Ratings added")
    
    # Get user ratings
    print("\nRetrieving user ratings...")
    user1_ratings = db.get_user_ratings(1)
    print(f"  User 1 has {len(user1_ratings)} ratings")
    
    # Save recommendations
    print("\nSaving recommendations...")
    db.save_recommendation(
        user_id=1,
        recommended_movies=[103, 104, 105],
        model_name='hybrid',
        confidence=[0.95, 0.87, 0.82]
    )
    print("✓ Recommendations saved")
    
    # Get recommendations
    print("\nRetrieving recommendations...")
    recs = db.get_recommendations(user_id=1, model_name='hybrid')
    if recs:
        print(f"  Recommended: {recs['recommended_movies']}")
    
    # Stats
    print("\nDatabase statistics:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_6_metrics_calculation():
    """Example 6: Calculate evaluation metrics."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Metrics Calculation")
    print("="*60)
    
    metrics = RecommendationMetrics()
    
    # Sample data
    recommended = [1, 2, 3, 4, 5]
    relevant = [2, 3, 5, 7, 9]
    
    print(f"\nRecommended items: {recommended}")
    print(f"Relevant items: {relevant}")
    
    # Calculate metrics
    print(f"\nMetrics @ 5:")
    print(f"  Precision@5: {metrics.precision_at_k(recommended, relevant, 5):.4f}")
    print(f"  Recall@5: {metrics.recall_at_k(recommended, relevant, 5):.4f}")
    print(f"  F1@5: {metrics.f1_at_k(recommended, relevant, 5):.4f}")
    print(f"  NDCG@5: {metrics.ndcg_at_k(recommended, relevant, 5):.4f}")
    print(f"  MAP@5: {metrics.map_at_k(recommended, relevant, 5):.4f}")
    
    # Metrics @ K
    print(f"\nMetrics across K values:")
    for k in [1, 3, 5]:
        precision = metrics.precision_at_k(recommended, relevant, k)
        recall = metrics.recall_at_k(recommended, relevant, k)
        print(f"  K={k}: Precision={precision:.4f}, Recall={recall:.4f}")


def example_7_save_and_load_models(models: dict):
    """Example 7: Save and load models."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Save and Load Models")
    print("="*60)
    
    # Create models directory
    models_dir = Path("models/saved_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    print("\nSaving models...")
    for model_name, model in models.items():
        filepath = models_dir / f'{model_name}_example.pkl'
        success = model.save_model(str(filepath))
        
        if success:
            print(f"✓ {model_name} saved to {filepath}")
    
    # Load models
    print("\nLoading models...")
    for model_name in models.keys():
        filepath = models_dir / f'{model_name}_example.pkl'
        
        if filepath.exists():
            loaded_model = type(models[model_name]).load_model(str(filepath))
            
            if loaded_model:
                print(f"✓ {model_name} loaded from {filepath}")
                # Test loaded model
                test_rec = loaded_model.recommend(1, n_recommendations=3)
                if test_rec:
                    print(f"  Sample recommendations: {test_rec}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  RECOMMENDATION SYSTEM - USAGE EXAMPLES".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # Example 1: Create data
        ratings_df, movies_df = example_1_create_dummy_data()
        
        # Example 2: Train models
        models = example_2_train_models(ratings_df)
        
        # Example 3: Generate recommendations
        example_3_generate_recommendations(models)
        
        # Example 4: Evaluate
        example_4_evaluate_models(models, ratings_df)
        
        # Example 5: Database
        example_5_database_operations()
        
        # Example 6: Metrics
        example_6_metrics_calculation()
        
        # Example 7: Save/Load
        example_7_save_and_load_models(models)
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"\n✗ Error during examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
