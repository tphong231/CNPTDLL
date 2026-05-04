"""
Visualization utilities for recommendation system.
Provides charts and plots for analysis and presentation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings

logger = logging.getLogger(__name__)

# Try to import plotting libraries, gracefully fail if not available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Charts will not be generated.")


class RecommendationVisualizer:
    """
    Generate visualizations for recommendation system analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: tuple = (12, 6)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(style)
            except:
                logger.warning(f"Style '{style}' not available, using default")
            sns.set_palette("husl")
    
    def plot_rating_distribution(self, ratings: np.ndarray, 
                                title: str = "Rating Distribution",
                                save_path: Optional[str] = None) -> None:
        """
        Plot distribution of ratings.
        
        Args:
            ratings: Array of rating values
            title: Plot title
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot: Matplotlib not available")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(ratings, bins=5, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_sparsity_analysis(self, user_item_matrix: pd.DataFrame,
                              title: str = "Rating Sparsity",
                              save_path: Optional[str] = None) -> None:
        """
        Analyze and visualize matrix sparsity.
        
        Args:
            user_item_matrix: User-item rating matrix
            title: Plot title
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot: Matplotlib not available")
            return
        
        # Calculate sparsity per user and item
        user_ratings = (user_item_matrix > 0).sum(axis=1)
        item_ratings = (user_item_matrix > 0).sum(axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # User sparsity
        axes[0].hist(user_ratings, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Number of Ratings per User')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('User Rating Distribution')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Item sparsity
        axes[1].hist(item_ratings, bins=20, edgecolor='black', alpha=0.7, color='coral')
        axes[1].set_xlabel('Number of Ratings per Item')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Item Rating Distribution')
        axes[1].grid(axis='y', alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics_data: Dict[str, Dict[str, float]],
                               metric_name: str = 'precision',
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Plot comparison of metrics across different k values.
        
        Args:
            metrics_data: Dict with structure {'@5': {...}, '@10': {...}, ...}
            metric_name: Which metric to plot
            title: Plot title
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot: Matplotlib not available")
            return
        
        k_values = []
        metric_values = []
        
        for k_str, metrics_dict in sorted(metrics_data.items()):
            k_values.append(k_str)
            metric_values.append(metrics_dict.get(metric_name, 0))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.bar(k_values, metric_values, color='steelblue', edgecolor='black', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
        
        ax.set_xlabel('Top-K')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(title or f'{metric_name.capitalize()} @ K')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_heatmap(self, metrics_data: Dict[str, Dict[str, float]],
                            title: str = "Metrics Heatmap",
                            save_path: Optional[str] = None) -> None:
        """
        Plot heatmap of all metrics for different k values.
        
        Args:
            metrics_data: Dict with structure {'@5': {...}, '@10': {...}, ...}
            title: Plot title
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot: Matplotlib not available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_data).T
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        sns.heatmap(df, annot=True, fmt='.4f', cmap='YlGn', ax=ax, 
                   cbar_kws={'label': 'Metric Value'})
        
        ax.set_title(title)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Top-K')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_user_item_heatmap(self, user_item_matrix: pd.DataFrame,
                              max_users: int = 20,
                              max_items: int = 20,
                              title: str = "User-Item Rating Matrix (Sample)",
                              save_path: Optional[str] = None) -> None:
        """
        Plot heatmap of user-item matrix (limited sample).
        
        Args:
            user_item_matrix: User-item rating matrix
            max_users: Maximum users to display
            max_items: Maximum items to display
            title: Plot title
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot: Matplotlib not available")
            return
        
        # Sample the matrix
        sampled = user_item_matrix.iloc[:min(max_users, len(user_item_matrix)),
                                       :min(max_items, len(user_item_matrix.columns))]
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], 8))
        
        sns.heatmap(sampled, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Rating'})
        
        ax.set_title(title)
        ax.set_xlabel('Movie ID')
        ax.set_ylabel('User ID')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_rmse_convergence(self, rmse_history: List[float],
                             title: str = "RMSE During Training",
                             save_path: Optional[str] = None) -> None:
        """
        Plot RMSE convergence during training.
        
        Args:
            rmse_history: List of RMSE values per epoch
            title: Plot title
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot: Matplotlib not available")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(rmse_history) + 1)
        ax.plot(epochs, rmse_history, marker='o', linewidth=2, markersize=4, color='steelblue')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_genre_preference_heatmap(self, user_genre_prefs: pd.DataFrame,
                                     max_users: int = 15,
                                     title: str = "User-Genre Preferences",
                                     save_path: Optional[str] = None) -> None:
        """
        Plot user genre preference heatmap.
        
        Args:
            user_genre_prefs: User-genre preference matrix
            max_users: Maximum users to display
            title: Plot title
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot: Matplotlib not available")
            return
        
        sampled = user_genre_prefs.iloc[:min(max_users, len(user_genre_prefs))]
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], 8))
        
        sns.heatmap(sampled, cmap='coolwarm', center=0, ax=ax, 
                   cbar_kws={'label': 'Average Rating'})
        
        ax.set_title(title)
        ax.set_xlabel('Genre')
        ax.set_ylabel('User ID')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_recommendation_coverage(self, coverage_data: Dict[str, float],
                                    title: str = "Recommendation Coverage",
                                    save_path: Optional[str] = None) -> None:
        """
        Plot coverage metrics.
        
        Args:
            coverage_data: Dict with model names and coverage values
            title: Plot title
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot: Matplotlib not available")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        models = list(coverage_data.keys())
        coverage_values = list(coverage_data.values())
        
        bars = ax.barh(models, coverage_values, color='teal', edgecolor='black', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.2%}', ha='left', va='center')
        
        ax.set_xlabel('Coverage')
        ax.set_title(title)
        ax.set_xlim([0, 1.1])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_top_rated_movies(self, ratings_df: pd.DataFrame,
                             n_movies: int = 10,
                             title: str = "Top Rated Movies",
                             save_path: Optional[str] = None) -> None:
        """
        Plot top-rated movies by average rating.
        
        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating]
            n_movies: Number of top movies to show
            title: Plot title
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot: Matplotlib not available")
            return
        
        # Calculate average rating per movie
        movie_stats = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count'])
        movie_stats = movie_stats.sort_values('mean', ascending=False).head(n_movies)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.barh(range(len(movie_stats)), movie_stats['mean'], color='coral', edgecolor='black', alpha=0.7)
        
        # Add value labels
        for i, (idx, row) in enumerate(movie_stats.iterrows()):
            ax.text(row['mean'], i, f" {row['mean']:.2f} ({int(row['count'])} ratings)",
                   va='center', fontsize=9)
        
        ax.set_yticks(range(len(movie_stats)))
        ax.set_yticklabels([f"Movie {mid}" for mid in movie_stats.index])
        ax.set_xlabel('Average Rating')
        ax.set_title(title)
        ax.set_xlim([0, 5.5])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
