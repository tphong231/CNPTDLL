"""
Evaluation metrics for recommendation systems.
Includes RMSE, Precision@K, Recall@K, NDCG, and more.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """
    Calculate evaluation metrics for recommendation systems.
    Supports both rating prediction (RMSE) and ranking metrics (Precision@K, Recall@K).
    """
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error for rating predictions.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            RMSE value
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        logger.info(f"RMSE calculated: {rmse:.4f}")
        return rmse
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error for rating predictions.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            MAE value
        """
        mae = np.mean(np.abs(y_true - y_pred))
        logger.info(f"MAE calculated: {mae:.4f}")
        return mae
    
    @staticmethod
    def precision_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
        """
        Calculate Precision@K: fraction of recommended items that are relevant.
        
        Args:
            recommended: List of recommended item IDs (top-k)
            relevant: List of relevant item IDs (ground truth)
            k: Number of recommendations to consider
            
        Returns:
            Precision@K value (0-1)
        """
        if k == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        hits = len(recommended_set.intersection(relevant_set))
        precision = hits / k if k > 0 else 0.0
        
        return precision
    
    @staticmethod
    def recall_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
        """
        Calculate Recall@K: fraction of relevant items that are recommended.
        
        Args:
            recommended: List of recommended item IDs (top-k)
            relevant: List of relevant item IDs (ground truth)
            k: Number of recommendations to consider
            
        Returns:
            Recall@K value (0-1)
        """
        if len(relevant) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        hits = len(recommended_set.intersection(relevant_set))
        recall = hits / len(relevant_set)
        
        return recall
    
    @staticmethod
    def f1_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
        """
        Calculate F1@K: harmonic mean of Precision@K and Recall@K.
        
        Args:
            recommended: List of recommended item IDs (top-k)
            relevant: List of relevant item IDs (ground truth)
            k: Number of recommendations to consider
            
        Returns:
            F1@K value (0-1)
        """
        precision = RecommendationMetrics.precision_at_k(recommended, relevant, k)
        recall = RecommendationMetrics.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def ndcg_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain).
        Measures ranking quality considering position of relevant items.
        
        Args:
            recommended: List of recommended item IDs (ranked)
            relevant: List of relevant item IDs (ground truth)
            k: Number of recommendations to consider
            
        Returns:
            NDCG@K value (0-1)
        """
        # Calculate DCG
        dcg = 0.0
        relevant_set = set(relevant)
        
        for i in range(min(k, len(recommended))):
            if recommended[i] in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant))):
            idcg += 1.0 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    @staticmethod
    def map_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
        """
        Calculate MAP@K (Mean Average Precision).
        
        Args:
            recommended: List of recommended item IDs (ranked)
            relevant: List of relevant item IDs (ground truth)
            k: Number of recommendations to consider
            
        Returns:
            MAP@K value (0-1)
        """
        relevant_set = set(relevant)
        ap = 0.0
        hits = 0
        
        for i in range(min(k, len(recommended))):
            if recommended[i] in relevant_set:
                hits += 1
                ap += hits / (i + 1)
        
        map_k = ap / min(k, len(relevant)) if len(relevant) > 0 else 0.0
        return map_k
    
    @staticmethod
    def coverage(recommended_items: List[List[int]], total_items: int) -> float:
        """
        Calculate catalog coverage: fraction of items recommended at least once.
        
        Args:
            recommended_items: List of recommendation lists for each user
            total_items: Total number of items in catalog
            
        Returns:
            Coverage value (0-1)
        """
        unique_recommended = set()
        for items in recommended_items:
            unique_recommended.update(items)
        
        coverage = len(unique_recommended) / total_items if total_items > 0 else 0.0
        return coverage
    
    @staticmethod
    def diversity(recommended: List[int], item_features: pd.DataFrame, 
                 similarity_metric: str = 'cosine') -> float:
        """
        Calculate diversity of recommendations.
        Measures average dissimilarity between recommended items.
        
        Args:
            recommended: List of recommended item IDs
            item_features: DataFrame with item features for similarity calculation
            similarity_metric: Type of similarity ('cosine', 'euclidean')
            
        Returns:
            Diversity value (0-1, higher = more diverse)
        """
        if len(recommended) < 2:
            return 0.0
        
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        # Get features for recommended items
        features = item_features.loc[recommended].values
        
        if similarity_metric == 'cosine':
            similarities = cosine_similarity(features)
        else:
            similarities = 1 / (1 + euclidean_distances(features))
        
        # Calculate average pairwise similarity
        n = len(recommended)
        mask = ~np.eye(n, dtype=bool)
        avg_similarity = similarities[mask].mean()
        
        # Diversity is inverse of similarity
        diversity = 1.0 - avg_similarity
        return max(0.0, min(1.0, diversity))


class EvaluationReport:
    """
    Generate comprehensive evaluation reports.
    """
    
    def __init__(self):
        self.metrics = RecommendationMetrics()
    
    def evaluate_rating_prediction(self, y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate rating prediction model.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            Dictionary with RMSE and MAE
        """
        report = {
            'rmse': self.metrics.rmse(y_true, y_pred),
            'mae': self.metrics.mae(y_true, y_pred)
        }
        logger.info(f"Rating prediction evaluation: {report}")
        return report
    
    def evaluate_ranking(self, recommendations: Dict[int, List[int]], 
                        relevant_items: Dict[int, List[int]], 
                        k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ranking/recommendation quality.
        
        Args:
            recommendations: Dict mapping user_id to recommended item IDs
            relevant_items: Dict mapping user_id to relevant item IDs
            k_values: List of k values to evaluate at
            
        Returns:
            Dictionary with metrics for each k value
        """
        results = {}
        
        for k in k_values:
            precisions, recalls, f1s, ndcgs, maps = [], [], [], [], []
            
            for user_id, recommended in recommendations.items():
                relevant = relevant_items.get(user_id, [])
                
                precisions.append(self.metrics.precision_at_k(recommended, relevant, k))
                recalls.append(self.metrics.recall_at_k(recommended, relevant, k))
                f1s.append(self.metrics.f1_at_k(recommended, relevant, k))
                ndcgs.append(self.metrics.ndcg_at_k(recommended, relevant, k))
                maps.append(self.metrics.map_at_k(recommended, relevant, k))
            
            results[f'@{k}'] = {
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'f1': np.mean(f1s),
                'ndcg': np.mean(ndcgs),
                'map': np.mean(maps)
            }
        
        logger.info(f"Ranking evaluation: {results}")
        return results
    
    def generate_report(self, y_true: np.ndarray = None, y_pred: np.ndarray = None,
                       recommendations: Dict[int, List[int]] = None,
                       relevant_items: Dict[int, List[int]] = None) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True ratings (optional)
            y_pred: Predicted ratings (optional)
            recommendations: Recommendations dict (optional)
            relevant_items: Ground truth items dict (optional)
            
        Returns:
            Complete evaluation report
        """
        report = {}
        
        if y_true is not None and y_pred is not None:
            report['rating_prediction'] = self.evaluate_rating_prediction(y_true, y_pred)
        
        if recommendations is not None and relevant_items is not None:
            report['ranking'] = self.evaluate_ranking(recommendations, relevant_items)
        
        return report
