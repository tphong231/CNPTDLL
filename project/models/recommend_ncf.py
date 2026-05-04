"""
Neural Collaborative Filtering (NCF) Model
Deep learning-based recommendation system using neural networks.
"""

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional
import os

logger = logging.getLogger(__name__)


class NeuralCollaborativeFiltering:
    """
    Neural Collaborative Filtering implementation using Multi-Layer Perceptron.
    Based on the paper: "Neural Collaborative Filtering" by He et al.
    """

    def __init__(self, n_users: int = None, n_items: int = None,
                 embedding_dim: int = 32, hidden_layers: List[int] = None,
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize NCF model.

        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of embedding vectors
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        logger.info(f"NCF initialized with {n_users} users, {n_items} items, embedding_dim={embedding_dim}")

    def _build_model(self) -> keras.Model:
        """Build the NCF neural network model."""
        # Input layers
        user_input = layers.Input(shape=(1,), name='user_input')
        item_input = layers.Input(shape=(1,), name='item_input')

        # Embedding layers
        user_embedding = layers.Embedding(
            input_dim=self.n_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)

        item_embedding = layers.Embedding(
            input_dim=self.n_items,
            output_dim=self.embedding_dim,
            name='item_embedding'
        )(item_input)

        # Flatten embeddings
        user_flat = layers.Flatten()(user_embedding)
        item_flat = layers.Flatten()(item_embedding)

        # Concatenate embeddings
        concat = layers.Concatenate()([user_flat, item_flat])

        # Hidden layers
        x = concat
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation='relu',
                           name=f'hidden_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.BatchNormalization()(x)

        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(x)

        # Build model
        model = keras.Model(inputs=[user_input, item_input], outputs=output)

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        return model

    def _prepare_data(self, ratings_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training by encoding users and items.

        Args:
            ratings_df: DataFrame with user_id, movie_id, rating columns

        Returns:
            Tuple of (encoded_data, labels)
        """
        # Encode user and item IDs
        self.user_encoder.fit(ratings_df['user_id'].unique())
        self.item_encoder.fit(ratings_df['movie_id'].unique())

        # Update n_users and n_items
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)

        # Encode data
        user_ids = self.user_encoder.transform(ratings_df['user_id'])
        item_ids = self.item_encoder.transform(ratings_df['movie_id'])

        # Convert ratings to binary labels (like/dislike)
        # Rating >= 4 is positive, < 4 is negative
        labels = (ratings_df['rating'] >= 4).astype(int).values

        # Create training data
        data = np.column_stack([user_ids, item_ids])

        logger.info(f"Data prepared: {len(data)} samples, {self.n_users} users, {self.n_items} items")

        return data, labels

    def train(self, ratings_df: pd.DataFrame, epochs: int = 20,
              batch_size: int = 256, validation_split: float = 0.2) -> None:
        """
        Train the NCF model.

        Args:
            ratings_df: Training data with user_id, movie_id, rating
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        logger.info("Training Neural Collaborative Filtering model...")

        # Prepare data
        data, labels = self._prepare_data(ratings_df)

        # Split data
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, labels, test_size=validation_split, random_state=42
        )

        # Build model
        self.model = self._build_model()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                restore_best_weights=True,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=3,
                mode='max'
            )
        ]

        # Train model
        history = self.model.fit(
            [train_data[:, 0], train_data[:, 1]], train_labels,
            validation_data=([val_data[:, 0], val_data[:, 1]], val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("NCF training completed!")
        logger.info(f"Final validation AUC: {history.history['val_auc'][-1]:.4f}")

    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """
        Predict ratings for user-item pairs.

        Args:
            user_id: User ID
            item_ids: List of item IDs

        Returns:
            Array of predicted scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        try:
            # Encode user and items
            user_encoded = self.user_encoder.transform([user_id])[0]
            items_encoded = self.item_encoder.transform(item_ids)

            # Make predictions
            predictions = self.model.predict(
                [np.full(len(items_encoded), user_encoded), items_encoded],
                verbose=0
            ).flatten()

            return predictions

        except ValueError as e:
            # Handle unknown users/items
            logger.warning(f"Unknown user/item in prediction: {str(e)}")
            return np.zeros(len(item_ids))

    def recommend(self, user_id: int, n_recommendations: int = 10,
                  exclude_rated: bool = True) -> List[int]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated items

        Returns:
            List of recommended item IDs
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Get all available items
        all_items = list(self.item_encoder.classes_)

        # Remove items already rated by user (if exclude_rated is True)
        if exclude_rated and hasattr(self, '_user_rated_items'):
            rated_items = self._user_rated_items.get(user_id, set())
            candidate_items = [item for item in all_items if item not in rated_items]
        else:
            candidate_items = all_items

        if not candidate_items:
            return []

        # Predict scores for all candidates
        predictions = self.predict(user_id, candidate_items)

        # Get top N recommendations
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        recommended_items = [candidate_items[i] for i in top_indices]

        return recommended_items

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")

        # Save model weights and encoders
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

        # Save encoders
        encoder_path = filepath.replace('.h5', '_encoders.pkl')
        import pickle
        with open(encoder_path, 'wb') as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'n_users': self.n_users,
                'n_items': self.n_items
            }, f)

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        # Load model
        self.model = keras.models.load_model(filepath)

        # Load encoders
        encoder_path = filepath.replace('.h5', '_encoders.pkl')
        import pickle
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
            self.user_encoder = encoders['user_encoder']
            self.item_encoder = encoders['item_encoder']
            self.n_users = encoders['n_users']
            self.n_items = encoders['n_items']

        logger.info(f"Model loaded from {filepath}")

    def get_stats(self) -> Dict:
        """Get model statistics."""
        return {
            'model_type': 'Neural Collaborative Filtering',
            'n_users': self.n_users,
            'n_items': self.n_items,
            'embedding_dim': self.embedding_dim,
            'hidden_layers': self.hidden_layers,
            'parameters': self.model.count_params() if self.model else 0
        }