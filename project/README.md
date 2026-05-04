# 🎬 Movie Recommendation System

A production-ready, end-to-end recommendation system using Python, FastAPI, and multiple machine learning algorithms. This project demonstrates best practices for building scalable, maintainable ML systems.

## ✨ Features

- **Multiple Recommendation Algorithms**
  - Content-based filtering
  - Collaborative filtering (SVD using Surprise)
  - Hybrid approach combining multiple methods
  - Cold-start handling with popular items fallback

- **Real Dataset Support**
  - MovieLens dataset integration (100K+ ratings)
  - Automatic dataset download and preprocessing
  - Genre-based content features
  - User-movie interaction matrices

- **Comprehensive Evaluation Metrics**
  - RMSE & MAE for rating prediction
  - Precision@K, Recall@K, F1@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MAP@K (Mean Average Precision)
  - Coverage & Diversity metrics

- **Rich Visualizations**
  - Rating distributions & sparsity analysis
  - Metrics heatmaps & comparison charts
  - User-item matrix visualization
  - Training convergence plots
  - Neural network training curves

- **Production Components**
  - FastAPI REST backend with full documentation
  - MongoDB integration with fallback in-memory DB
  - Streamlit interactive frontend
  - Docker & Docker Compose support
  - Comprehensive logging & error handling

- **Clean Architecture**
  - Modular, OOP design
  - Factory pattern for model creation
  - Separation of concerns
  - Easy to extend with new models/features

## 📁 Project Structure

```
project/
├── data/
│   ├── raw/                    # Raw dataset files
│   └── processed/              # Cleaned/processed data
├── models/
│   ├── train.py               # Training pipeline
│   ├── recommend.py           # Recommendation models
│   └── saved_models/          # Trained model artifacts
├── api/
│   ├── main.py               # FastAPI application
│   ├── routes.py             # API endpoints
│   └── schemas.py            # Pydantic models
├── db/
│   └── db.py                 # Database layer (MongoDB/In-Memory)
├── app/
│   └── streamlit_app.py      # Web UI
├── utils/
│   ├── preprocess.py         # Data preprocessing
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Charting utilities
├── config/
│   └── .env                  # Configuration
├── logs/                     # Application logs
├── requirements.txt          # Dependencies
├── Dockerfile                # Container definition
├── docker-compose.yml        # Multi-container setup
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- MongoDB (optional, falls back to in-memory DB)
- Docker & Docker Compose (for containerized deployment)

### Local Setup (Development)

1. **Clone and setup environment:**
```bash
cd project
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp config/.env .env
# Edit .env if needed (MongoDB URI, ports, etc.)
```

4. **Create necessary directories:**
```bash
mkdir -p data/raw data/processed models/saved_models logs
```

5. **Train models (choose one option):**

   **Option A: With MovieLens dataset (Recommended for production):**
   ```bash
   python models/train.py --movielens
   ```
   This will automatically download and use the MovieLens dataset with 100K+ real ratings.

   **Option B: With dummy data (for testing):**
   ```bash
   python models/train.py --dummy
   ```

   This will:
   - Generate dummy dataset (100 users, 50 movies, 1000 ratings)
- Train all three models
- Evaluate on sample data
- Save trained models to `models/saved_models/`

6. **Run API server:**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit API documentation: http://localhost:8000/docs

7. **Run Streamlit UI (in another terminal):**
```bash
streamlit run app/streamlit_app.py
```

Visit UI: http://localhost:8501

### Docker Deployment

1. **Start all services with Docker Compose:**
```bash
# Use either command depending on your Docker installation:
# docker-compose up -d
docker compose up -d
```

This starts:
- MongoDB at localhost:27017
- FastAPI at localhost:8000
- Streamlit at localhost:8501

2. **Check logs:**
```bash
# docker-compose logs -f api
docker compose logs -f api
```

3. **Stop services:**
```bash
# docker-compose down
docker compose down
```

## 📊 API Endpoints

### Health & Status
- `GET /api/v1/health` - Check API health
- `GET /api/v1/stats` - Get system statistics

### Users
- `POST /api/v1/users/create` - Create new user
- `GET /api/v1/users/{user_id}` - Get user profile

### Ratings
- `POST /api/v1/ratings` - Submit rating
- `GET /api/v1/users/{user_id}/ratings` - Get user's ratings
- `GET /api/v1/movies/{movie_id}/ratings` - Get movie ratings

### Recommendations
- `POST /api/v1/recommend` - Get recommendations
  ```json
  {
    "user_id": 1,
    "n_recommendations": 5,
    "model_name": "hybrid"
  }
  ```
- `GET /api/v1/users/{user_id}/recommendations/{model_name}` - Get stored recommendations

### Predictions
- `POST /api/v1/predict` - Predict rating for user-movie pair
- `GET /api/v1/models` - List available models

## 📚 Usage Examples

### Training with Custom Dataset

```python
from models.train import RecommendationTrainer

trainer = RecommendationTrainer(data_dir="data", models_dir="models/saved_models")
trainer.run_pipeline()
```

### Using Models Directly

```python
from models.recommend import RecommenderFactory
import pandas as pd

# Load your data
ratings_df = pd.read_csv('ratings.csv')

# Create model
model = RecommenderFactory.create_recommender('hybrid')

# Train
model.train(ratings_df)

# Get recommendations
recommendations = model.recommend(user_id=1, n_recommendations=5)
print(recommendations)  # [32, 45, 12, 78, 23]
```

### Evaluation

```python
from utils.metrics import EvaluationReport
import numpy as np

evaluator = EvaluationReport()

# Rating prediction metrics
report = evaluator.evaluate_rating_prediction(y_true, y_pred)
print(f"RMSE: {report['rmse']:.4f}")

# Ranking metrics
ranking_report = evaluator.evaluate_ranking(recommendations, ground_truth)
print(ranking_report)
```

### Visualization

```python
from utils.visualization import RecommendationVisualizer

viz = RecommendationVisualizer()

# Plot rating distribution
viz.plot_rating_distribution(ratings_array)

# Plot metrics heatmap
viz.plot_metrics_heatmap(metrics_data, save_path="metrics.png")
```

## 🔧 Configuration

Edit `.env` file to configure:

```bash
# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=recommendation_system

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model Training
MODEL_TYPE=svd
N_FACTORS=50
N_EPOCHS=100
LEARNING_RATE=0.005

# Recommendations
TOP_N_RECOMMENDATIONS=5
COLD_START_STRATEGY=popular

# Application
DEBUG=True
LOG_LEVEL=INFO
RANDOM_SEED=42
```

## 📈 Evaluation Metrics

### Rating Prediction
- **RMSE** (Root Mean Square Error): Measures average prediction error
- **MAE** (Mean Absolute Error): Average absolute deviation

### Ranking Quality
- **Precision@K**: Fraction of top-K recommendations that are relevant
- **Recall@K**: Fraction of relevant items in top-K
- **F1@K**: Harmonic mean of precision and recall
- **NDCG@K**: Ranking quality considering position
- **MAP@K**: Mean average precision

### Coverage & Diversity
- **Coverage**: Percentage of catalog recommended
- **Diversity**: Average dissimilarity between recommendations

## 🎯 Model Comparison

| Model | Pros | Cons |
|-------|------|------|
| Content-Based | Good for new items, interpretable | Limited novelty, requires feature engineering |
| Collaborative | Finds diverse recommendations | Requires user history, cold-start problem |
| Hybrid | Combines both approaches | More complex, higher computation |

## 🔐 Error Handling

The system gracefully handles:
- Missing or invalid data
- Database connection failures
- Model training errors
- Cold-start users (new users with no history)
- API request validation
- Malformed configurations

## 📝 Logging

Logs are written to:
- **File**: `logs/training.log` (training events)
- **File**: `logs/api.log` (API requests)
- **Console**: Real-time output

Configure log level in `.env`:
```bash
LOG_LEVEL=DEBUG  # Detailed information
LOG_LEVEL=INFO   # General information
LOG_LEVEL=WARNING # Potential issues
LOG_LEVEL=ERROR  # Error events
```

## 🧪 Testing

Basic model testing:

```bash
# Train models with dummy data
python models/train.py --dummy

# Test API locally
curl http://localhost:8000/api/v1/health

# Test recommendation
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n_recommendations": 5}'
```

## 🚀 Performance Optimization

- **Model Caching**: Trained models are pickled and loaded at startup
- **Database Indexes**: MongoDB indexes on frequently queried fields
- **Batch Processing**: Support for bulk operations
- **Sparse Matrices**: Efficient storage for user-item matrices
- **Async API**: Non-blocking FastAPI endpoints

## 📦 Dependencies

See `requirements.txt` for complete list:

```
pandas==2.0.3          # Data manipulation
numpy==1.24.3          # Numerical computing
scikit-learn==1.3.0    # ML algorithms
scikit-surprise==1.1.3 # Collaborative filtering
fastapi==0.104.1       # Web API framework
uvicorn==0.24.0        # ASGI server
pymongo==4.6.0         # MongoDB driver
streamlit==1.28.1      # Web UI framework
matplotlib==3.7.1      # Plotting
seaborn==0.12.2        # Statistical visualization
```

## 🎓 Learning Resources

- [Recommendation Systems Overview](https://towardsdatascience.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Surprise Library](https://surprise.readthedocs.io/)
- [MongoDB University](https://university.mongodb.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🐛 Troubleshooting

### MongoDB Connection Fails
```
→ System falls back to in-memory database
→ Check MONGO_URI in .env
→ Ensure MongoDB is running: docker run -p 27017:27017 mongo
```

### Models Not Training
```
→ Check data in data/raw/
→ Verify pandas/numpy installation
→ Check logs/training.log for errors
```

### Streamlit Not Showing Recommendations
```
→ Models may not be trained
→ Add ratings first using the API or UI
→ Check that models files exist in models/saved_models/
```

### Port Already in Use
```
→ API: Change API_PORT in .env
→ Streamlit: Change STREAMLIT_PORT in .env
→ Or kill process: lsof -ti:8000 | xargs kill -9
```

## 🔮 Future Enhancements

- [ ] Deep learning models (Neural Collaborative Filtering)
- [ ] Real-time recommendations with streaming data
- [ ] A/B testing framework
- [ ] Explainability module
- [ ] Context-aware recommendations
- [ ] Multi-armed bandits for exploration
- [ ] GraphQL API support
- [ ] Advanced caching strategies
- [ ] Distributed training with Ray
- [ ] Production-grade monitoring with Prometheus

## 📄 License

MIT License - Feel free to use, modify, and distribute.

## 👥 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📧 Contact

For questions or suggestions, open an issue on GitHub.

---

**Built with ❤️ using Python, FastAPI, and Machine Learning**
