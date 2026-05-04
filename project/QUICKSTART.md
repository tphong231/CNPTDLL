# ⚡ Quick Start Guide

Get the recommendation system running in 5 minutes!

## Option 1: Docker (Recommended - 2 minutes)

### Prerequisites
- Docker installed with Compose support
- If `docker-compose` is unavailable, use `docker compose` instead

### Steps

```bash
# 1. Navigate to project directory
cd project

# 2. Start all services
# Use one of these based on your Docker installation:
# docker-compose up -d
docker compose up -d

# 3. Wait for services to be ready (30 seconds)

# 4. Access the applications:
#    - API Docs: http://localhost:8000/docs
#    - Web UI: http://localhost:8501
#    - MongoDB: localhost:27017 (admin/password)
```

Done! 🎉

### Useful Docker Commands

```bash
# View logs
# docker-compose logs -f api
docker compose logs -f api

# Stop services
# docker-compose down
docker compose down

# Remove all data
# docker-compose down -v
docker compose down -v

# Rebuild containers
# docker-compose build --no-cache
docker compose build --no-cache
```

---

## Option 2: Local Python Setup

### Prerequisites
- Python 3.10+
- MongoDB running locally (or skip for in-memory DB)

### Steps

```bash
# 1. Create virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create directories
mkdir -p data/raw data/processed models/saved_models logs

# 4. Configure environment (optional)
cp config/.env .env
# Edit .env if needed

# 5. Train models with dummy data
python models/train.py --dummy

# 6. Terminal 1 - Start API
python -m uvicorn api.main:app --reload

# 7. Terminal 2 - Start Web UI
streamlit run app/streamlit_app.py

# 8. Access applications:
#    - API Docs: http://localhost:8000/docs
#    - Web UI: http://localhost:8501
```

---

## First Steps

### 1. Create a User
```bash
curl -X POST http://localhost:8000/api/v1/users/create \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "username": "john_doe"}'
```

### 2. Submit Some Ratings
```bash
curl -X POST http://localhost:8000/api/v1/ratings \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "movie_id": 1,
    "rating": 5
  }'

# Repeat for more ratings...
```

### 3. Get Recommendations
```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "n_recommendations": 5,
    "model_name": "hybrid"
  }'
```

### 4. Or Use Web UI
Visit http://localhost:8501 and:
1. Go to "Manage Users" → Create a user
2. Go to "Manage Users" → Rate some movies
3. Go to "Get Recommendations" → Click buttons for different models

---

## Verify Installation

### Check API Health
```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "API is running",
  "database_connected": true
}
```

### View API Documentation
Visit: http://localhost:8000/docs

Interactive Swagger UI with all endpoints!

---

## Project Structure at a Glance

```
project/
├── 📁 data/              # Datasets
├── 📁 models/            # ML models & training
│   ├── train.py         # ← Run this to train
│   └── recommend.py     # Recommendation algorithms
├── 📁 api/              # FastAPI backend
│   └── main.py         # ← Run this to start API
├── 📁 app/              # Streamlit frontend
│   └── streamlit_app.py # ← Run this for UI
├── 📁 utils/            # Utilities
│   ├── preprocess.py   # Data preprocessing
│   ├── metrics.py      # Evaluation metrics
│   └── visualization.py # Charting
├── 📁 db/              # Database
│   └── db.py          # MongoDB operations
├── .env               # Configuration
├── requirements.txt   # Dependencies
├── Dockerfile         # Docker image
└── docker-compose.yml # Multi-container setup
```

---

## Configuration

Edit `.env` file to customize:

```bash
# MongoDB connection
MONGO_URI=mongodb://localhost:27017

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Model configuration
N_FACTORS=50        # SVD latent factors
N_EPOCHS=20         # Training epochs

# Recommendations
TOP_N_RECOMMENDATIONS=5

# Logging
LOG_LEVEL=INFO      # Options: DEBUG, INFO, WARNING, ERROR
```

---

## Common Commands

```bash
# Train models with custom data
python models/train.py --data-dir path/to/data

# Using dummy data (default)
python models/train.py --dummy

# Start API with auto-reload
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit with different settings
streamlit run app/streamlit_app.py --logger.level=debug

# View database stats
curl http://localhost:8000/api/v1/stats
```

---

## Troubleshooting

### "Port already in use"
```bash
# Find and kill process
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# On macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

### "Database connection refused"
```bash
# MongoDB is optional - the system uses in-memory DB as fallback
# To use MongoDB, start it:
docker run -d -p 27017:27017 mongo
```

### "No recommendations generated"
```bash
# Models need training data. Either:
# 1. Train with dummy data:
python models/train.py --dummy

# 2. Or add ratings via API, then models will train on them
```

---

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Read the docs**: Check [README.md](README.md)
3. **Add your data**: Place CSV files in `data/raw/`
4. **Train models**: Run `python models/train.py`
5. **Build on it**: Extend with your features!

---

## Need Help?

- 📖 Check [README.md](README.md) for detailed documentation
- 🔍 View API docs: http://localhost:8000/docs
- 📋 Check logs: `cat logs/training.log`
- 💬 Open an issue on GitHub

---

**Happy recommending! 🎬🍿**
