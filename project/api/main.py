
import logging
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import sys

# Load environment variables
load_dotenv()

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import router, set_db, set_models
from db.db import MongoDBHandler, InMemoryDatabase
from models.recommend import RecommenderFactory

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_models() -> dict:
    """
    Load trained models from disk.
    
    Returns:
        Dictionary of loaded models
    """
    models = {}
    models_dir = Path("models/saved_models")
    
    model_files = {
        'content_based': 'content_based_model.pkl',
        'collaborative': 'collaborative_model.pkl',
        'hybrid': 'hybrid_model.pkl'
    }
    
    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        
        if model_path.exists():
            try:
                model = RecommenderFactory.create_recommender(model_name)
                loaded_model = type(model).load_model(str(model_path))
                
                if loaded_model:
                    models[model_name] = loaded_model
                    logger.info(f"Loaded {model_name} model from {model_path}")
                else:
                    logger.warning(f"Failed to load {model_name} model")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {str(e)}")
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    if not models:
        logger.warning("No trained models found. Using untrained models.")
        models = {
            'content_based': RecommenderFactory.create_recommender('content_based'),
            'collaborative': RecommenderFactory.create_recommender('collaborative'),
            'hybrid': RecommenderFactory.create_recommender('hybrid')
        }
    
    return models


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI app
    """
    # Initialize database
    logger.info("Initializing database...")
    try:
        db = MongoDBHandler()
        if not db.is_connected():
            logger.warning("MongoDB not available. Using in-memory database.")
            db = InMemoryDatabase()
    except Exception as e:
        logger.warning(f"Database initialization failed: {str(e)}. Using in-memory database.")
        db = InMemoryDatabase()
    
    # Load models
    logger.info("Loading recommendation models...")
    models = load_models()
    
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Application startup")
        yield
        logger.info("Application shutdown")
        if hasattr(db, 'close'):
            db.close()
    
    # Initialize FastAPI app
    app = FastAPI(
        title=os.getenv('API_TITLE', 'Movie Recommendation System API'),
        description="Production-ready recommendation system with multiple algorithms",
        version=os.getenv('API_VERSION', '1.0.0'),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    set_db(db)
    set_models(models)
    
    # Include routes
    app.include_router(router, prefix="/api/v1", tags=["recommendations"])
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Movie Recommendation System API",
            "version": os.getenv('API_VERSION', '1.0.0'),
            "docs": "/docs",
            "redoc": "/redoc"
        }
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    
    logger.info(f"Starting API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
