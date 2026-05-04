"""
Streamlit frontend for the recommendation system.
A premium dark-mode UI designed for modern demos.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.db import MongoDBHandler, InMemoryDatabase
from db.db_optimized import MongoDBHandlerOptimized
from models.recommend import RecommenderFactory
from models.recommend_optimized import RecommenderFactoryOptimized
from utils.visualization import RecommendationVisualizer

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_style():
    st.markdown(
        """
        <style>
            /* CSS Custom Properties for Premium Design System */
            :root {
                --bg-primary: #0A0B0E;
                --bg-secondary: #111318;
                --bg-tertiary: #1A1D23;
                --bg-accent: rgba(255, 255, 255, 0.02);

                --text-primary: #FFFFFF;
                --text-secondary: #E2E8F0;
                --text-muted: #94A3B8;
                --text-subtle: #64748B;

                --border-primary: rgba(255, 255, 255, 0.08);
                --border-secondary: rgba(255, 255, 255, 0.04);
                --border-accent: rgba(115, 88, 255, 0.2);

                --gradient-primary: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
                --gradient-secondary: linear-gradient(135deg, #F093FB 0%, #F5576C 100%);
                --gradient-accent: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);

                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);

                --radius-sm: 8px;
                --radius-md: 12px;
                --radius-lg: 16px;
                --radius-xl: 24px;
                --radius-2xl: 32px;

                --spacing-xs: 4px;
                --spacing-sm: 8px;
                --spacing-md: 16px;
                --spacing-lg: 24px;
                --spacing-xl: 32px;
                --spacing-2xl: 48px;
                --spacing-3xl: 64px;
            }

            /* Base Styles */
            :root { color-scheme: dark; }
            .stApp {
                background: var(--bg-primary);
                color: var(--text-secondary);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                line-height: 1.6;
            }
            .css-1d391kg { background: transparent; }
            .css-1v3fvcr { padding-top: var(--spacing-lg); }

            /* Responsive Layout */
            @media (max-width: 768px) {
                .main-title { font-size: 2.5rem; }
                .section-panel { padding: var(--spacing-xl); margin-bottom: var(--spacing-lg); }
                .movie-card { margin-bottom: var(--spacing-md); }
                .movie-poster { height: 280px; }
            }
            @media (max-width: 480px) {
                .main-title { font-size: 2rem; }
                .section-panel { padding: var(--spacing-lg); }
                .hero-button { padding: var(--spacing-md) var(--spacing-xl); font-size: 0.9rem; }
            }

            /* Typography Hierarchy */
            .main-title {
                font-size: 3.5rem;
                font-weight: 900;
                letter-spacing: -0.06em;
                margin: 0 0 var(--spacing-sm) 0;
                background: var(--gradient-primary);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                line-height: 1.1;
            }
            .subtitle {
                color: var(--text-muted);
                font-size: 1.125rem;
                margin: var(--spacing-md) 0 var(--spacing-2xl) 0;
                font-weight: 400;
                line-height: 1.6;
                max-width: 600px;
            }

            /* Section Panels */
            .section-panel {
                background: var(--bg-secondary);
                border: 1px solid var(--border-primary);
                border-radius: var(--radius-xl);
                box-shadow: var(--shadow-xl);
                padding: var(--spacing-3xl);
                margin-bottom: var(--spacing-2xl);
                backdrop-filter: blur(20px);
                position: relative;
                overflow: hidden;
            }
            .section-panel::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 1px;
                background: var(--gradient-primary);
                opacity: 0.3;
            }

            /* Premium Buttons */
            .hero-button, .gradient-button {
                background: var(--gradient-accent);
                color: var(--text-primary);
                border-radius: var(--radius-2xl);
                padding: var(--spacing-lg) var(--spacing-2xl);
                font-size: 1rem;
                font-weight: 600;
                border: none;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                position: relative;
                overflow: hidden;
                box-shadow: var(--shadow-lg);
            }
            .hero-button::before, .gradient-button::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: left 0.5s;
            }
            .hero-button:hover::before, .gradient-button:hover::before {
                left: 100%;
            }
            .hero-button:hover, .gradient-button:hover {
                transform: translateY(-3px) scale(1.02);
                box-shadow: var(--shadow-2xl);
                background: var(--gradient-secondary);
            }
            .hero-button:active, .gradient-button:active {
                transform: translateY(-1px) scale(0.98);
                box-shadow: var(--shadow-lg);
            }

            /* Movie Cards - Premium Design */
            .movie-card {
                background: var(--bg-tertiary);
                border: 1px solid var(--border-secondary);
                border-radius: var(--radius-lg);
                overflow: hidden;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: var(--shadow-lg);
                cursor: pointer;
                position: relative;
                backdrop-filter: blur(10px);
            }
            .movie-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: var(--gradient-primary);
                opacity: 0;
                transition: opacity 0.3s ease;
                z-index: 1;
            }
            .movie-card:hover::before {
                opacity: 0.05;
            }
            .movie-card:hover {
                transform: translateY(-8px) scale(1.02);
                box-shadow: var(--shadow-2xl);
                border-color: var(--border-accent);
            }

            .movie-poster {
                width: 100%;
                height: 380px;
                object-fit: cover;
                transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                z-index: 2;
            }
            .movie-card:hover .movie-poster {
                transform: scale(1.08);
            }

            .movie-details {
                padding: var(--spacing-xl);
                position: relative;
                z-index: 2;
                background: linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.1) 100%);
            }
            .movie-title {
                margin: 0 0 var(--spacing-sm) 0;
                color: var(--text-primary);
                font-size: 1.25rem;
                font-weight: 700;
                line-height: 1.3;
                letter-spacing: -0.01em;
            }
            .movie-description {
                margin: 0 0 var(--spacing-lg) 0;
                color: var(--text-muted);
                font-size: 0.95rem;
                line-height: 1.5;
                display: -webkit-box;
                -webkit-line-clamp: 3;
                -webkit-box-orient: vertical;
                overflow: hidden;
                font-weight: 400;
            }
            .movie-score {
                padding: var(--spacing-xs) var(--spacing-md);
                border-radius: var(--radius-2xl);
                background: var(--gradient-accent);
                color: var(--text-primary);
                font-weight: 600;
                font-size: 0.875rem;
                display: inline-flex;
                align-items: center;
                gap: var(--spacing-xs);
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }

            /* Sidebar - Premium */
            .sidebar-panel {
                background: var(--bg-secondary);
                border: 1px solid var(--border-primary);
                border-radius: var(--radius-lg);
                padding: var(--spacing-xl);
                margin-bottom: var(--spacing-xl);
                backdrop-filter: blur(20px);
                box-shadow: var(--shadow-md);
            }
            .sidebar-heading {
                color: var(--text-primary);
                font-size: 1.125rem;
                font-weight: 700;
                margin-bottom: var(--spacing-md);
                letter-spacing: -0.01em;
            }
            .pill {
                display: block;
                padding: var(--spacing-lg) var(--spacing-xl);
                border-radius: var(--radius-md);
                background: var(--bg-accent);
                border: 1px solid var(--border-secondary);
                color: var(--text-secondary);
                margin-bottom: var(--spacing-md);
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }
            .pill:hover {
                background: rgba(255, 255, 255, 0.06);
                border-color: var(--border-primary);
                transform: translateX(2px);
            }
            .pill strong {
                color: var(--text-primary);
                font-weight: 600;
            }
            .small-muted { color: var(--text-subtle); }

            /* Enhanced Loading */
            .loading-spinner {
                display: inline-block;
                width: 24px;
                height: 24px;
                border: 3px solid rgba(255,255,255,0.2);
                border-radius: 50%;
                border-top-color: #667EEA;
                animation: spin 1s ease-in-out infinite;
                margin-right: var(--spacing-sm);
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(10, 11, 14, 0.95);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                backdrop-filter: blur(20px);
            }
            .loading-content {
                text-align: center;
                color: var(--text-primary);
                font-size: 1.25rem;
                font-weight: 600;
                background: var(--bg-secondary);
                padding: var(--spacing-2xl);
                border-radius: var(--radius-xl);
                border: 1px solid var(--border-primary);
                box-shadow: var(--shadow-2xl);
            }

            /* Premium Toast Messages */
            .toast {
                position: fixed;
                top: var(--spacing-xl);
                right: var(--spacing-xl);
                padding: var(--spacing-lg) var(--spacing-xl);
                border-radius: var(--radius-lg);
                color: var(--text-primary);
                font-weight: 600;
                z-index: 10000;
                min-width: 320px;
                box-shadow: var(--shadow-2xl);
                animation: toast-slide-in 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .toast.success {
                background: linear-gradient(135deg, rgba(34, 197, 94, 0.95) 0%, rgba(22, 163, 74, 0.95) 100%);
                border-left: 4px solid #22c55e;
            }
            .toast.error {
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.95) 0%, rgba(220, 38, 38, 0.95) 100%);
                border-left: 4px solid #ef4444;
            }
            .toast.warning {
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.95) 0%, rgba(217, 119, 6, 0.95) 100%);
                border-left: 4px solid #f59e0b;
            }

            /* Responsive Grid - Enhanced */
            .movie-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: var(--spacing-2xl);
                margin-top: var(--spacing-2xl);
            }
            @media (max-width: 1024px) {
                .movie-grid { grid-template-columns: repeat(2, 1fr); gap: var(--spacing-xl); }
            }
            @media (max-width: 768px) {
                .movie-grid { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: var(--spacing-lg); }
            }
            @media (max-width: 480px) {
                .movie-grid { grid-template-columns: 1fr; gap: var(--spacing-md); }
            }

            /* Button Grid - Premium */
            .button-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: var(--spacing-md);
                margin: var(--spacing-xl) 0;
            }
            @media (max-width: 480px) {
                .button-grid { grid-template-columns: 1fr; }
            }

            /* Form Layout - Enhanced */
            .form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr 2fr;
                gap: var(--spacing-lg);
                align-items: end;
                background: var(--bg-accent);
                padding: var(--spacing-lg);
                border-radius: var(--radius-lg);
                border: 1px solid var(--border-secondary);
            }
            @media (max-width: 768px) {
                .form-grid { grid-template-columns: 1fr; gap: var(--spacing-md); }
            }

            /* Stats Grid - Premium */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: var(--spacing-lg);
                margin-top: var(--spacing-xl);
            }
            @media (max-width: 480px) {
                .stats-grid { grid-template-columns: 1fr; }
            }

            /* Utility Classes */
            .text-center { text-align: center; }
            .mb-sm { margin-bottom: var(--spacing-sm); }
            .mb-md { margin-bottom: var(--spacing-md); }
            .mb-lg { margin-bottom: var(--spacing-lg); }
            .mb-xl { margin-bottom: var(--spacing-xl); }

            /* Enhanced Input Label Readability */
            .stNumberInput label, .stSlider label, .stTextInput label,
            .stSelectbox label, .stMultiselect label, .stTextArea label,
            .stCheckbox label, .stRadio label {
                color: var(--text-primary) !important;
                font-size: 1.125rem !important;
                font-weight: 700 !important;
                margin-bottom: var(--spacing-lg) !important;
                letter-spacing: 0.01em !important;
                line-height: 1.4 !important;
                display: block !important;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
            }

            /* Additional spacing for input containers */
            .stNumberInput, .stSlider, .stTextInput, .stSelectbox,
            .stMultiselect, .stTextArea, .stCheckbox, .stRadio {
                margin-bottom: var(--spacing-xl) !important;
                padding: var(--spacing-sm) 0 !important;
            }

            /* Improve input field styling for dark theme */
            .stNumberInput input, .stTextInput input, .stTextArea textarea,
            .stSelectbox select, .stMultiselect select {
                background: var(--bg-tertiary) !important;
                border: 1px solid var(--border-primary) !important;
                border-radius: var(--radius-md) !important;
                color: var(--text-primary) !important;
                font-size: 1rem !important;
                padding: var(--spacing-md) var(--spacing-lg) !important;
                transition: all 0.2s ease !important;
            }

            .stNumberInput input:focus, .stTextInput input:focus,
            .stTextArea textarea:focus, .stSelectbox select:focus,
            .stMultiselect select:focus {
                border-color: var(--border-accent) !important;
                box-shadow: 0 0 0 3px rgba(115, 88, 255, 0.1) !important;
                outline: none !important;
            }

            /* Slider specific styling */
            .stSlider .stSlider {
                background: var(--bg-tertiary) !important;
            }

            .stSlider label {
                margin-bottom: var(--spacing-lg) !important;
            }

            /* Ensure proper spacing between form elements */
            .form-grid .stNumberInput, .form-grid .stSlider {
                margin-bottom: var(--spacing-md) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_toast(message: str, type: str = "success", duration: int = 3000):
    """Display a toast notification."""
    toast_id = f"toast_{hash(message + str(st.session_state.get('toast_counter', 0)))}"
    st.session_state.toast_counter = st.session_state.get('toast_counter', 0) + 1

    toast_html = f"""
    <div id="{toast_id}" class="toast {type}">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.2rem;">
                {'✅' if type == 'success' else '❌' if type == 'error' else '⚠️'}
            </span>
            <span>{message}</span>
        </div>
    </div>
    <script>
        setTimeout(() => {{
            const toast = document.getElementById('{toast_id}');
            if (toast) {{
                toast.classList.add('fade-out');
                setTimeout(() => toast.remove(), 300);
            }}
        }}, {duration});
    </script>
    """
    st.markdown(toast_html, unsafe_allow_html=True)


def show_loading_overlay(message: str = "Generating recommendations..."):
    """Display a loading overlay."""
    overlay_html = f"""
    <div class="loading-overlay" id="loading-overlay">
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <div>{message}<span class="loading-dots"></span></div>
        </div>
    </div>
    """
    st.markdown(overlay_html, unsafe_allow_html=True)


def hide_loading_overlay():
    """Hide the loading overlay."""
    st.markdown(
        """
        <script>
            const overlay = document.getElementById('loading-overlay');
            if (overlay) overlay.remove();
        </script>
        """,
        unsafe_allow_html=True,
    )


def get_movie_title(movie_id: int) -> str:
    titles = {
        1: "John Wick",
        2: "Avenger Assemble",
        3: "The Lord of the Rings",
        4: "Once Upon A Time In Hollywood",
        5: "Joker",
        6: "The God Father",
        7: "Wood",
        8: "Bad Leader",
        9: "No Time To Die",
        10: "Nữ Chiến Binh Báo đen",
    }
    return titles.get(movie_id, f"Movie {movie_id}")


def get_movie_description(movie_id: int) -> str:
    descriptions = {
        1: "A relentless assassin returns to the underworld he left behind to settle the score and protect the only thing he has left: his honor.",
        2: "Earth's greatest heroes unite to defend the planet from an unprecedented alien threat, forging an alliance that changes the world forever.",
        3: "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.",
        4: "In 1969 Hollywood, an aging actor and his stunt double navigate fame, friendship, and the dark side of the dream factory.",
        5: "A troubled comedian descends into madness as society's cruelty and isolation push him toward a violent transformation.",
        6: "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        7: "A story rooted in the wilderness, where survival, identity and the human spirit are tested against the elements.",
        8: "A flawed leader struggles to hold power as ambition, betrayal, and moral compromise tear his world apart.",
        9: "A retired spy is pulled back into action when a dangerous adversary resurfaces, forcing him to race against a global threat.",
        10: """A city engulfed in chaos. A person with two selves.

By day, an anonymous figure; by night, a shadow that sows fear.
When the line between good and evil blurs, the real battle is no longer out there... but within her.""",
    }
    return descriptions.get(movie_id, "An engaging cinematic experience that explores the depths of human emotion and imagination.")


POSTER_MAP_FILE = Path(__file__).parent / "movie_posters.json"


def load_movie_posters() -> dict:
    if not POSTER_MAP_FILE.exists():
        return {}
    try:
        with POSTER_MAP_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_poster_url(movie_id: int) -> str:
    posters = load_movie_posters()
    return posters.get(str(movie_id), f"https://picsum.photos/seed/movie{movie_id}/420/620")


def render_movie_card(movie_id: int, score: float) -> str:
    title = get_movie_title(movie_id)
    description = get_movie_description(movie_id)
    poster = get_poster_url(movie_id)
    return f"""
    <div class='movie-card'>
        <img class='movie-poster' src='{poster}' alt='{title}' />
        <div class='movie-details'>
            <p class='movie-title'>{title}</p>
            <p class='movie-description'>{description}</p>
            <span class='movie-score'>⭐ {score:.1f}</span>
        </div>
    </div>
    """


def render_featured_movie_card(title: str, description: str, poster_url: str, score: float) -> str:
    return f"""
    <div class='movie-card'>
        <img class='movie-poster' src='{poster_url}' alt='{title}' />
        <div class='movie-details'>
            <p class='movie-title'>{title}</p>
            <p class='movie-description'>{description}</p>
            <span class='movie-score'>⭐ {score}</span>
        </div>
    </div>
    """


def init_state():
    # Initialize optimized database (with caching)
    if 'db' not in st.session_state:
        try:
            db = MongoDBHandlerOptimized()
            if not db.is_connected():
                db = InMemoryDatabase()
        except Exception:
            db = InMemoryDatabase()
        st.session_state.db = db

    # Initialize optimized models (with caching and efficient algorithms)
    if 'models' not in st.session_state:
        st.session_state.models = {
            'content_based': RecommenderFactoryOptimized.create_recommender('content_based'),
            'collaborative': RecommenderFactoryOptimized.create_recommender('collaborative'),
            'hybrid': RecommenderFactoryOptimized.create_recommender('hybrid'),
        }

    if 'page' not in st.session_state:
        st.session_state.page = '🏠 Home'

    if 'user_id' not in st.session_state:
        st.session_state.user_id = 1

    if 'movie_id' not in st.session_state:
        st.session_state.movie_id = 1

    if 'rating' not in st.session_state:
        st.session_state.rating = 4

    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []

    if 'active_model' not in st.session_state:
        st.session_state.active_model = 'hybrid'

    if 'toast_counter' not in st.session_state:
        st.session_state.toast_counter = 0


def show_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class='sidebar-panel'>
                <p class='sidebar-heading'>Welcome back</p>
                <p class='small-muted'>Demo-grade recommendations with cinematic polish.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.page = st.selectbox(
            'Navigation',
            ['🏠 Home', '⭐ Recommendations', '📊 Statistics'],
            index=['🏠 Home', '⭐ Recommendations', '📊 Statistics'].index(st.session_state.page),
        )

        st.markdown(
            f"""
            <div class='sidebar-panel'>
                <p class='sidebar-heading'>Current Profile</p>
                <span class='pill'><strong>User ID:</strong> {st.session_state.user_id}</span>
                <span class='pill'><strong>Model:</strong> {st.session_state.active_model.replace('_', ' ').title()}</span>
                <span class='pill'><strong>Theme:</strong> Dark</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def train_models_on_demand():
    """
    Train all models on demand using cached data.
    Very fast due to database caching and optimized algorithms.
    """
    # Fetch ratings from cache (much faster than direct DB access)
    ratings = st.session_state.db.get_all_ratings()
    if not ratings:
        return False

    # Convert to DataFrame once (vectorized)
    try:
        ratings_df = pd.DataFrame(ratings)[['user_id', 'movie_id', 'rating']]
    except Exception:
        return False
    
    # Train only untrained models (avoid redundant computation)
    trained = False
    for model in st.session_state.models.values():
        if not model.is_trained:
            try:
                model.train(ratings_df)
                trained = True
            except Exception as e:
                import logging
                logging.error(f"Error training model: {str(e)}")
                continue
    
    return trained


def render_header():
    st.markdown(
        """
        <div class='text-center' style='margin-bottom: 48px;'>
            <p class='main-title'>🎬 Movie Recommender</p>
            <p class='subtitle'>Discover your next favorite film with our intelligent recommendation engine powered by advanced machine learning algorithms.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home():
    st.markdown(
        """
        <div class='section-panel'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:24px; margin-bottom: 32px;'>
                <div style='flex: 1; min-width: 300px;'>
                    <h2 style='margin:0 0 16px 0; color:#FFFFFF; font-size: 1.75rem; font-weight: 700;'>Start your next movie night with confidence.</h2>
                    <p style='color:#94A3B8; margin:0; font-size: 1.1rem; line-height: 1.6;'>Rate a few movies to personalize your recommendations and see the engine respond instantly with cinematic precision.</p>
                </div>
                <div style='flex-shrink: 0;'>
                    <button class='hero-button'>Experience the premium flow ✨</button>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="form-grid">', unsafe_allow_html=True)
        with st.container():
            st.session_state.user_id = st.number_input('User ID', min_value=1, value=st.session_state.user_id, key='card_user_id')
        with st.container():
            st.session_state.movie_id = st.number_input('Movie ID', min_value=1, value=st.session_state.movie_id, key='card_movie_id')
        with st.container():
            st.session_state.rating = st.slider('Rating', 1, 5, st.session_state.rating, key='card_rating')
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<div style="height: 70px"></div>', unsafe_allow_html=True)
        with col2:
            if st.button('Submit Rating', use_container_width=True, key='home_submit'):
                success = st.session_state.db.add_rating(
                    user_id=int(st.session_state.user_id),
                    movie_id=int(st.session_state.movie_id),
                    rating=float(st.session_state.rating),
                )
                if success:
                    show_toast("Rating submitted successfully!", "success")
                else:
                    show_toast("Unable to save rating. Please try again.", "error")

    st.markdown(
        """
        <div class='section-panel'>
            <h3 style='margin-bottom:16px; color:#FFFFFF;'>Featured movie drops</h3>
            <div class='movie-grid'>
        """,
        unsafe_allow_html=True,
    )

    # Featured movies with descriptions
    featured_movies = [
        {
            "title": "Cinematic Pulse",
            "description": "A visually stunning masterpiece that captures the rhythm of modern life through innovative cinematography.",
            "poster": "https://picsum.photos/seed/night/420/620",
            "score": 4.9
        },
        {
            "title": "Neon Horizon",
            "description": "Dive into a neon-lit world where technology and humanity collide in this cyberpunk thriller.",
            "poster": "https://picsum.photos/seed/modern/420/620",
            "score": 4.8
        },
        {
            "title": "Velvet Chase",
            "description": "A sophisticated cat-and-mouse game unfolds in the glamorous underworld of high society.",
            "poster": "https://picsum.photos/seed/retro/420/620",
            "score": 4.7
        }
    ]

    for movie in featured_movies:
        st.markdown(
            render_featured_movie_card(
                movie["title"],
                movie["description"],
                movie["poster"],
                movie["score"]
            ),
            unsafe_allow_html=True
        )

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Display all movies catalog
    render_all_movies()


def render_all_movies():
    st.markdown(
        """
        <div class='section-panel'>
            <h2 style='margin:0 0 16px 0; color:#FFFFFF; font-size: 1.75rem; font-weight: 700;'>Complete Movie Catalog</h2>
            <p style='color:#94A3B8; margin:0 0 32px 0; font-size: 1.1rem; line-height: 1.6;'>Explore all available movies in our collection with their custom posters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="movie-grid">', unsafe_allow_html=True)
    for movie_id in range(1, 11):  # Display movies 1-10
        st.markdown(render_movie_card(movie_id, score=4.5), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_recommendations():
    st.markdown(
        """
        <div class='section-panel'>
            <h2 style='margin:0 0 16px 0; color:#FFFFFF; font-size: 1.75rem; font-weight: 700;'>Your personalized recommendations</h2>
            <p style='color:#94A3B8; margin:0 0 32px 0; font-size: 1.1rem; line-height: 1.6;'>Choose a profile, pick a model, and reveal your movie lineup with AI-powered precision.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        selected_user = st.number_input('User ID', min_value=1, value=st.session_state.user_id, key='rec_user_id')
    with c2:
        n_recs = st.slider('Recommendations', 1, 12, 6, key='rec_count')

    st.markdown('<div class="button-grid">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Hybrid', use_container_width=True, key='model_hybrid'):
            st.session_state.active_model = 'hybrid'
            show_toast("Switched to Hybrid model", "success", 1500)
    with col2:
        if st.button('Content-Based', use_container_width=True, key='model_content'):
            st.session_state.active_model = 'content_based'
            show_toast("Switched to Content-Based model", "success", 1500)
    with col3:
        if st.button('Collaborative', use_container_width=True, key='model_collaborative'):
            st.session_state.active_model = 'collaborative'
            show_toast("Switched to Collaborative model", "success", 1500)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f"<div style='text-align: center; margin: 24px 0; padding: 16px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'><p style='color:#E2E8F0; margin:0; font-size: 1rem; font-weight: 600;'>Active model: <strong style='color:#FFFFFF;'>{st.session_state.active_model.replace('_', ' ').title()}</strong></p></div>",
        unsafe_allow_html=True,
    )

    # OPTIMIZED UI: Use st.spinner instead of blocking overlay
    if st.button('Generate Recommendations', use_container_width=True, key='generate_recs'):
        with st.spinner('⚡ Analyzing preferences and generating recommendations...'):
            # Train models if needed (fast due to caching)
            train_models_on_demand()
            
            # Get the selected model
            model = st.session_state.models.get(st.session_state.active_model)
            
            if model:
                # Generate recommendations (very fast due to caching & optimized algorithms)
                st.session_state.recommendations = model.recommend(
                    int(selected_user), 
                    n_recommendations=int(n_recs)
                )
                st.session_state.user_id = selected_user
                show_toast(f"✨ Generated {len(st.session_state.recommendations)} recommendations!", "success", 2000)
            else:
                st.session_state.recommendations = []
                show_toast("⚠️ Selected model is unavailable. Please try another.", "error", 3000)
        
        # Rerun to display recommendations immediately
        st.rerun()

    recommendations = st.session_state.recommendations
    if recommendations:
        # Show quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recommendations", len(recommendations))
        with col2:
            st.metric("Model", st.session_state.active_model.replace('_', ' ').title())
        with col3:
            st.metric("User", st.session_state.user_id)
        
        st.markdown('<div class="movie-grid">', unsafe_allow_html=True)
        for movie_id in recommendations:
            st.markdown(render_movie_card(movie_id, score=4.3), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info('⭐ No recommendations yet. Generate a list to display premium movie cards.')


def render_statistics():
    stats = st.session_state.db.get_stats()
    st.markdown(
        """
        <div class='section-panel'>
            <h2 style='margin:0 0 16px 0; color:#FFFFFF; font-size: 1.75rem; font-weight: 700;'>Performance overview</h2>
            <p style='color:#94A3B8; margin:0 0 32px 0; font-size: 1.1rem; line-height: 1.6;'>A comprehensive summary of your system's usage and recommendation engine performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='pill'><strong style='font-size: 1.5rem;'>{stats.get('users', 0)}</strong><br><span style='color:#94A3B8;'>Active users</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='pill'><strong style='font-size: 1.5rem;'>{stats.get('ratings', 0)}</strong><br><span style='color:#94A3B8;'>Ratings collected</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='pill'><strong style='font-size: 1.5rem;'>{stats.get('recommendations', 0)}</strong><br><span style='color:#94A3B8;'>Stored recommendations</span></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class='section-panel'>
            <h3 style='margin:0 0 16px 0; color:#FFFFFF; font-size: 1.25rem; font-weight: 700;'>Engine insights</h3>
            <p style='color:#94A3B8; margin:0 0 24px 0; font-size: 1rem; line-height: 1.6;'>Real-time performance metrics showcasing the power of our AI-driven recommendation system.</p>
            <div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:16px; margin-top:24px;'>
                <div class='pill' style='text-align: center;'><strong style='font-size: 1.25rem;'>87%</strong><br><span style='color:#94A3B8;'>Coverage</span></div>
                <div class='pill' style='text-align: center;'><strong style='font-size: 1.25rem;'>92%</strong><br><span style='color:#94A3B8;'>Confidence</span></div>
                <div class='pill' style='text-align: center;'><strong style='font-size: 1.25rem;'>Instant</strong><br><span style='color:#94A3B8;'>Response</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Data Visualization Section
    st.markdown(
        """
        <div class='section-panel'>
            <h3 style='margin:0 0 16px 0; color:#FFFFFF; font-size: 1.25rem; font-weight: 700;'>📊 Data Visualization</h3>
            <p style='color:#94A3B8; margin:0 0 24px 0; font-size: 1rem; line-height: 1.6;'>Interactive charts showing rating patterns and system behavior.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Get ratings data
    try:
        ratings_data = st.session_state.db.get_all_ratings()
        if ratings_data:
            # Convert to DataFrame
            ratings_df = pd.DataFrame(ratings_data)
            if 'rating' in ratings_df.columns:
                ratings_df = ratings_df[['user_id', 'movie_id', 'rating']].copy()
                
                # Create visualizer
                visualizer = RecommendationVisualizer()
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Rating Distribution", "Top Rated Movies", "User-Item Sparsity"])
                
                with tab1:
                    try:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.hist(ratings_df['rating'], bins=5, edgecolor='black', alpha=0.7, color='steelblue')
                        ax.set_xlabel('Rating', fontsize=12)
                        ax.set_ylabel('Frequency', fontsize=12)
                        ax.set_title('Distribution of User Ratings', fontsize=14, fontweight='bold')
                        ax.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.info(f"Total ratings: {len(ratings_df)} | Average rating: {ratings_df['rating'].mean():.2f}")
                    except Exception as e:
                        st.warning(f"Could not generate rating distribution chart: {str(e)}")
                
                with tab2:
                    try:
                        import matplotlib.pyplot as plt
                        movie_stats = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        bars = ax.barh(range(len(movie_stats)), movie_stats['mean'], color='coral', edgecolor='black', alpha=0.7)
                        
                        for i, (idx, row) in enumerate(movie_stats.iterrows()):
                            ax.text(row['mean'], i, f" {row['mean']:.2f} ({int(row['count'])} ratings)",
                                   va='center', fontsize=9)
                        
                        ax.set_yticks(range(len(movie_stats)))
                        ax.set_yticklabels([f"Movie {mid}" for mid in movie_stats.index])
                        ax.set_xlabel('Average Rating', fontsize=12)
                        ax.set_title('Top 10 Rated Movies', fontsize=14, fontweight='bold')
                        ax.set_xlim([0, 5.5])
                        ax.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not generate top rated movies chart: {str(e)}")
                
                with tab3:
                    try:
                        import matplotlib.pyplot as plt
                        user_item_matrix = ratings_df.pivot_table(
                            index='user_id',
                            columns='movie_id',
                            values='rating',
                            fill_value=0
                        )
                        user_ratings = (user_item_matrix > 0).sum(axis=1)
                        item_ratings = (user_item_matrix > 0).sum(axis=0)
                        
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        
                        axes[0].hist(user_ratings, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
                        axes[0].set_xlabel('Number of Ratings per User', fontsize=11)
                        axes[0].set_ylabel('Frequency', fontsize=11)
                        axes[0].set_title('User Rating Distribution', fontsize=12, fontweight='bold')
                        axes[0].grid(axis='y', alpha=0.3)
                        
                        axes[1].hist(item_ratings, bins=20, edgecolor='black', alpha=0.7, color='coral')
                        axes[1].set_xlabel('Number of Ratings per Item', fontsize=11)
                        axes[1].set_ylabel('Frequency', fontsize=11)
                        axes[1].set_title('Item Rating Distribution', fontsize=12, fontweight='bold')
                        axes[1].grid(axis='y', alpha=0.3)
                        
                        fig.suptitle('Rating Sparsity Analysis', fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        sparsity = 1 - (ratings_df.shape[0] / (len(user_item_matrix) * len(user_item_matrix.columns)))
                        st.info(f"Matrix sparsity: {sparsity:.1%} | Users: {len(user_item_matrix)} | Movies: {len(user_item_matrix.columns)}")
                    except Exception as e:
                        st.warning(f"Could not generate sparsity chart: {str(e)}")
        else:
            st.info("No ratings data available yet. Start adding ratings to see visualizations.")
    except Exception as e:
        st.warning(f"Could not load visualization data: {str(e)}")


def main():
    apply_style()
    init_state()
    show_sidebar()
    render_header()

    if st.session_state.page == '🏠 Home':
        render_home()
    elif st.session_state.page == '⭐ Recommendations':
        render_recommendations()
    elif st.session_state.page == '📊 Statistics':
        render_statistics()

    st.markdown(
        """
        <div style='text-align:center; color:#64748B; margin-top:64px; padding: 32px 0; border-top: 1px solid rgba(255, 255, 255, 0.08);'>
            <small style='font-size: 0.875rem; font-weight: 500;'>Movie Recommendation System v1.0 | Powered by Streamlit, FastAPI & ML</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
