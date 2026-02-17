"""
California Housing Price Predictor - Main Application
This Streamlit application allows users to explore housing data, predict prices
using a trained model, view prediction history, and gain insights into the
model's performance.
"""

import os
import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Local imports
from database import HousingPriceDB
from predictor import HousingPredictor
from styles import configure_page, apply_custom_styles
from pages import (
    render_home_page,
    render_price_predictor_page,
    render_data_explorer_page,
    render_prediction_history_page,
    render_model_insights_page,
    render_about_page
)

# Load environment variables
load_dotenv()

# Configure page (must be first Streamlit command)
configure_page()

# Apply custom CSS
apply_custom_styles()

# Get the absolute path to the directory containing the current script
SCRIPT_DIR = Path(__file__).resolve().parent

# Navigate up one level to the project root
PROJECT_ROOT = SCRIPT_DIR.parent

# Define paths relative to the project root
MODEL_PATH = PROJECT_ROOT / 'models' / 'housing_price_model.pkl'
DATA_PATH = PROJECT_ROOT / 'data' / 'housing_cleaned.csv'

# For verification
print(f'Resolved MODEL_PATH: {MODEL_PATH}')
print(f'Resolved DATA_PATH: {DATA_PATH}')


@st.cache_resource
def load_model() -> HousingPredictor:
    """Load the trained housing price prediction model from disk"""
    return HousingPredictor(MODEL_PATH)


@st.cache_resource
def get_database() -> HousingPriceDB | None:
    """Initialize database connection"""
    try:
        db = HousingPriceDB(
            dbname=os.getenv('DB_NAME', 'housing_db'),
            user=os.getenv('DB_USER', 'housing_user'),
            # In production, use a more secure method to handle passwords
            password=os.getenv('DB_PASSWORD', 'password'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432))
        )
        return db

    except Exception as e:
        st.error(f'Database connection failed: {e}')
        return None


@st.cache_resource
def load_data() -> pd.DataFrame:
    """Load the cleaned housing dataset from CSV file"""
    return pd.read_csv(DATA_PATH)


# Load Resources
model: HousingPredictor = load_model()
db: HousingPriceDB | None = get_database()
data: pd.DataFrame = load_data()

# Map page names to functions and their required arguments
PAGES = {
    'Home': lambda: render_home_page(data),
    'Price Predictor': lambda: render_price_predictor_page(model, db, data),
    'Data Explorer': lambda: render_data_explorer_page(data),
    'Model Insights': lambda: render_model_insights_page(model),
    'Prediction History': lambda: render_prediction_history_page(db),
    'About': render_about_page,
}

# Sidebar Setup
st.sidebar.title('🏠 Navigation')
selection = st.sidebar.radio('Go to', list(PAGES.keys()))

# Database Status (Ternary Operator)
status_msg = "✅ Database Connected" if db else "⚠️ Database Offline"
render_status = st.sidebar.success if db else st.sidebar.warning
render_status(status_msg)

# Execute the selected page
PAGES[selection]()  # Call the function associated with the selected page
