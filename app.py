import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AI Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CLASSY GOLD & BLACK THEME CSS ---
st.markdown("""
<style>
    /* 1. Warm Yellowish-Orangish Gold Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #F9D423 0%, #FF4E50 100%) !important;
        color: #000000 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* 2. Premium Translucent White Card Container */
    .stForm {
        background-color: rgba(255, 255, 255, 0.92) !important;
        padding: 35px !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
    }

    /* 3. Global Text Overrides to Rich Jet Black */
    h1, h2, h3, p, span, label, .subtitle {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    h1 {
        font-weight: 900 !important;
        letter-spacing: -0.05em;
        text-align: center;
        margin-top: 10px;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
    }
    
    .subtitle {
        font-size: 1.2rem !important;
        text-align: center;
        margin-bottom: 40px !important;
        font-weight: 500 !important;
    }

    /* 4. ROBUST OVERRIDES FOR DROPDOWN TABS AND SELECTION FIELDS */
    div[data-baseweb="select"], 
    div[data-baseweb="select"] > div, 
    div[role="button"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 10px !important;
    }
    
    div[data-baseweb="select"] {
        border: 2px solid #000000 !important;
    }
    
    div[data-baseweb="select"] * {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* 5. ROBUST OVERRIDES FOR NUMBER SCORE INPUT FIELDS */
    div[data-baseweb="input"], 
    div[data-baseweb="input"] > div, 
    div[data-baseweb="input"] input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 10px !important;
    }
    
    div[data-baseweb="input"] {
        border: 2px solid #000000 !important;
    }
    
    div[data-baseweb="input"] input {
        font-weight: 600 !important;
    }
    
    div[data-testid="stNumberInputStepDown"], 
    div[data-testid="stNumberInputStepUp"] {
        background-color: #E2E8F0 !important;
        color: #000000 !important;
        border-left: 1px solid #000000 !important;
    }

    /* 6. ENHANCED FORM SUBMIT BUTTON OVERRIDES (FIXES LOWER TAB COLOR) */
    div[data-testid="stFormSubmitButton"] {
        background-color: transparent !important;
        border: none !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: #000000 !important; /* Premium solid jet black button */
        color: #F9D423 !important; /* Premium gold text inside */
        border-radius: 10px !important;
        width: 100% !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        border: 2px solid #000000 !important;
        padding: 14px 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #FFFFFF !important; /* Inverts cleanly to white on hover state */
        color: #000000 !important;
        border: 2px solid #000000 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    div[data-testid="stNotification"] {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border: 2px solid #000000 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stNotification"] * {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- MAIN INTERFACE ---
st.markdown("<h1>🎓 Student Performance Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict specialized quantitative testing indicators through predictive analytical models.</p>", unsafe_allow_html=True)

# Main Grid Partitioning
main_col, result_col = st.columns([2, 1], gap="large")

with main_col:
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("<h3>📋 Demographic & Academic Background</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            gender = st.selectbox("Gender 👤", ["female", "male"])
            
            ethnicity = st.selectbox(
                "Race/Ethnicity 🌍", 
                ["group A", "group B", "group C", "group D", "group E"]
            )
            
            parental_level_of_education = st.selectbox(
                "Parental Education 🎓",
                [
                    "associate's degree",
                    "bachelor's degree",
                    "high school",
                    "master's degree",
                    "some college",
                    "some high school"
                ]
            )
        
        with col2:
            lunch = st.selectbox("Lunch Type 🍎", ["standard", "free/reduced"])
            
            test_preparation_course = st.selectbox("Test Prep Course 📝", ["none", "completed"])
            
            reading_score = st.number_input(
                "Reading Score 📚", 
                min_value=0.0, 
                max_value=100.0, 
                value=50.0, 
                step=1.0
            )
            
            writing_score = st.number_input(
                "Writing Score ✍️", 
                min_value=0.0, 
                max_value=100.0, 
                value=50.0, 
                step=1.0
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="Calculate Prediction Engine ✨")

# --- HANDLING SUBMISSION & OUTPUT GRAPHING ---
if submit_button:
    with result_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.spinner("Processing deep learning pipelines..."):
            try:
                data = CustomData(
                    gender=gender,
                    race_ethnicity=ethnicity,
                    parental_level_of_education=parental_level_of_education,
                    lunch=lunch,
                    test_preparation_course=test_preparation_course,
                    reading_score=float(reading_score),
                    writing_score=float(writing_score)
                )
                
                pred_df = data.get_data_as_data_frame()
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(pred_df)
                
                predicted_score_rounded = round(float(results[0]), 2)

                if predicted_score_rounded < 50:
                    score_color = "#EF4444"
                elif predicted_score_rounded < 75:
                    score_color = "#F59E0B"
                else:
                    score_color = "#10B981"

                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_score_rounded,
                    title = {'text': "Predicted Math Performance", 'font': {'size': 18, 'color': '#000000', 'bold': True}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#000000"},
                        'bar': {'color': '#000000'},
                        'bgcolor': "rgba(255,255,255,0.5)",
                        'borderwidth': 2,
                        'bordercolor': "#000000",
                        'steps': [
                            {'range': [0, 50], 'color': '#FEE2E2'},
                            {'range': [50, 75], 'color': '#FEF3C7'},
                            {'range': [75, 100], 'color': '#D1FAE5'}
                        ]
                    }
                ))

                fig.update_layout(
                    paper_bgcolor = 'rgba(0,0,0,0)',
                    font = {'color': "#000000", 'family': "Inter", 'bold': True},
                    margin=dict(l=20, r=20, t=40, b=10)
                )

                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction Pipeline Error: {e}")
else:
    with result_col:
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.info("👈 System status ready. Toggle profile properties and evaluate output scoring values.")
