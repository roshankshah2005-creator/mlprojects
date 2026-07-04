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

# --- CUSTOM CSS (Colors & Styling) ---
st.markdown("""
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #f8fafd;
        color: #2c3e50;
    }

    /* Titles and Headers */
    h1 {
        color: #1a508b;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #7f8c8d;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Form Container */
    .stForm {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e1e8ed;
    }

    /* Labels and Input Styling */
    .stSelectbox label, .stNumberInput label {
        color: #34495e;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Submit Button */
    div.stButton > button:first-child {
        background-color: #1a508b;
        color: white;
        border-radius: 25px;
        width: 100%;
        font-weight: bold;
        border: none;
        padding: 10px;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #0d3b66;
        color: white;
        border: none;
    }

    /* Prediction Result Area */
    .prediction-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #2ecc71;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }
    .prediction-label {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 5px;
    }
    .prediction-value {
        color: #2ecc71;
        font-size: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- MAIN INTERFACE ---
st.markdown("<h1>🎓 Student Performance AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Input student details below to generate an AI-powered math score prediction.</p>", unsafe_allow_html=True)

# Using columns for the main layout to give space to results
main_col, result_col = st.columns([2, 1], gap="large")

with main_col:
    # Use a form container for better visual grouping
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("📋 Student Profile & Background")
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
                step=1.0,
                help="Score from 0-100"
            )
            
            writing_score = st.number_input(
                "Writing Score ✍️", 
                min_value=0.0, 
                max_value=100.0, 
                value=50.0, 
                step=1.0,
                help="Score from 0-100"
            )

        # Space before button
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="Calculate Prediction ✨")

# --- HANDLING SUBMISSION ---
if submit_button:
    with result_col:
        # Custom loading animation (centered)
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.spinner("🧠 AI is processing data..."):
            try:
                # Map values into your existing CustomData pipeline structure
                data = CustomData(
                    gender=gender,
                    race_ethnicity=ethnicity,
                    parental_level_of_education=parental_level_of_education,
                    lunch=lunch,
                    test_preparation_course=test_preparation_course,
                    reading_score=float(reading_score),
                    writing_score=float(writing_score)
                )
                
                # Convert input data to the DataFrame format your model expects
                pred_df = data.get_data_as_data_frame()
                
                # Run the prediction pipeline
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(pred_df)
                
                # Clean up the result (handle single output or array)
                predicted_score = float(results[0])
                predicted_score_rounded = round(predicted_score, 2)

                # --- VISUAL PREDICTION GAUGE ---
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Define color based on score (Red to Green)
                if predicted_score_rounded < 50:
                    score_color = "#e74c3c" # Red
                elif predicted_score_rounded < 75:
                    score_color = "#f39c12" # Orange
                else:
                    score_color = "#2ecc71" # Green

                # Plotly Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_score_rounded,
                    title = {'text': "Predicted Math Score", 'font': {'size': 20, 'color': '#1a508b'}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#7f8c8d"},
                        'bar': {'color': score_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#e1e8ed",
                        'steps': [
                            {'range': [0, 50], 'color': '#fadbd8'},
                            {'range': [50, 75], 'color': '#fdebd0'},
                            {'range': [75, 100], 'color': '#d4efdf'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': predicted_score_rounded
                        }
                    }
                ))

                fig.update_layout(
                    paper_bgcolor = 'rgba(0,0,0,0)', # Transparent background
                    font = {'color': "#2c3e50", 'font': {'size': 16}},
                    margin=dict(l=20, r=20, t=30, b=10)
                )

                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {e}")
else:
    with result_col:
        # Information box shown before a prediction is run
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.info("👈 Fill out the profile and click the button to see the AI's predicted Math Score appear here.")
