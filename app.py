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

# --- CLASSY INJECTED CSS CUSTOMIZATIONS ---
st.markdown("""
<style>
    /* 1. Complete Web Body Clean Classy Aesthetic */
    .stApp {
        background: linear-gradient(180deg, #F4F6F9 0%, #EBF0F5 100%) !important;
        color: #1E293B !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* 2. Premium Container Framing */
    .stForm {
        background-color: #FFFFFF !important;
        padding: 35px !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 25px rgba(30, 41, 59, 0.05) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
    }

    /* 3. Dropdown Menu & Select Box Styling Overrides */
    div[data-baseweb="select"] {
        background-color: #F8FAFC !important;
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
        color: #1E293B !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    /* Interactive Highlight Focused Selection Tab State */
    div[data-baseweb="select"]:focus-within {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
    }
    
    div[data-baseweb="select"] * {
        color: #1E293B !important;
        font-weight: 500 !important;
    }

    /* 4. Number Score Fields Matching Style */
    div[data-baseweb="input"] {
        background-color: #F8FAFC !important;
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #4F46E5 !important;
    }

    /* 5. Clean Structured Font Hierarchy */
    h1 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.05em;
        text-align: center;
        margin-top: 10px;
    }
    .subtitle {
        color: #64748B !important;
        font-size: 1.15rem !important;
        text-align: center;
        margin-bottom: 40px !important;
    }
    
    .stMarkdown h3 {
        color: #1E293B !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* 6. Dynamic High-Contrast Button Design */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        width: 100% !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        border: none !important;
        padding: 14px 20px !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.3) !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- MAIN INTERFACE ---
st.markdown("<h1>🎓 Student Performance Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict specialized quantitative testing indicators through predictive analytical models.</p>", unsafe_allow_html=True)

# Main Structural Multi-column Grid
main_col, result_col = st.columns([2, 1], gap="large")

with main_col:
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("📋 Demographic & Academic Background")
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

                # Responsive color logic 
                if predicted_score_rounded < 50:
                    score_color = "#EF4444"
                elif predicted_score_rounded < 75:
                    score_color = "#F59E0B"
                else:
                    score_color = "#10B981"

                # High-fidelity dashboard gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_score_rounded,
                    title = {'text': "Predicted Math Performance", 'font': {'size': 18, 'color': '#0F172A', 'bold': True}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
                        'bar': {'color': score_color},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 1,
                        'bordercolor': "#E2E8F0",
                        'steps': [
                            {'range': [0, 50], 'color': '#FEE2E2'},
                            {'range': [50, 75], 'color': '#FEF3C7'},
                            {'range': [75, 100], 'color': '#D1FAE5'}
                        ]
                    }
                ))

                fig.update_layout(
                    paper_bgcolor = 'rgba(0,0,0,0)',
                    font = {'color': "#0F172A", 'family': "Inter"},
                    margin=dict(l=20, r=20, t=40, b=10)
                )

                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction Pipeline Error: {e}")
else:
    with result_col:
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.info("👈 System status ready. Toggle profile properties and evaluate output scoring values.")
