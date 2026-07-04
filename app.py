import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set up the webpage title and layout
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction")
st.write("Enter the student details below to predict their performance score.")
st.divider()

# Create a clean input form using Streamlit widgets
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        
        ethnicity = st.selectbox(
            "Race/Ethnicity", 
            ["group A", "group B", "group C", "group D", "group E"]
        )
        
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
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
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        
        test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
        
        reading_score = st.number_input(
            "Reading Score", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0, 
            step=1.0
        )
        
        writing_score = st.number_input(
            "Writing Score", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0, 
            step=1.0
        )

    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict Performance")

# What happens when the user clicks the button
if submit_button:
    with st.spinner("Processing data and calculating prediction..."):
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
            
            # Display the result beautifully on screen
            st.success(f"🎯 **Predicted Score:** {results[0]:.2f}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
