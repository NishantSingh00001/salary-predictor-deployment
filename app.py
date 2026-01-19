import streamlit as st
import pickle
import numpy as np

# 1. Load the trained model from Project 2
try:
    model = pickle.load(open('salary_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please upload salary_model.pkl to the repository.")

# 2. Set up the Web Interface
st.set_page_config(page_title="AI Salary Predictor", page_icon="💰")
st.title("💰 AI Salary Predictor")
st.write("This app uses a Machine Learning model (Linear Regression) to predict salary based on inputs.")

st.divider()

# 3. Create Input Fields
col1, col2 = st.columns(2)

with col1:
    experience = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
    test_score = st.slider("Aptitude Test Score (1-10)", 1, 10, 7)

with col2:
    interview_score = st.slider("Interview Performance (1-10)", 1, 10, 8)

# 4. Make Prediction
if st.button("Calculate Estimated Salary"):
    # Reshape input for the model
    input_data = np.array([[experience, test_score, interview_score]])
    prediction = model.predict(input_data)
    
    # Display Result
    st.balloons()
    st.success(f"### Predicted Salary: ${prediction[0]:,.2f}")
    st.info("Note: This prediction has a ~73% confidence interval based on historical training data.")
