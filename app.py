import subprocess
import sys

def install_requirements():
    try:
        # This runs the 'pip install' command from inside your script
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        print(f"Failed to install dependencies: {e}")

# Run the installation function
install_requirements()

import streamlit as st

# Importing Analyzer function from Csat_prediction
from csat_pipeline import analyze_feedback

st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Feedback Sentiment Analyzer")

st.write(
    "Enter customer feedback to identify sentiment, "
    "potential issues, and recommended actions."
)

feedback = st.text_area(
    "Customer feedback",
    placeholder="Enter a customer review here..."
)

if st.button("Analyze Feedback"):
    if not feedback.strip():
        st.warning("Please enter some feedback.")
    else:
        result = analyze_feedback(feedback)

        st.subheader("Analysis Results")

        st.metric("Sentiment", result["sentiment"])
        st.metric("Compound Score", result["sentiment_score"])

        st.write("**Detected issue:**", result["issue"])
        st.write("**Recommended action:**", result["recommendation"])

