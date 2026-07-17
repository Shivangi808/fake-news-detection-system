import streamlit as st

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Home",
    layout="wide"
)

# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.markdown(
    """
    <h1 style='text-align:center;'>
        Fake News Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align:center; color:gray;'>
        Machine Learning Based News Classification using
        TF-IDF Vectorization and Logistic Regression
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("")

# ----------------------------------------------------
# Project Overview
# ----------------------------------------------------

st.header("Project Overview")

st.write(
    """
Fake news spreads rapidly through online platforms and social media.
This application uses Natural Language Processing (NLP) and Machine Learning
to classify whether a news article is **Real** or **Fake**.

The text is first preprocessed, transformed into numerical features using
TF-IDF Vectorization, and then classified using a Logistic Regression model.
"""
)

# ----------------------------------------------------
# Project Information
# ----------------------------------------------------

st.header("Project Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Dataset", "9,900+ Articles")

with col2:
    st.metric("Algorithm", "Logistic Regression")

col3, col4 = st.columns(2)

with col3:
    st.metric("Vectorizer", "TF-IDF")

with col4:
    st.metric("Model Accuracy", "99.49%")

# ----------------------------------------------------
# Workflow
# ----------------------------------------------------

st.header("Workflow")

st.code(
"""
News Article
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression
      │
      ▼
Prediction
(Real / Fake)
"""
)

# ----------------------------------------------------
# Features
# ----------------------------------------------------

st.header("Project Features")

left, right = st.columns(2)

with left:
    st.markdown("""
- Fake News Detection
- Text Preprocessing
- TF-IDF Vectorization
- Logistic Regression
""")

with right:
    st.markdown("""
- Confidence Score
- Probability Visualization
- Interactive Web Application
- Fast Prediction
""")

# ----------------------------------------------------
# Technology Stack
# ----------------------------------------------------

st.header("Technology Stack")

st.table(
{
    "Technology": [
        "Python",
        "Streamlit",
        "Pandas",
        "NumPy",
        "NLTK",
        "Scikit-Learn",
        "Matplotlib"
    ],

    "Purpose": [
        "Programming Language",
        "Web Application",
        "Data Manipulation",
        "Numerical Computing",
        "Text Preprocessing",
        "Machine Learning",
        "Visualization"
    ]
}
)

# ----------------------------------------------------
# Model Information
# ----------------------------------------------------

st.header("Machine Learning Model")

st.write(
"""
**Model Used:** Logistic Regression

**Feature Extraction:** TF-IDF Vectorization

**Classification Type:** Binary Classification

The model predicts whether a news article belongs to one of the following classes:

- Real News
- Fake News
"""
)

# ----------------------------------------------------
# Navigation
# ----------------------------------------------------

st.header("Try the Model")

st.write(
"""
Click the button below to open the prediction page.
"""
)

if st.button("Start Prediction", use_container_width=True):
    st.switch_page("pages/1_Predict.py")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.write("---")

st.caption(
    "Developed by Shivangi Singh | Fake News Detection System | IBM SkillsBuild Project"
)