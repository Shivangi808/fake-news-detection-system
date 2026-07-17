import streamlit as st
import pickle
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="News Prediction",
    layout="wide"
)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# ----------------------------------------------------
# NLTK Setup
# ----------------------------------------------------

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ----------------------------------------------------
# Preprocessing Function
# ----------------------------------------------------

def preprocess(text):

    text = text.lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    words = text.split()

    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(stemmer.stem(word))

    return " ".join(filtered_words)

# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.markdown(
    """
    <h1 style='text-align:center;'>
        News Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align:center; color:gray;'>
        Enter a news article to check whether it is Real or Fake.
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("")

# ----------------------------------------------------
# Input
# ----------------------------------------------------

news = st.text_area(
    "News Article",
    height=250,
    placeholder="Paste the complete news article here..."
)

predict = st.button(
    "Predict",
    use_container_width=True
)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if predict:

    if news.strip() == "":

        st.warning("Please enter a news article.")

    else:

        clean_news = preprocess(news)

        vector = vectorizer.transform([clean_news])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]

        confidence = max(probabilities) * 100

        classes = model.classes_

        real_probability = 0
        fake_probability = 0

        for cls, prob in zip(classes, probabilities):

            if cls == "Real":
                real_probability = prob * 100

            else:
                fake_probability = prob * 100

        st.write("---")

        st.header("Prediction Result")

        if prediction == "Real":

            st.success("REAL NEWS")

        else:

            st.error("FAKE NEWS")

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Real News Probability",
                f"{real_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Fake News Probability",
                f"{fake_probability:.2f}%"
            )

        st.write("---")

        st.subheader("Prediction Probability")

        fig, ax = plt.subplots(figsize=(6,4))

        categories = ["Real", "Fake"]
        values = [real_probability, fake_probability]

        bars = ax.bar(categories, values, width=0.5)

        ax.set_ylim(0, 100)

        ax.set_ylabel("Probability (%)")

        ax.set_title("Prediction Probability")

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 1,
                f"{height:.1f}%",
                ha="center"
            )

        st.pyplot(fig)

        st.caption(
            "The chart displays the probability assigned by the model to each class."
        )

# ----------------------------------------------------
# Navigation
# ----------------------------------------------------

st.write("---")

if st.button(
    "Back to Home",
    use_container_width=True
):
    st.switch_page("app.py")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.write("---")

st.caption(
    "Developed by Shivangi Singh | Fake News Detection System | IBM SkillsBuild Project"
)