import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and get its sentiment prediction!")

user_input = st.text_area("Enter your review here:")

if st.button("Predict"):
    if user_input.strip():
        cleaned = preprocess(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"
        st.subheader(f"Prediction: {sentiment}")
    else:
        st.warning("Please enter a review to analyze.")

