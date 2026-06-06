
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords

model = pickle.load(
    open('spam_model.pkl', 'rb')
)

vectorizer = pickle.load(
    open('vectorizer.pkl', 'rb')
)

def preprocess_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)

st.title("📧 Spam Email Detection")

email = st.text_area(
    "Enter Email Message"
)

if st.button("Predict"):

    processed_email = preprocess_text(email)

    vector = vectorizer.transform(
        [processed_email]
    )

    prediction = model.predict(vector)

    if prediction[0] == 'spam':
        st.error("🚫 Spam Email")
    else:
        st.success("✅ Not Spam")