
import pandas as pd
import pickle
import string
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

df = pd.read_csv('spam.csv')

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

df['message'] = df['message'].apply(preprocess_text)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['message'])

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model Saved Successfully")