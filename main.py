import joblib
from bs4 import BeautifulSoup


vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")


def preprocess(text):
    """Remove HTML tags and clean text."""
    return BeautifulSoup(text, "html.parser").get_text()


def new_prediction(review):
    preprocessed_text = preprocess(review)
    vectorized_review = vectorizer.transform([preprocessed_text])  
    prediction = model.predict(vectorized_review)  
    return "Positive" if prediction[0] == 1 else "Negative"


new_review = "Don’t worry this is not a sequel to A Dog’s Way Home, which came out in January, and I gave a marginal recommendation to. No, this is a sequel to the 2017 film A Dog’s Purpose, which I didn’t enjoy, but this manages to be even worse. This is a PG family film, and we get multiple dog deaths, human deaths, parental abuse, alcoholism, a toxic teenage relationship, car crashes and cancer. Good grief! The only thing I liked about the movie aside from the cute doggies was the relationship between Kathryn Prescott and Henry Lau. I’d watch them in another movie- maybe a light-hearted romantic comedy and not this massive downer…It makes Old Yeller look like a laugh-fest.."
print("Prediction:", new_prediction(new_review))
