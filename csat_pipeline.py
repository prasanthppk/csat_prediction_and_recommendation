import joblib

# importing SentimentIntensity Analyzer from VaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Loading artifacts from exported pkl files
artifacts = joblib.load("sentiment_artifacts.pkl")

# Importing supporting libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


for resourse in ['stopwords','wordnet','omw-1.4']:
    nltk.download(resourse, quiet=True)

stop_words = set(stopwords.words('english'))
stop_words -= {"no", "not", "nor", "never"}

# Creating a Model
lemmatizer = WordNetLemmatizer()

# Creating function for Tokenizing sentenses and Lemmatizing words
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    tokens = nltk.word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)

# Creating SentimentIntensityAnalyzer model
analyzer = SentimentIntensityAnalyzer()

# Defining function for Analyzing Feedback
def analyze_feedback(feedback):
    cleaned = clean_text(feedback)

    score = analyzer.polarity_scores(feedback)["compound"]

    if score >= 0.05:
        sentiment = "Positive"
    elif score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    issue = "Not applicable"
    action = "No complaint-specific action triggered."

    if sentiment == "Negative" and cleaned.strip():
        vector = artifacts["vectorizer"].transform([cleaned])
        topic_probs = artifacts["lda"].transform(vector)
        topic_id = int(topic_probs.argmax(axis=1)[0])

        issue = artifacts["topic_names"].get(
            topic_id, "Other Issue"
        )

        action = artifacts["recommendations"].get(
            issue, "Review the feedback and investigate the root cause."
        )

    return {
        "feedback": feedback,
        "sentiment": sentiment,
        "sentiment_score": round(score, 3),
        "issue": issue,
        "recommendation": action
    }




