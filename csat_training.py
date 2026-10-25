# importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Data
df0 = pd.read_csv('british_airways_reviews.csv')

# removing Unnamed column
df0 = df0.drop('Unnamed: 0', axis = 1)

# Text Cleaning
# assigning cleaned data to df
df = df0.copy()

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

# Calling the clean_text function and store in new created column of cleaned text
df['cleaned_reviews'] = df['reviews'].fillna("").apply(clean_text)

# Install vaderSentiment Library
get_ipython().system('pip install vaderSentiment')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Creating model SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Creating function for Predict sentiment based on vader score
def predict_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return pd.Series({
        "sentiment": sentiment,
        "sentiment_score": compound
    })

# Creating new columns for sentiment and compound score
df[["sentiment", "sentiment_score"]] = df["reviews"].apply(predict_sentiment)

# Visualizing distribution
df['sentiment'].value_counts().plot(kind='bar', title='Customer Sentiment Distribution')

# Identifying the issue behind Negative Reviews
#  Selected model: Latent Dirichlet Allocation (LDA)

# Importing required Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Filtering Negative reviews and store into new dataframe
negative_reviews = df.loc[df['sentiment']=='Negative', 'cleaned_reviews']
negative_reviews.head()

# Creating Vectorizer object along with proper parameters
vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=1000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(negative_reviews)

# Creating LDA model
lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42,
    learning_method="batch",
    max_iter=20
)

# Run the model with Converted vectors
lda.fit(X)

feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")

# Assigning each negative review to its dominant topic
topic_probabilities = lda.transform(X)
dominant_topics = topic_probabilities.argmax(axis=1)

df['topic_id'] = pd.NA

df.loc[df['sentiment']=='Negative','topic_id'] = dominant_topics

df_n = df[df['sentiment']=='Negative']

# Mapping topic_id with corresponding names
topic_names = {
    0:'Customer Service',
    1:'Baggage Handling',
    2:'Food Quality',
    3:'Seat Comfort',
    4:'Flight Delays'
}

# Building the Recommendation Engine
# Mapping topic names with Recommendations
recommendations = {
    'Customer Service': 'Provide targeted staff training and monitor complaint resolution.',
    'Food Quality': 'Review catering quality and collect meal-specific ratings.',
    'Baggage Handling': 'Audit baggage tracking and handover processes.',
    'Seat Comfort': 'Analyze seat-related complaints and prioritize cabin improvements.',
    'Flight Delays': 'Review delay causes and improve proactive passenger notifications.'
}

# Built a function for Recommendation 
def recommend_action(sentiment, topic_id):
    if sentiment != "Negative" or pd.isna(topic_id):
        return "No complaint-specific action triggered."

    issue = topic_names.get(int(topic_id), "Other Issue")

    action = recommendations.get(
        issue,
        "Review the feedback and investigate the root cause."
    )

    return action

# Applying these recommedations to main data set
df['recommendation'] = df.apply(
    lambda row: recommend_action(row['sentiment'], row['topic_id']),
    axis = 1
)

df.head()

# Combining the pipeline and save the models
import joblib

artifacts = {
    'vectorizer': vectorizer,
    'lda':lda,
    'topic_names':topic_names,
    'recommendations':recommendations
}

joblib.dump(artifacts, 'sentiment_artifacts.pkl')



