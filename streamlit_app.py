import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import requests
from bs4 import BeautifulSoup


# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
finbert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Starting the streamlit UI setup
st.title("Financial Sentiment Analysis")


# Fetch Yahoo Finance Headlines
def fetch_yahoo_finance_headlines():
    url = "https://finance.yahoo.com/"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = soup.find_all('h3', {'class': 'Mb(5px)'})  # This class name might change; we shall inspect the page
            return [headline.text for headline in headlines[:5]]  # Return the first 5 headlines
        else:
            st.error("Failed to retrieve data from Yahoo Finance.")
            return []
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# Preprocessing and sentiment analysis functions
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word != ""]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def sentiment_analysis(text: str):
    try:
        preprocessed_text = preprocess_text(text)
        predictions = finbert_pipeline(preprocessed_text)
        if predictions and isinstance(predictions[0], list) and isinstance(predictions[0][0], dict):
            best_prediction = max(predictions[0], key=lambda x: x['score'])
            return best_prediction['label'], best_prediction['score']
        else:
            raise ValueError("Unexpected prediction format")
    except Exception as e:
        st.error(f"An error occurred during sentiment analysis: {e}")
        return "Error", 0.0

# User input for manual text analysis
user_input = st.text_area("Enter Text for Analysis", "Type your text here...")

# Manual Analysis Button
if st.button('Analyze Text'):
    with st.spinner('Analyzing sentiment...'):
        label, score = sentiment_analysis(user_input)
        if label != "Error":
            st.success(f"Sentiment: {label} with a score of {score:.4f}")
        else:
            st.error("Failed to analyze sentiment. Please try again.")

# Automated Fetch and Analysis Button for Yahoo Finance Headlines
if st.button('Fetch and Analyze Yahoo Finance Headlines'):
    headlines = fetch_yahoo_finance_headlines()
    if headlines:
        for headline in headlines:
            st.write(headline)
            label, score = sentiment_analysis(headline)
            st.write(f"Sentiment: {label} with a score of {score:.4f}")