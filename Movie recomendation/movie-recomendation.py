# Movie Genre Prediction Model - Robust Version for VS Code
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data():
    """Robust CSV loading using title as plot if plot column is missing"""
    try:
        for encoding in ['utf-8', 'latin1', 'windows-1252']:
            try:
                df = pd.read_csv('movies.csv', encoding=encoding, on_bad_lines='warn')
                print(f"Successfully loaded with {encoding} encoding")
                print("Available columns:", df.columns.tolist())

                # Use 'title' as plot and 'genres' directly
                if 'title' in df.columns and 'genres' in df.columns:
                    df = df.rename(columns={'title': 'plot'})
                else:
                    raise ValueError("Required columns 'title' and 'genres' not found.")

                return df

            except UnicodeDecodeError:
                continue

        raise ValueError("Failed to read file with standard encodings")

    except Exception as e:
        print(f"\nERROR LOADING DATA: {str(e)}")
        if 'df' in locals():
            print("\nColumns found in file:", df.columns.tolist())
        print("\nPlease ensure:")
        print("1. The file is named 'movies.csv' and in the same folder as this script")
        print("2. It contains columns named 'title' and 'genres'")
        return None


def preprocess_data(df):
    """Clean and preprocess the text data"""
    # Convert all text to strings
    df['plot'] = df['plot'].astype(str)
    df['genres'] = df['genres'].astype(str)
    
    # Clean text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
    # Tokenization and lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    # Apply cleaning
    df['clean_plot'] = df['plot'].apply(clean_text).apply(preprocess_text)
    
    # Convert genres to lists
def preprocess_data(df):
    """Clean and preprocess the text data"""
    # Convert all text to strings
    df['plot'] = df['plot'].astype(str)
    df['genres'] = df['genres'].astype(str)
    
    # Clean text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
    # Tokenization and lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    # Apply cleaning
    df['clean_plot'] = df['plot'].apply(clean_text).apply(preprocess_text)
    
    # Convert genres to lists
    df['genres'] = df['genres'].apply(lambda x: [g.strip() for g in x.split(',')] if pd.notna(x) else [])
    
    return df


def train_model(df):
    """Train and evaluate the genre prediction model"""
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = tfidf.fit_transform(df['clean_plot'])
    
    # Multi-label binarization
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genres'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining model...")
    model = OneVsRestClassifier(LogisticRegression(max_iter=500))
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
    
    return model, tfidf, mlb

def predict_genres(model, tfidf, mlb, plot_text, threshold=0.3):
    """Predict genres for new plot text"""
    # Preprocessing functions
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    # Process input text
    cleaned_text = preprocess_text(clean_text(plot_text))
    text_vec = tfidf.transform([cleaned_text])
    
    # Get predictions
    probs = model.predict_proba(text_vec)
    genres = []
    for i, genre in enumerate(mlb.classes_):
        if probs[0][i] > threshold:
            genres.append((genre, round(probs[0][i], 2)))
    
    return sorted(genres, key=lambda x: x[1], reverse=True)

def main():
    print("Movie Genre Prediction Model")
    print("="*50)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    if df is None:
        return
    
    # Preprocess
    print("\nPreprocessing data...")
    df = preprocess_data(df)
    
    print("\nData sample:")
    print(df[['plot', 'genres']].head(2))
    
    # Train model
    model, tfidf, mlb = train_model(df)
    
    # Sample predictions
    samples = [
        "A group of astronauts travel through a wormhole in search of a new habitable planet",
        "A high school student gets bitten by a radioactive spider and gains superpowers",
        "Animated toys come to life when humans aren't around"
    ]
    
    print("\nSample Predictions:")
    for plot in samples:
        print(f"\nPlot: {plot}")
        predictions = predict_genres(model, tfidf, mlb, plot)
        for genre, prob in predictions:
            print(f"{genre}: {prob:.0%}")

if __name__ == "__main__":
    main()