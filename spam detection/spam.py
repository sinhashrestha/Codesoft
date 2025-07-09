import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Step 1: Enhanced data loading with error handling
def load_data(filepath):
    """
    Load dataset from a CSV file with error handling
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
        # Keep only relevant columns and rename them
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
        # Convert labels to binary (0 for ham, 1 for spam)
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Basic data validation
        if df.isnull().values.any():
            df = df.dropna()
            print("Warning: Null values found and dropped.")
            
        print(f"Dataset loaded successfully with {len(df)} records.")
        print(f"Class distribution:\n{df['label'].value_counts(normalize=True)}")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Step 2: Enhanced preprocessing with options
def preprocess_data(df, test_size=0.2, random_state=42, vectorizer_type='tfidf'):
    """
    Preprocess data with options for different vectorizers
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    # Choose vectorizer
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    else:  # default to tfidf
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # Vectorize text data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

# Step 3: Enhanced model training with hyperparameter tuning
def train_model(X_train, y_train, model_type='logistic', tune_params=False):
    """
    Train a model with optional hyperparameter tuning
    """
    models = {
        'naive_bayes': MultinomialNB(),
        'svm': LinearSVC(),
        'logistic': LogisticRegression(max_iter=1000)
    }
    
    param_grids = {
        'naive_bayes': {'alpha': [0.1, 0.5, 1.0]},
        'svm': {'C': [0.1, 1, 10], 'penalty': ['l2']},
        'logistic': {'C': [0.1, 1, 10], 'penalty': ['l2']}
    }
    
    if model_type not in models:
        print(f"Invalid model type. Using logistic regression as default.")
        model_type = 'logistic'
    
    if tune_params:
        print(f"Performing hyperparameter tuning for {model_type}...")
        grid_search = GridSearchCV(
            models[model_type],
            param_grids[model_type],
            cv=5,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        return best_model
    else:
        print(f"Training {model_type} model with default parameters...")
        model = models[model_type]
        model.fit(X_train, y_train)
        return model

# Step 4: Enhanced evaluation with visualization
def evaluate_model(model, X_test, y_test, vectorizer=None):
    """
    Evaluate model performance with metrics and visualizations
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Print metrics
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if y_prob is not None:
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Feature importance for linear models
    if hasattr(model, 'coef_') and vectorizer is not None:
        print("\nTop 10 features for each class:")
        feature_names = vectorizer.get_feature_names_out()
        coef = model.coef_[0]
        top_spam = sorted(zip(feature_names, coef), key=lambda x: x[1], reverse=True)[:10]
        top_ham = sorted(zip(feature_names, coef), key=lambda x: x[1])[:10]
        
        print("\nTop spam indicators:")
        for feat, score in top_spam:
            print(f"{feat}: {score:.4f}")
            
        print("\nTop ham indicators:")
        for feat, score in top_ham:
            print(f"{feat}: {score:.4f}")

# Step 5: Save and load model functionality
def save_model(model, vectorizer, model_dir='models'):
    """
    Save model and vectorizer to disk
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'spam_classifier.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model and vectorizer saved to {model_dir}")

def load_model(model_dir='models'):
    """
    Load model and vectorizer from disk
    """
    model_path = os.path.join(model_dir, 'spam_classifier.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("Model and vectorizer loaded successfully")
    return model, vectorizer

# Step 6: Enhanced main function with more options
def main(data_path, model_type='logistic', tune_params=False, vectorizer_type='tfidf', save_model_flag=False):
    # Load data
    df = load_data(data_path)
    if df is None:
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(
        df, vectorizer_type=vectorizer_type
    )
    
    # Train model
    model = train_model(
        X_train, y_train, 
        model_type=model_type, 
        tune_params=tune_params
    )
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, vectorizer)
    
    # Save model if requested
    if save_model_flag:
        save_model(model, vectorizer)

if __name__ == "__main__":
    # Example usage with command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Spam Classification System')
    parser.add_argument('--data', type=str, default='spam.csv', help='Path to dataset')
    parser.add_argument('--model', type=str, default='logistic', 
                       choices=['logistic', 'naive_bayes', 'svm'], 
                       help='Model type to use')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--vectorizer', type=str, default='tfidf',
                       choices=['tfidf', 'count'], 
                       help='Text vectorization method')
    parser.add_argument('--save', action='store_true', help='Save trained model')
    
    args = parser.parse_args()
    
    main(
        data_path=args.data,
        model_type=args.model,
        tune_params=args.tune,
        vectorizer_type=args.vectorizer,
        save_model_flag=args.save
    )