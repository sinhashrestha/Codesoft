# Credit Card Fraud Detection System
# Complete end-to-end implementation


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, 
                            confusion_matrix, 
                            roc_auc_score,
                            precision_recall_curve,
                            average_precision_score)
from imblearn.under_sampling import RandomUnderSampler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. Data Preparation
# ======================

print("Loading and preparing data...")
# Load the dataset (replace with your actual file path)
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("\nError: File 'creditcard.csv' not found.")
    print("Please download the dataset from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
    print("and place it in the same directory as this script.")
    exit()

# Basic EDA
print("\nDataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nClass distribution:")
print(df['Class'].value_counts())
print(f"\nFraud percentage: {df['Class'].mean()*100:.2f}%")

# Visualize class imbalance
plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0=Legitimate, 1=Fraud)')
plt.show()

# Handle class imbalance
print("\nBalancing classes with undersampling...")
X = df.drop('Class', axis=1)
y = df['Class']

# Undersample majority class
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Scale the 'Amount' feature
scaler = StandardScaler()
X_res['Amount'] = scaler.fit_transform(X_res['Amount'].values.reshape(-1, 1))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42)

# ======================
# 2. Model Training
# ======================

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a model, returning metrics"""
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    # Plot PR curve
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
    return {
        'model': model,
        'name': model_name,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

# Initialize models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42))
]

# Train and evaluate all models
results = []
for name, model in models:
    result = train_and_evaluate_model(model, name, X_train, X_test, y_train, y_test)
    results.append(result)

# ======================
# 3. Model Comparison
# ======================

print("\nModel Comparison:")
comparison = pd.DataFrame({
    'Model': [r['name'] for r in results],
    'ROC AUC': [r['roc_auc'] for r in results],
    'Average Precision': [r['avg_precision'] for r in results]
}).sort_values('ROC AUC', ascending=False)

print(comparison)

# Select best model
best_result = max(results, key=lambda x: x['roc_auc'])
best_model = best_result['model']
print(f"\nBest model: {best_result['name']} with ROC AUC: {best_result['roc_auc']:.4f}")

# ======================
# 4. Feature Importance
# ======================

if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importances:")
    importances = best_model.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)[::-1]
    
    # Display top 10 features
    print("\nTop 10 Important Features:")
    for i in range(10):
        print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.title('Top 10 Feature Importances')
    plt.bar(range(10), importances[indices[:10]], align='center')
    plt.xticks(range(10), [features[i] for i in indices[:10]], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# ======================
# 5. Model Optimization
# ======================

if isinstance(best_model, RandomForestClassifier):
    print("\nPerforming hyperparameter tuning for Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("\nOptimized model performance:")
    print(classification_report(y_test, y_pred))

# ======================
# 6. Model Deployment
# ======================

print("\nSaving the best model...")
joblib.dump(best_model, 'credit_card_fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for new data

print("\nModel saved as 'credit_card_fraud_model.pkl'")
print("Scaler saved as 'scaler.pkl'")

# ======================
# 7. Example Prediction
# ======================

print("\nExample prediction with the trained model:")
sample = X_test.iloc[0:1].copy()
print("\nSample transaction features:")
print(sample)

# Load model (just to demonstrate)
loaded_model = joblib.load('credit_card_fraud_model.pkl')
prediction = loaded_model.predict(sample)
prob = loaded_model.predict_proba(sample)[:, 1]

print(f"\nPrediction: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")
print(f"Probability: {prob[0]*100:.2f}%")

print("\nFraud detection pipeline complete!")