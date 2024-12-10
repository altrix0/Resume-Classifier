import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'split', 'train_balanced.json')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'split', 'test.json')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'random_forest_cv.pkl')

def load_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    texts, labels = [], []
    for item in data:
        texts.append(item['text'])
        labels.append(item['category'])
    return texts, labels

def train_random_forest_with_cv(X_train, y_train, cv_folds=5):
    """Train a Random Forest model with cross-validation."""
    model = RandomForestClassifier(n_estimators=100, random_state=70)

    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.2f}")

    # Train the model on the full training set
    print("Training final model on full training dataset...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    print("Evaluation Metrics on Test Data:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy on Test Data: {accuracy_score(y_test, y_pred):.2f}")

def main():
    # Load data
    print("Loading data...")
    X_train_text, y_train = load_data(TRAIN_PATH)
    X_test_text, y_test = load_data(TEST_PATH)
    
    # Feature extraction
    print("Extracting features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    # Train the model with cross-validation
    print("Training Random Forest model with cross-validation...")
    model = train_random_forest_with_cv(X_train, y_train, cv_folds=5)
    
    # Evaluate the model on test data
    print("Evaluating the model on test data...")
    evaluate_model(model, X_test, y_test)
    
    # Save the model and vectorizer
    print(f"Saving the model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump({'model': model, 'vectorizer': vectorizer}, file)

if __name__ == "__main__":
    main()
