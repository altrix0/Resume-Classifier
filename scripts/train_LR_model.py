import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import numpy as np

def load_json(file_path):
    """Load JSON dataset and return texts and labels."""
    with open(file_path, "r") as f:
        data = json.load(f)
    texts = [entry["text"] for entry in data]
    labels = [entry["category"] for entry in data]
    return texts, labels

def train_with_cross_validation(train_file, output_model_dir, cv_folds=5):
    """Train a Logistic Regression model with cross-validation and save the model."""
    # Load dataset
    train_texts, train_labels = load_json(train_file)
    
    # Split data into training and validation for cross-validation
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)

    # Initialize Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=70, class_weight='balanced')

    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, train_labels, cv=cv_folds, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.2f}")

    # Train final model on the full training data
    print("Training final model on full training dataset...")
    model.fit(X_train, train_labels)

    # Save the trained model and vectorizer
    os.makedirs(output_model_dir, exist_ok=True)
    model_path = os.path.join(output_model_dir, 'logistic_regression_cv.pkl')
    with open(model_path, "wb") as f:
        pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
    print(f"Model saved to {model_path}")

def evaluate_on_test_data(test_file, model_path):
    """Evaluate the trained model on test data."""
    # Load dataset
    test_texts, test_labels = load_json(test_file)

    # Load the saved model and vectorizer
    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)
    model = saved_data['model']
    vectorizer = saved_data['vectorizer']

    # Transform test data
    X_test = vectorizer.transform(test_texts)

    # Evaluate the model
    predictions = model.predict(X_test)
    print("Evaluation Metrics on Test Data:")
    print(classification_report(test_labels, predictions))
    print(f"Accuracy on Test Data: {accuracy_score(test_labels, predictions):.2f}")

if __name__ == "__main__":
    train_file = "data/split/train_balanced.json"
    test_file = "data/split/test.json"
    output_model_dir = "data/models"

    # Train model with cross-validation
    train_with_cross_validation(train_file, output_model_dir)

    # Evaluate model on test data
    model_path = os.path.join(output_model_dir, 'logistic_regression_cv.pkl')
    evaluate_on_test_data(test_file, model_path)
