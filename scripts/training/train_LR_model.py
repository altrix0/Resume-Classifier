import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "split", "train_balanced.json")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "split", "test.json")
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "models", "logistic_regression_cv.pkl")

def load_json(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        tuple: Lists of texts and labels.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    texts = [entry["text"] for entry in data]
    labels = [entry["label"] for entry in data]  # Using the updated 'label' key
    return texts, labels

def train_with_cross_validation(train_file, model_path, cv_folds=5):
    """
    Train Logistic Regression model with cross-validation and save the model.

    Args:
        train_file (str): Path to the training dataset.
        model_path (str): Path to save the trained model.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        None
    """
    # Load training data
    print("Loading training data...")
    texts, labels = load_json(train_file)

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Vectorize text data
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    features = vectorizer.fit_transform(texts)

    # Initialize Logistic Regression model
    print("Initializing Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=70, class_weight="balanced")

    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, features, labels_encoded, cv=cv_folds, scoring="accuracy")
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.2f}")

    # Train the model on the full training dataset
    print("Training the final model on the full training dataset...")
    model.fit(features, labels_encoded)

    # Save the model, vectorizer, and label encoder
    print(f"Saving the model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump({"model": model, "vectorizer": vectorizer, "label_encoder": label_encoder}, file)
    print("Model saved successfully!")

def evaluate_on_test_data(test_file, model_path):
    """
    Evaluate the trained model on test data.

    Args:
        test_file (str): Path to the test dataset.
        model_path (str): Path to the saved model.

    Returns:
        None
    """
    # Load test data
    print("Loading test data...")
    texts, labels = load_json(test_file)

    # Load the trained model
    print("Loading the trained model...")
    with open(model_path, "rb") as file:
        saved_data = pickle.load(file)
    model = saved_data["model"]
    vectorizer = saved_data["vectorizer"]
    label_encoder = saved_data["label_encoder"]

    # Encode labels
    print("Encoding test labels...")
    labels_encoded = label_encoder.transform(labels)

    # Transform test data
    print("Vectorizing test data...")
    features = vectorizer.transform(texts)

    # Evaluate the model
    print("Evaluating the model...")
    predictions_encoded = model.predict(features)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    print("Evaluation Metrics:")
    print(classification_report(labels, predictions))
    print(f"Accuracy: {accuracy_score(labels, predictions):.2f}")

if __name__ == "__main__":
    # Train and evaluate the Logistic Regression model
    print("Starting training process...")
    train_with_cross_validation(TRAIN_PATH, MODEL_PATH)

    print("\nEvaluating on test data...")
    evaluate_on_test_data(TEST_PATH, MODEL_PATH)
