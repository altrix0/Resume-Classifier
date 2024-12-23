import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "split", "train_balanced.json")
TEST_PATH = os.path.join(BASE_DIR, "data", "split", "test.json")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "decision_tree_cv.pkl")

def load_data(file_path):
    """
    Load text data and labels from a JSON file.
    
    Args:
        file_path (str): Path to the JSON dataset.
    
    Returns:
        tuple: List of texts and corresponding labels.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels

def train_decision_tree_with_cv(X_train, y_train, cv_folds=5):
    """
    Train a Decision Tree classifier with cross-validation.
    
    Args:
        X_train: Feature matrix for training.
        y_train: Encoded labels for training.
        cv_folds (int): Number of cross-validation folds.
    
    Returns:
        DecisionTreeClassifier: Trained model.
    """
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=30,
        random_state=70,
        class_weight="balanced"
    )

    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Std Deviation of CV Accuracy: {np.std(cv_scores):.2f}")

    # Train the model on the full training dataset
    print("Training final model on full dataset...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model on test data and print metrics.
    
    Args:
        model: Trained model.
        X_test: Feature matrix for testing.
        y_test: Encoded labels for testing.
        label_encoder: LabelEncoder instance.
    """
    # Predict on test data
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    # Print evaluation metrics
    print("\nEvaluation on Test Data:")
    print(classification_report(y_test_decoded, y_pred))
    print(f"Test Accuracy: {accuracy_score(y_test_decoded, y_pred):.2f}")

def main():
    """
    Main function to train and evaluate a Decision Tree model.
    """
    # Load training and testing data
    print("Loading data...")
    X_train_text, y_train = load_data(TRAIN_PATH)
    X_test_text, y_test = load_data(TEST_PATH)

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Extract features using TF-IDF
    print("Extracting features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Train the model with cross-validation
    print("Training Decision Tree with cross-validation...")
    model = train_decision_tree_with_cv(X_train, y_train_encoded, cv_folds=5)

    # Evaluate the model on test data
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test_encoded, label_encoder)

    # Save the model, vectorizer, and label encoder
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as file:
        pickle.dump({
            "model": model,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder
        }, file)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
