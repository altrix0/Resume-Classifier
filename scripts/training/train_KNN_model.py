import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "split", "train_balanced.json")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "split", "test.json")
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "models", "knn_cv.pkl")

def load_data(file_path):
    """
    Load and preprocess data from a JSON file.

    Args:
        file_path (str): Path to the JSON dataset.

    Returns:
        tuple: A list of texts and their corresponding labels.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels

def train_knn_with_cv(X_train, y_train, k=5, cv_folds=5):
    """
    Train a k-Nearest Neighbors model with cross-validation.

    Args:
        X_train (sparse matrix): Training features.
        y_train (array): Encoded training labels.
        k (int): Number of neighbors for k-NN.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        KNeighborsClassifier: The trained k-NN model.
    """
    model = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine")

    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.2f}")

    # Train the model on the full training set
    print("Training the final k-NN model on the full training dataset...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model and print metrics.

    Args:
        model (KNeighborsClassifier): Trained k-NN model.
        X_test (sparse matrix): Test features.
        y_test (array): Encoded test labels.
        label_encoder (LabelEncoder): Label encoder for decoding labels.

    Returns:
        None
    """
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    print("Evaluation Metrics on Test Data:")
    print(classification_report(y_test_decoded, y_pred))
    print(f"Accuracy on Test Data: {accuracy_score(y_test_decoded, y_pred):.2f}")

def main():
    """
    Main function to train, evaluate, and save the k-NN model.
    """
    # Load data
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

    # Train the k-NN model with cross-validation
    print("Training k-NN model with cross-validation...")
    model = train_knn_with_cv(X_train, y_train_encoded, k=5, cv_folds=5)

    # Evaluate the model on test data
    print("Evaluating the model on test data...")
    evaluate_model(model, X_test, y_test_encoded, label_encoder)

    # Save the trained model, vectorizer, and label encoder
    print(f"Saving the k-NN model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(
            {"model": model, "vectorizer": vectorizer, "label_encoder": label_encoder},
            file,
        )
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
