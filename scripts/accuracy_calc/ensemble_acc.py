import os
import pickle
import json
from sklearn.metrics import classification_report, accuracy_score


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "..", "data", "prepared", "test.json")  # Adjusted for scripts folder
ENSEMBLE_MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "data", "models", "ensemble_model.pkl")


def load_data(file_path):
    """
    Load and preprocess the test dataset.

    Args:
        file_path (str): Path to the test dataset (JSON format).

    Returns:
        texts (list): List of text samples.
        labels (list): Corresponding labels.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def evaluate_ensemble():
    """
    Evaluate the ensemble model on the test dataset.
    Prints classification metrics and accuracy.
    """
    # Step 1: Load test dataset
    print("Loading test dataset...")
    texts, labels = load_data(DATASET_PATH)

    # Step 2: Load pre-trained ensemble model
    print("Loading pre-trained ensemble model...")
    with open(ENSEMBLE_MODEL_PATH, "rb") as file:
        saved_model = pickle.load(file)

    ensemble = saved_model["model"]
    vectorizer = saved_model["vectorizer"]
    label_encoder = saved_model["label_encoder"]

    # Step 3: Vectorize test data
    print("Vectorizing test data...")
    features = vectorizer.transform(texts)
    labels_encoded = label_encoder.transform(labels)

    # Step 4: Make predictions
    print("Making predictions...")
    predictions = ensemble.predict(features)

    # Step 5: Evaluate model performance
    print("\nEvaluation Metrics:")
    print(classification_report(
        label_encoder.inverse_transform(labels_encoded),
        label_encoder.inverse_transform(predictions),
        zero_division=0
    ))
    accuracy = accuracy_score(labels_encoded, predictions)
    print(f"Overall Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    evaluate_ensemble()
