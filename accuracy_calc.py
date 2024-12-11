import os
import pickle
import json
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR,  "data", "prepared", "dataset.json")
ENSEMBLE_MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "ensemble_model.pkl")

def load_data(file_path):
    """Load and preprocess the dataset."""
    with open(file_path, "r") as file:
        data = json.load(file)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels

def evaluate_ensemble():
    print("Loading dataset...")
    texts, labels = load_data(DATASET_PATH)

    print("Loading ensemble model...")
    with open(ENSEMBLE_MODEL_PATH, "rb") as file:
        saved_model = pickle.load(file)

    ensemble = saved_model["model"]
    vectorizer = saved_model["vectorizer"]
    label_encoder = saved_model["label_encoder"]

    print("Vectorizing data...")
    features = vectorizer.transform(texts)
    labels_encoded = label_encoder.transform(labels)

    print("Making predictions...")
    predictions = ensemble.predict(features)

    print("Evaluation Metrics:")
    print(classification_report(label_encoder.inverse_transform(labels_encoded),
                                label_encoder.inverse_transform(predictions)))
    print(f"Accuracy: {accuracy_score(labels_encoded, predictions):.2f}")

if __name__ == "__main__":
    evaluate_ensemble()
