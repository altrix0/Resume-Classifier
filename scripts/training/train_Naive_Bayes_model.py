import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "prepared", "train_balanced.json")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "prepared", "test.json")
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "models", "naive_bayes_cv.pkl")


def load_data(file_path):
    """Load and parse JSON data into texts and labels."""
    with open(file_path, "r") as file:
        data = json.load(file)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def train_naive_bayes_with_cv(X_train, y_train, cv_folds=5):
    """Train a Naive Bayes model with cross-validation."""
    model = MultinomialNB()

    # Cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Standard Deviation: {np.std(cv_scores):.2f}")

    # Train final model on full dataset
    print("Training model on full training dataset...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model on test data and print metrics."""
    predictions_encoded = model.predict(X_test)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    true_labels = label_encoder.inverse_transform(y_test)

    print("Evaluation Metrics:")
    print(classification_report(true_labels, predictions))
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.2f}")


def main():
    # Step 1: Load training and test data
    print("Loading data...")
    X_train_text, y_train = load_data(TRAIN_PATH)
    X_test_text, y_test = load_data(TEST_PATH)

    # Step 2: Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Step 3: Extract features with TF-IDF
    print("Extracting features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Step 4: Train model with cross-validation
    print("Training Naive Bayes model...")
    model = train_naive_bayes_with_cv(X_train, y_train_encoded)

    # Step 5: Evaluate model on test data
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test_encoded, label_encoder)

    # Step 6: Save the trained model
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as file:
        pickle.dump({
            "model": model,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder
        }, file)


if __name__ == "__main__":
    main()
