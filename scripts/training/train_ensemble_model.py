import os
import pickle
import json
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "prepared", "dataset.json")
MODELS_DIR = os.path.join(BASE_DIR, "..", "data", "models")
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, "ensemble_model.pkl")


def load_data(file_path):
    """
    Load and preprocess the dataset from a JSON file.
    
    Args:
        file_path (str): Path to the dataset JSON file.

    Returns:
        tuple: Lists of texts and corresponding labels.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def load_models():
    """
    Load pre-trained models and their associated vectorizer.

    Returns:
        tuple: Dictionary of models and the shared vectorizer.
    """
    models = {}
    vectorizer = None

    model_names = [
        "logistic_regression_cv",
        "random_forest_cv",
        "naive_bayes_cv",
        "knn_cv",
        "svm_cv",
        "xgboost_cv",
        "decision_tree_cv",
    ]

    for model_name in model_names:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        try:
            with open(model_path, "rb") as file:
                data = pickle.load(file)
                models[model_name] = data["model"]
                if vectorizer is None:
                    vectorizer = data["vectorizer"]
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    if vectorizer is None:
        raise ValueError("Vectorizer not found among the loaded models.")
    return models, vectorizer


def create_ensemble(models):
    """
    Create a voting ensemble classifier using the loaded models.

    Args:
        models (dict): Dictionary of base models.

    Returns:
        VotingClassifier: Ensemble model with soft voting.
    """
    return VotingClassifier(
        estimators=[
            ("lr_cv", models["logistic_regression_cv"]),
            ("rf", models["random_forest_cv"]),
            ("nb", models["naive_bayes_cv"]),
            ("knn", models["knn_cv"]),
            ("svm", models["svm_cv"]),
            ("xgb", models["xgboost_cv"]),
            ("dt", models["decision_tree_cv"]),
        ],
        voting="soft"
    )


def main():
    """Main function to train and evaluate the ensemble model."""
    # Load data
    print("Loading dataset...")
    texts, labels = load_data(DATASET_PATH)

    # Vectorize data
    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    features = vectorizer.fit_transform(texts)

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data into train and test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded, test_size=0.2, random_state=70
    )

    # Load base models
    print("Loading pre-trained models...")
    models, model_vectorizer = load_models()

    # Create ensemble model
    print("Creating ensemble model...")
    ensemble = create_ensemble(models)

    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")

    # Train the ensemble model
    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating ensemble model...")
    y_pred = ensemble.predict(X_test)
    print("Evaluation Metrics:")
    print(classification_report(label_encoder.inverse_transform(y_test),
                                label_encoder.inverse_transform(y_pred)))
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save the ensemble model
    print(f"Saving the ensemble model to {ENSEMBLE_MODEL_PATH}...")
    with open(ENSEMBLE_MODEL_PATH, "wb") as file:
        pickle.dump({
            "model": ensemble,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder
        }, file)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
