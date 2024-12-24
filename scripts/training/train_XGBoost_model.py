import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "prepared", "train_balanced.json")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "prepared", "test.json")
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "models", "xgboost_cv.pkl")

def load_data(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        tuple: Lists of texts and labels.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]  # Updated for 'label' key
    return texts, labels

def extract_features(X_train_text, X_test_text):
    """
    Convert text data into TF-IDF features.
    
    Args:
        X_train_text (list): Training text data.
        X_test_text (list): Testing text data.
    
    Returns:
        tuple: Feature matrices for training and testing, and the TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()
    return X_train, X_test, vectorizer

def train_model_with_cv(X_train, y_train, cv_folds=5):
    """
    Train an XGBoost model with cross-validation.
    
    Args:
        X_train (array): Feature matrix for training.
        y_train (array): Labels for training.
        cv_folds (int): Number of cross-validation folds.
    
    Returns:
        XGBClassifier: Trained XGBoost model.
    """
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softmax",
        num_class=len(set(y_train)),
        random_state=70
    )

    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.2f}")

    # Train the model on the full dataset
    print("Training final model on full training dataset...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model on the test dataset.
    
    Args:
        model (XGBClassifier): Trained model.
        X_test (array): Feature matrix for testing.
        y_test (array): Encoded labels for testing.
        label_encoder (LabelEncoder): Encoder for decoding labels.
    """
    y_pred = model.predict(X_test)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    print("Evaluation Metrics:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    print(f"Accuracy: {accuracy_score(y_test_decoded, y_pred_decoded):.2f}")

def save_model(model, vectorizer, label_encoder, model_path):
    """
    Save the trained model, vectorizer, and label encoder to a file.
    
    Args:
        model (XGBClassifier): Trained model.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        label_encoder (LabelEncoder): Fitted label encoder.
        model_path (str): Path to save the model.
    """
    with open(model_path, "wb") as file:
        pickle.dump({
            "model": model,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder
        }, file)
    print(f"Model saved to {model_path}")

def main():
    """
    Main function to load data, train the model, evaluate it, and save the model.
    """
    print("Loading data...")
    X_train_text, y_train = load_data(TRAIN_PATH)
    X_test_text, y_test = load_data(TEST_PATH)

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("Extracting features...")
    X_train, X_test, vectorizer = extract_features(X_train_text, X_test_text)

    print("Training XGBoost model with cross-validation...")
    model = train_model_with_cv(X_train, y_train_encoded, cv_folds=5)

    print("Evaluating the model on test data...")
    evaluate_model(model, X_test, y_test_encoded, label_encoder)

    print(f"Saving the model to {MODEL_PATH}...")
    save_model(model, vectorizer, label_encoder, MODEL_PATH)

if __name__ == "__main__":
    main()
