import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "split", "train_balanced.json")
TEST_PATH = os.path.join(BASE_DIR, "data", "split", "test.json")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "xgboost_cv.pkl")

def load_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    texts, labels = [], []
    for item in data:
        texts.append(item['text'])
        labels.append(item['label'])  # Changed 'category' to 'label'
    return texts, labels

def extract_features(X_train_text, X_test_text):
    """Convert text data into TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()
    return X_train, X_test, vectorizer

def train_model_with_cv(X_train, y_train, cv_folds=5):
    """Train the XGBoost model with cross-validation."""
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        objective='multi:softmax',
        num_class=len(set(y_train)),
        random_state=70
    )

    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.2f}")

    # Train the model on the full dataset
    print("Training final model on full training dataset...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model on the test data."""
    y_pred = model.predict(X_test)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    print("Evaluation Metrics:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    print(f"Accuracy: {accuracy_score(y_test_decoded, y_pred_decoded):.2f}")

def save_model(model, vectorizer, label_encoder, model_path):
    """Save the trained model, vectorizer, and label encoder to a file."""
    with open(model_path, 'wb') as file:
        pickle.dump({'model': model, 'vectorizer': vectorizer, 'label_encoder': label_encoder}, file)

def main():
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