import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

def load_json(file_path):
    """Load JSON dataset and return texts and labels."""
    with open(file_path, "r") as f:
        data = json.load(f)
    texts = [entry["text"] for entry in data]
    labels = [entry["category"] for entry in data]
    return texts, labels

def train_and_evaluate(train_file, test_file, output_model_dir):
    """Train a Logistic Regression model and evaluate it on test data."""
    # Load datasets
    train_texts, train_labels = load_json(train_file)
    test_texts, test_labels = load_json(test_file)

    # Vectorize data
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Train the model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, train_labels)

    # Evaluate the model
    predictions = model.predict(X_test)
    print("Evaluation Metrics:")
    print(classification_report(test_labels, predictions))
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.2f}")

    # Save the trained model
    os.makedirs(output_model_dir, exist_ok=True)
    model_path = os.path.join(output_model_dir, 'logistic_regression_balanced.pkl')
    with open(model_path, "wb") as f:
        pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_file = "data/split/train_balanced.json"
    test_file = "data/split/test.json"
    output_model_dir = "data/models"
    train_and_evaluate(train_file, test_file, output_model_dir)
