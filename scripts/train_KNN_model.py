import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'split', 'train_balanced.json')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'split', 'test.json')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'knn.pkl')

def load_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    texts, labels = [], []
    for item in data:
        texts.append(item['text'])
        labels.append(item['category'])
    return texts, labels

def train_knn(X_train, y_train, k=5):
    """Train a k-NN model."""
    model = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    print("Evaluation Metrics:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

def main():
    # Load data
    print("Loading data...")
    X_train_text, y_train = load_data(TRAIN_PATH)
    X_test_text, y_test = load_data(TEST_PATH)
    
    # Feature extraction
    print("Extracting features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    # Train the model
    print("Training k-NN model...")
    model = train_knn(X_train, y_train, k=5)
    
    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    
    # Save the model and vectorizer
    print(f"Saving the model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump({'model': model, 'vectorizer': vectorizer}, file)

if __name__ == "__main__":
    main()
