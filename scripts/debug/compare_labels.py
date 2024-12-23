import json
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "prepared", "dataset.json")
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "models", "ensemble_model.pkl")

# Function to load labels from dataset
def load_labels_from_dataset():
    try:
        with open(DATASET_PATH, "r") as file:
            data = json.load(file)
        labels = list({str(item["label"]) for item in data})  # Ensure labels are strings
        return labels
    except Exception as e:
        print(f"Error loading dataset labels: {e}")
        return []

# Function to load model and get the label encoder
def load_model_labels():
    try:
        with open(MODEL_PATH, "rb") as file:
            data = joblib.load(file)
        label_encoder = data["label_encoder"]
        model_labels = [str(label) for label in label_encoder.classes_]  # Convert np.str_ to strings
        return model_labels
    except Exception as e:
        print(f"Error loading model labels: {e}")
        return []

# Compare labels between dataset and model
def compare_labels():
    # Load labels from dataset and model
    dataset_labels = load_labels_from_dataset()
    model_labels = load_model_labels()

    # Display the labels
    print("\n=== Labels in Dataset ===")
    print(dataset_labels)
    
    print("\n=== Labels in Model ===")
    print(model_labels)

    # Compare labels
    missing_from_model = set(dataset_labels) - set(model_labels)
    missing_from_dataset = set(model_labels) - set(dataset_labels)

    # Print the comparison results
    print("\n=== Labels in Dataset but NOT in Model ===")
    if missing_from_model:
        print(missing_from_model)
    else:
        print("None")

    print("\n=== Labels in Model but NOT in Dataset ===")
    if missing_from_dataset:
        print(missing_from_dataset)
    else:
        print("None")

# Run the label comparison
if __name__ == "__main__":
    print("Starting label comparison...\n")
    compare_labels()
    print("\nLabel comparison complete.")
