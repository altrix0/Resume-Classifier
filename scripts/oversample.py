import json
import os
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd

# Resolve absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_FILE = os.path.join(BASE_DIR, "data", "prepared", "train.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "prepared", "train_balanced.json")

def oversample_data(train_file, output_file):
    """
    Apply oversampling to balance the training data.

    Args:
        train_file (str): Path to the input JSON file containing the training dataset.
        output_file (str): Path to save the balanced dataset as a JSON file.
    """
    # Load the training data
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")

    with open(train_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert data into a DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Validate input data
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input data must contain 'text' and 'label' fields.")
    
    print(f"Class distribution before oversampling: {Counter(df['label'])}")
    
    # Prepare the data for oversampling
    X = df['text'].values.reshape(-1, 1)  # Features (texts as 2D array)
    y = df['label'].values  # Labels (categories)
    
    # Apply RandomOverSampler to balance the dataset
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Create a balanced dataset
    balanced_data = [
        {"text": text[0], "label": label}
        for text, label in zip(X_resampled, y_resampled)
    ]
    
    # Save the balanced dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=4)
    
    print(f"Oversampled data saved to {output_file}")
    print(f"Class distribution after oversampling: {Counter(y_resampled)}")

if __name__ == "__main__":
    # Perform oversampling
    oversample_data(TRAIN_FILE, OUTPUT_FILE)
