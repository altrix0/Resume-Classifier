import json
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import os

def oversample_data(train_file, output_file):
    """Applies random oversampling to balance the training data."""
    with open(train_file, "r") as f:
        train_data = json.load(f)

    # Convert to DataFrame for processing
    df = pd.DataFrame(train_data)

    # Extract texts and labels
    X = df["text"].values.reshape(-1, 1)  # Reshape texts for oversampling
    y = df["category"]

    # Apply random oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Create oversampled dataset
    oversampled_data = [{"text": text[0], "category": category} for text, category in zip(X_resampled, y_resampled)]

    # Save oversampled data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(oversampled_data, f, indent=4)
    
    print("Oversampled data saved to:", output_file)
    print("Class distribution after oversampling:", Counter(y_resampled))

if __name__ == "__main__":
    train_file = "data/split/train.json"
    output_file = "data/split/train_balanced.json"
    oversample_data(train_file, output_file)
