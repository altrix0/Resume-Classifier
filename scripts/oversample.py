import json
import os
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd

def oversample_data(train_file, output_file):
    """
    Apply oversampling to balance the training data.

    Args:
        train_file (str): Path to the input JSON file containing the training dataset.
        output_file (str): Path to save the balanced dataset as a JSON file.
    """
    # Load the training data
    with open(train_file, 'r') as f:
        data = json.load(f)
    
    # Convert data into a DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Validate input data
    if 'text' not in df.columns or 'category' not in df.columns:
        raise ValueError("Input data must contain 'text' and 'category' fields.")
    
    print(f"Class distribution before oversampling: {Counter(df['category'])}")
    
    # Prepare the data for oversampling
    X = df['text'].values.reshape(-1, 1)  # Features (texts as 2D array)
    y = df['category'].values  # Labels (categories)
    
    # Apply RandomOverSampler to balance the dataset
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Create a balanced dataset
    balanced_data = [
        {"text": text[0], "category": category}
        for text, category in zip(X_resampled, y_resampled)
    ]
    
    # Save the balanced dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(balanced_data, f, indent=4)
    
    print(f"Oversampled data saved to {output_file}")
    print(f"Class distribution after oversampling: {Counter(y_resampled)}")

if __name__ == "__main__":
    # Input and output file paths
    train_file = "data/split/train.json"
    output_file = "data/split/train_balanced.json"
    
    # Perform oversampling
    oversample_data(train_file, output_file)
