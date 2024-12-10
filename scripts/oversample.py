import json
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd
import os

def oversample_data(train_file, output_file):
    """Applies oversampling to balance training data."""
    with open(train_file, 'r') as f:
        data = json.load(f)
    
    # Convert data into a DataFrame for processing
    df = pd.DataFrame(data)
    
    # Check if data format is valid
    if 'text' not in df.columns or 'category' not in df.columns:
        raise ValueError("Input data must contain 'text' and 'category' fields.")
    
    # Prepare for oversampling
    X = df['text'].values.reshape(-1, 1)  # Features
    y = df['category'].values  # Labels
    
    # Apply RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Recreate balanced dataset
    balanced_data = [{"text": text[0], "category": category} for text, category in zip(X_resampled, y_resampled)]
    
    # Save the balanced data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(balanced_data, f, indent=4)
    
    print(f"Oversampled data saved to {output_file}")
    print(f"Class distribution after oversampling: {Counter(y_resampled)}")

if __name__ == "__main__":
    train_file = "data/split/train.json"
    output_file = "data/split/train_balanced.json"
    oversample_data(train_file, output_file)
