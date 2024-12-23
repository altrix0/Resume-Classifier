import json
import os
import random

def split_json(input_file, train_file, test_file, split_ratio=0.8):
    """
    Split a JSON dataset into train and test sets based on a specified ratio.

    Parameters:
        input_file (str): Path to the input JSON dataset file.
        train_file (str): Path to save the training dataset.
        test_file (str): Path to save the testing dataset.
        split_ratio (float): Ratio of training data (default is 0.8).
    """
    # Load the input JSON data
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Initialize train and test datasets
    train_data = []
    test_data = []

    # Split data into train and test sets
    for entry in data:
        label = entry.get('label')  # Ensure key existence
        text = entry.get('text')
        if label and text:  # Include only valid entries
            if random.random() < split_ratio:
                train_data.append({"text": text, "category": label})
            else:
                test_data.append({"text": text, "category": label})
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    # Save train and test datasets as JSON
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Train data saved to {train_file} ({len(train_data)} entries)")
    print(f"Test data saved to {test_file} ({len(test_data)} entries)")

if __name__ == "__main__":
    # Input and output file paths
    input_file = "data/prepared/dataset.json"  # Path to the original dataset
    train_file = "data/split/train.json"       # Path to save the training data
    test_file = "data/split/test.json"         # Path to save the testing data

    # Call the split function
    split_json(input_file, train_file, test_file)
