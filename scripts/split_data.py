import json
import os
import random

def split_json(input_file, train_file, test_file, split_ratio=0.8):
    """Split JSON dataset into train and test sets."""
    with open(input_file, "r") as f:
        data = json.load(f)
    
    train_data = []
    test_data = []
    
    for entry in data:
        label = entry['label']
        text = entry['text']
        if random.random() < split_ratio:
            train_data.append({"text": text, "category": label})
        else:
            test_data.append({"text": text, "category": label})
    
    # Save train and test datasets
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=4)
    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Train data saved to {train_file}")
    print(f"Test data saved to {test_file}")

if __name__ == "__main__":
    input_file = "data/prepared/dataset.json"  # Updated path to dataset.json
    train_file = "data/split/train.json"
    test_file = "data/split/test.json"
    split_json(input_file, train_file, test_file)
