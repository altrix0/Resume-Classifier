import json
import os
from sklearn.model_selection import train_test_split

def split_data(input_file, train_file, test_file, train_ratio=0.8):
    """Splits the cleaned dataset into training and testing datasets."""
    with open(input_file, "r") as f:
        data = json.load(f)

    texts = [entry["text"] for entry in data]
    labels = [entry["category"] for entry in data]

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, train_size=train_ratio, stratify=labels, random_state=42
    )

    # Prepare train and test datasets
    train_data = [{"text": text, "category": label} for text, label in zip(train_texts, train_labels)]
    test_data = [{"text": text, "category": label} for text, label in zip(test_texts, test_labels)]

    # Save datasets
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=4)
    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Train/Test split completed. Train size: {len(train_data)}, Test size: {len(test_data)}")

if __name__ == "__main__":
    input_file = "data/prepared/cleaned_dataset.json"
    train_file = "data/split/train.json"
    test_file = "data/split/test.json"
    split_data(input_file, train_file, test_file)
