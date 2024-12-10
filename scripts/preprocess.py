import json
import os
import re
from collections import defaultdict

def clean_text(text):
    """Clean the input text by removing punctuation and lowering case."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def preprocess_list_json(input_file, output_file):
    """Read JSON as a list, preprocess text, and save in categorized format."""
    with open(input_file, "r") as f:
        data = json.load(f)
    
    categorized_data = defaultdict(list)
    for item in data:
        label = item.get("label")
        text = item.get("text")
        if label and text:
            categorized_data[label].append(clean_text(text))
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(categorized_data, f, indent=4)

if __name__ == "__main__":
    input_file = "data/prepared/dataset.json"
    output_file = "data/prepared/cleaned_dataset.json"
    preprocess_list_json(input_file, output_file)
