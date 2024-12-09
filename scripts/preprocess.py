import json
import os
import re

def clean_text(text):
    """Cleans the input text by removing special characters, punctuation, and extra spaces."""
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    text = text.strip().lower()  # Lowercase and strip leading/trailing spaces
    return text

def preprocess_json(input_file, output_file):
    """Preprocess the dataset by cleaning resume text and saving the cleaned data."""
    with open(input_file, "r") as f:
        data = json.load(f)
    
    cleaned_data = []
    for entry in data:
        cleaned_data.append({
            "category": entry["category"],
            "text": clean_text(entry["text"])
        })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(cleaned_data, f, indent=4)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/prepared/dataset.json"
    output_file = "data/prepared/cleaned_dataset.json"
    preprocess_json(input_file, output_file)
