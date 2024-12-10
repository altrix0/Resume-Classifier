import os
import shutil
from datetime import datetime
from PyPDF2 import PdfReader
import re

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

# Extract text from PDFs
def extract_text_from_pdfs(pdf_paths):
    """Extract text from a list of PDF files."""
    extracted_data = []
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            processed_text = preprocess_text(text)
            extracted_data.append((os.path.basename(pdf_path), processed_text))
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    return extracted_data

# Preprocess text
def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# Organize resumes into folders
def organize_resumes(categorized_data, pdf_paths, label_encoder):
    """Organize resumes into category-specific folders."""
    # Decode the numeric categories to their original label names
    try:
        decoded_categories = label_encoder.inverse_transform(list(categorized_data.keys()))
    except ValueError as e:
        print(f"Error decoding categories: {e}")
        return None

    # Create output folder
    date_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(OUTPUT_DIR, date_folder)
    os.makedirs(output_folder, exist_ok=True)

    for label_idx, files in categorized_data.items():
        # Decode category name
        try:
            label = label_encoder.inverse_transform([label_idx])[0]
        except ValueError:
            print(f"Skipping invalid category: {label_idx}")
            continue

        label_folder = os.path.join(output_folder, label)
        os.makedirs(label_folder, exist_ok=True)
        for file in files:
            for pdf_path in pdf_paths:
                if os.path.basename(pdf_path) == file:
                    try:
                        shutil.copy(pdf_path, label_folder)
                    except Exception as e:
                        print(f"Error copying file {file} to {label_folder}: {e}")
                    break
    return output_folder
