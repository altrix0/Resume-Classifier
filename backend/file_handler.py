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
    extracted_data = []
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            extracted_data.append((os.path.basename(pdf_path), text.strip()))
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    return extracted_data

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text

# Organize resumes into folders
def organize_resumes(categorized_data, pdf_paths):
    # Create output folder
    date_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(OUTPUT_DIR, date_folder)
    os.makedirs(output_folder, exist_ok=True)

    for label, files in categorized_data.items():
        label_folder = os.path.join(output_folder, label)
        os.makedirs(label_folder, exist_ok=True)
        for file in files:
            for pdf_path in pdf_paths:
                if os.path.basename(pdf_path) == file:
                    shutil.copy(pdf_path, label_folder)
                    break
    return output_folder
