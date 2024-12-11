import os
import shutil
from datetime import datetime
from PyPDF2 import PdfReader
import re

OUTPUT_DIR = "data/outputs"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def extract_text_from_pdfs(pdf_paths):
    extracted_data = []
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            processed_text = preprocess_text(text)
            extracted_data.append((os.path.basename(pdf_path), processed_text))
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    return extracted_data

def organize_resumes(categorized_data, pdf_paths, selected_categories, output_path, session_name):
    session_folder = os.path.join(output_path, session_name)
    os.makedirs(session_folder, exist_ok=True)

    pdf_path_mapping = {os.path.basename(path): path for path in pdf_paths}

    for category, files in categorized_data.items():
        if category not in selected_categories:
            continue

        label_folder = os.path.join(session_folder, category)
        os.makedirs(label_folder, exist_ok=True)

        for file in files:
            if file in pdf_path_mapping:
                shutil.copy(pdf_path_mapping[file], label_folder)
                print(f"Copied {file} to {label_folder}")

    return session_folder




