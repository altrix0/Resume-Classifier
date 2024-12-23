import os
import shutil
from datetime import datetime
from PyPDF2 import PdfReader
import re

# Default output directory for sorted resumes
OUTPUT_DIR = "data/outputs"

def preprocess_text(text):
    """
    Preprocess text by cleaning and standardizing it.

    Args:
        text (str): Raw text to preprocess.

    Returns:
        str: Cleaned and standardized text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text.strip()

def extract_text_from_pdfs(pdf_paths):
    """
    Extract and preprocess text from a list of PDF files.

    Args:
        pdf_paths (list): List of paths to PDF files.

    Returns:
        list: List of tuples containing filenames and preprocessed text.
    """
    extracted_data = []
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            # Concatenate text from all pages in the PDF
            text = "".join(page.extract_text() or "" for page in reader.pages)
            processed_text = preprocess_text(text)
            extracted_data.append((os.path.basename(pdf_path), processed_text))
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    return extracted_data

def organize_resumes(categorized_data, pdf_paths, selected_categories, output_path, session_name):
    """
    Organize resumes into category-specific folders.

    Args:
        categorized_data (dict): Resumes categorized by their respective labels.
        pdf_paths (list): List of paths to original PDF files.
        selected_categories (list): List of selected categories for sorting.
        output_path (str): Base output directory for storing sorted resumes.
        session_name (str): Name of the session folder.

    Returns:
        str: Path to the session folder containing sorted resumes.
    """
    # Create a session-specific folder under the output path
    session_folder = os.path.join(output_path, session_name)
    os.makedirs(session_folder, exist_ok=True)

    # Map PDF filenames to their full paths
    pdf_path_mapping = {os.path.basename(path): path for path in pdf_paths}

    for category, files in categorized_data.items():
        if category not in selected_categories:
            continue  # Skip categories not selected by the user

        # Create a folder for the category
        label_folder = os.path.join(session_folder, category)
        os.makedirs(label_folder, exist_ok=True)

        for file in files:
            if file in pdf_path_mapping:
                shutil.copy(pdf_path_mapping[file], label_folder)
                print(f"Copied {file} to {label_folder}")

    return session_folder
