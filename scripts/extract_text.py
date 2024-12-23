import os
import json
from PyPDF2 import PdfReader

# Paths
PDF_DIR = "data/raw"  # Directory containing labeled PDFs
OUTPUT_FILE = "data/prepared/dataset.json"  # File to save extracted text

def extract_text_from_pdfs(pdf_dir):
    """
    Extract text from PDF files stored in labeled subdirectories.
    
    Args:
        pdf_dir (str): Directory containing subdirectories of PDFs labeled by category.
    
    Returns:
        list: List of dictionaries containing "label" and "text" for each PDF.
    """
    data = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.endswith(".pdf"):
                label = os.path.basename(root)  # Use folder name as label
                pdf_path = os.path.join(root, file)
                try:
                    with open(pdf_path, "rb") as pdf_file:
                        reader = PdfReader(pdf_file)
                        text = "".join(page.extract_text() or "" for page in reader.pages)
                        
                        # Only include entries with non-empty text
                        if text.strip():
                            data.append({"label": label, "text": text})
                        else:
                            print(f"Warning: No text extracted from {pdf_path}")
                except Exception as e:
                    print(f"Error reading {pdf_path}: {e}")
    return data

def save_to_json(data, output_file):
    """
    Save the extracted data to a JSON file.
    
    Args:
        data (list): List of dictionaries containing extracted text and labels.
        output_file (str): File path to save the JSON data.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Starting text extraction from PDFs...")
    dataset = extract_text_from_pdfs(PDF_DIR)
    print(f"Successfully extracted text from {len(dataset)} documents.")
    
    print(f"Saving dataset to {OUTPUT_FILE}...")
    save_to_json(dataset, OUTPUT_FILE)
    print(f"Dataset saved at {OUTPUT_FILE}.")
