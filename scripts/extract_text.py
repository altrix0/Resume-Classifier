import os
import json
from PyPDF2 import PdfReader
import multiprocessing
from functools import partial

# Paths
PDF_DIR = "data/raw"  # Directory containing labeled PDFs
OUTPUT_FILE = "data/prepared/dataset.json"  # File to save extracted text
LOG_FILE = "data/prepared/extraction_errors.log"  # Log file for errors

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text or None if extraction fails.
    """
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            return text.strip()  # Return stripped text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def extract_text_worker(pdf_path, label):
    """
    Worker function to process a PDF and return labeled data.

    Args:
        pdf_path (str): Path to the PDF file.
        label (str): Label for the file.

    Returns:
        dict: Labeled text data or None if extraction fails.
    """
    text = extract_text_from_pdf(pdf_path)
    if text:
        return {"label": label, "text": text}
    else:
        return None

def process_directory(pdf_dir):
    """
    Extract text from all PDFs in the directory.

    Args:
        pdf_dir (str): Directory containing subdirectories of PDFs.

    Returns:
        list: List of dictionaries with extracted data.
    """
    data = []
    errors = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Use multiple processes

    for root, _, files in os.walk(pdf_dir):
        label = os.path.basename(root)
        pdf_paths = [os.path.join(root, file) for file in files if file.endswith(".pdf")]

        # Process PDFs in parallel
        worker_func = partial(extract_text_worker, label=label)
        results = pool.map(worker_func, pdf_paths)

        for result, pdf_path in zip(results, pdf_paths):
            if result:
                data.append(result)
            else:
                errors.append(pdf_path)

    pool.close()
    pool.join()

    # Log errors
    if errors:
        with open(LOG_FILE, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(errors))
        print(f"Logged {len(errors)} failed extractions to {LOG_FILE}.")

    return data

def save_to_json(data, output_file):
    """
    Save extracted data to a JSON file.

    Args:
        data (list): Extracted text data.
        output_file (str): File path to save JSON.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Starting text extraction from PDFs...")
    dataset = process_directory(PDF_DIR)
    print(f"Successfully extracted text from {len(dataset)} documents.")

    print(f"Saving dataset to {OUTPUT_FILE}...")
    save_to_json(dataset, OUTPUT_FILE)
    print(f"Dataset saved at {OUTPUT_FILE}.")
