import os
import json
import random
from shutil import copy2
from PyPDF2 import PdfReader
import multiprocessing
from functools import partial

# Paths
PDF_DIR = "data/raw"
TRAIN_DIR = "data/prepared/train"
TEST_DIR = "data/prepared/test"
TRAIN_JSON = "data/prepared/train.json"
TEST_JSON = "data/prepared/test.json"
LOG_FILE = "data/prepared/extraction_errors.log"
SPLIT_RATIO = 0.8

def split_pdfs(pdf_dir, train_dir, test_dir, split_ratio=0.8):
    """
    Split PDFs into train and test directories.

    Args:
        pdf_dir (str): Directory containing labeled PDFs.
        train_dir (str): Directory to save training PDFs.
        test_dir (str): Directory to save testing PDFs.
        split_ratio (float): Ratio of training data.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for root, _, files in os.walk(pdf_dir):
        label = os.path.basename(root)
        train_label_dir = os.path.join(train_dir, label)
        test_label_dir = os.path.join(test_dir, label)

        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

        pdf_files = [file for file in files if file.endswith(".pdf")]
        random.shuffle(pdf_files)

        split_point = int(len(pdf_files) * split_ratio)
        train_files = pdf_files[:split_point]
        test_files = pdf_files[split_point:]

        for file in train_files:
            copy2(os.path.join(root, file), os.path.join(train_label_dir, file))
        for file in test_files:
            copy2(os.path.join(root, file), os.path.join(test_label_dir, file))

    print(f"PDFs split into train and test directories: {train_dir}, {test_dir}")

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
            return text.strip()
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

def process_directory(pdf_dir, log_file):
    """
    Extract text from all PDFs in a directory.

    Args:
        pdf_dir (str): Directory containing labeled PDFs.
        log_file (str): Path to log file for errors.

    Returns:
        list: Extracted text data.
    """
    data = []
    errors = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for root, _, files in os.walk(pdf_dir):
        label = os.path.basename(root)
        pdf_paths = [os.path.join(root, file) for file in files if file.endswith(".pdf")]

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
        with open(log_file, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(errors))
        print(f"Logged {len(errors)} failed extractions to {log_file}.")

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
    # Step 1: Split PDFs
    split_pdfs(PDF_DIR, TRAIN_DIR, TEST_DIR, SPLIT_RATIO)

    # Step 2: Extract text from train and test directories
    print("Extracting text from training PDFs...")
    train_data = process_directory(TRAIN_DIR, LOG_FILE.replace(".log", "_train.log"))
    save_to_json(train_data, TRAIN_JSON)

    print("Extracting text from testing PDFs...")
    test_data = process_directory(TEST_DIR, LOG_FILE.replace(".log", "_test.log"))
    save_to_json(test_data, TEST_JSON)

    print("Text extraction and JSON saving completed.")
