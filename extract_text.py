import os
import PyPDF2
import json

# Define paths
pdf_dir = "data/raw"  # Directory containing PDFs
output_file = "data/prepared/dataset.json"  # Path to save extracted text

def extract_text_from_pdfs(pdf_dir):
    data = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.endswith(".pdf"):
                category = os.path.basename(root)  # Folder name as category/label
                pdf_path = os.path.join(root, file)
                try:
                    # Read PDF
                    with open(pdf_path, "rb") as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text()
                        # Append extracted text with label
                        data.append({"category": category, "text": text})
                except Exception as e:
                    print(f"Error reading {pdf_path}: {e}")
    return data

def save_to_json(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Extracting text from PDFs...")
    dataset = extract_text_from_pdfs(pdf_dir)
    print(f"Extracted {len(dataset)} documents.")
    save_to_json(dataset, output_file)
    print(f"Saved dataset to {output_file}")
