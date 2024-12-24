import os
import subprocess

# Paths
BASE_DIR = "data/raw"

def convert_and_cleanup(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".doc") or file.endswith(".docx"):
                doc_path = os.path.join(root, file)
                pdf_path = os.path.splitext(doc_path)[0] + ".pdf"
                
                try:
                    # Convert using LibreOffice CLI
                    subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", root, doc_path], check=True)
                    print(f"Converted: {doc_path} -> {pdf_path}")
                    
                    # Delete the original file
                    os.remove(doc_path)
                    print(f"Deleted: {doc_path}")
                except Exception as e:
                    print(f"Error processing {doc_path}: {e}")

if __name__ == "__main__":
    print(f"Starting conversion in '{BASE_DIR}'...")
    convert_and_cleanup(BASE_DIR)
    print("Conversion completed!")
