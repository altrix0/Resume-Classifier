import os
import shutil
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from PyPDF2 import PdfReader

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

# Load pre-trained ensemble model and vectorizer
def load_model():
    model_path = os.path.join(MODELS_DIR, "ensemble_model.pkl")
    with open(model_path, "rb") as file:
        data = joblib.load(file)
    return data["model"], data["vectorizer"]

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

# Preprocess and classify resumes
def classify_resumes(data, model, vectorizer, selected_categories):
    filenames, texts = zip(*data)
    features = vectorizer.transform(texts).toarray()  # Convert sparse matrix to dense
    predictions = model.predict(features)
    categorized_data = {label: [] for label in selected_categories}
    for filename, label in zip(filenames, predictions):
        if label in selected_categories:  # Only include selected categories
            categorized_data[label].append(filename)
    return categorized_data

# Organize and save sorted resumes
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

# Updated function to select categories
def select_categories(label_encoder):
    """
    Allow the user to select categories from a list of classes.
    Args:
        label_encoder: The label encoder used during training to decode labels.
    Returns:
        List of selected categories.
    """
    categories = label_encoder.classes_  # Retrieve categories from the label encoder
    selected = []

    def add_selection():
        selected.extend([categories[i] for i in listbox.curselection()])
        selection_window.destroy()

    # Create selection window
    selection_window = tk.Tk()
    selection_window.title("Select Categories")
    tk.Label(selection_window, text="Select categories to classify resumes into:").pack(pady=10)

    listbox = tk.Listbox(selection_window, selectmode=tk.MULTIPLE, width=50, height=15)
    for category in categories:
        listbox.insert(tk.END, category)
    listbox.pack(pady=10)

    tk.Button(selection_window, text="OK", command=add_selection).pack(pady=10)
    selection_window.mainloop()

    return selected

# Updated Tkinter GUI function
def run_app():
    root = tk.Tk()
    root.title("Resume Classifier")
    root.geometry("600x400")

    # Load model and label encoder
    model_data = load_model()
    model, vectorizer = model_data["model"], model_data["vectorizer"]
    label_encoder = model_data["label_encoder"]

    def import_pdfs():
        pdf_paths = filedialog.askopenfilenames(
            title="Select PDF Files",
            filetypes=[("PDF Files", "*.pdf")],
            multiple=True  # Allow multiple file selection
        )
        if not pdf_paths:
            messagebox.showwarning("No Files Selected", "Please select at least one PDF file.")
            return

        # Let user select categories
        selected_categories = select_categories(label_encoder)
        if not selected_categories:
            messagebox.showwarning("No Categories Selected", "Please select at least one category.")
            return

        # Process PDFs
        extracted_data = extract_text_from_pdfs(pdf_paths)
        if not extracted_data:
            messagebox.showerror("Extraction Failed", "Failed to extract text from selected files.")
            return

        # Classify Resumes
        categorized_data = classify_resumes(extracted_data, model, vectorizer, selected_categories)

        # Organize Resumes
        output_folder = organize_resumes(categorized_data, pdf_paths)

        # Success Message
        messagebox.showinfo("Process Completed", f"Resumes sorted and saved to {output_folder}")

    # GUI Elements
    tk.Label(root, text="Resume Classifier", font=("Helvetica", 16)).pack(pady=20)
    tk.Button(root, text="Import PDF Files", command=import_pdfs, font=("Helvetica", 14)).pack(pady=20)
    tk.Label(root, text="Select PDFs to classify them into specific categories and organize them.", wraplength=400).pack(pady=10)

    root.mainloop()

# Updated load_model function to include the label encoder
def load_model():
    model_path = os.path.join(MODELS_DIR, "ensemble_model.pkl")
    with open(model_path, "rb") as file:
        data = joblib.load(file)
    return {
        "model": data["model"],
        "vectorizer": data["vectorizer"],
        "label_encoder": data["label_encoder"],  # Ensure label encoder is loaded
    }

if __name__ == "__main__":
    run_app()
