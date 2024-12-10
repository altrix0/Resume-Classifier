import os
import shutil
import json
import tkinter as tk
from tkinter import messagebox, Toplevel, Listbox, MULTIPLE, filedialog
from tkinter import ttk
from datetime import datetime
from PyPDF2 import PdfReader
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "prepared", "dataset.json")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "ensemble_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

# Load pre-trained model, vectorizer, and label encoder
def load_model():
    with open(MODEL_PATH, "rb") as file:
        data = joblib.load(file)
    return data["model"], data["vectorizer"], data["label_encoder"]

def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

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

def organize_resumes(categorized_data, pdf_paths, selected_categories):
    """Organize resumes into category-specific folders."""
    date_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(OUTPUT_DIR, date_folder)
    os.makedirs(output_folder, exist_ok=True)

    pdf_path_mapping = {os.path.basename(path): path for path in pdf_paths}

    for category, files in categorized_data.items():
        if category not in selected_categories:
            print(f"Skipping category not selected: {category}")
            continue

        label_folder = os.path.join(output_folder, category)
        os.makedirs(label_folder, exist_ok=True)
        for file in files:
            if file in pdf_path_mapping:
                shutil.copy(pdf_path_mapping[file], label_folder)
                print(f"Copied {file} to {label_folder}")
            else:
                print(f"File {file} not found in PDF paths.")

    return output_folder

def classify_resumes(data, model, vectorizer, selected_categories, label_encoder):
    filenames, texts = zip(*data)
    features = vectorizer.transform(texts).toarray()  # Convert to dense matrix
    predictions = model.predict(features)
    decoded_predictions = label_encoder.inverse_transform(predictions)

    categorized_data = {category: [] for category in selected_categories}
    for filename, label in zip(filenames, decoded_predictions):
        if label in selected_categories:  # Only include selected categories
            categorized_data[label].append(filename)

    print(f"Categorized Data: {categorized_data}")
    return categorized_data

def select_categories(categories):
    """Show a dialog box to select categories."""
    selected_categories = []

    def confirm_selection():
        nonlocal selected_categories
        selected_categories = [categories[i] for i in listbox.curselection()]
        selection_window.destroy()

    # Create category selection window
    selection_window = Toplevel()
    selection_window.title("Select Categories")
    tk.Label(selection_window, text="Select categories to classify resumes into:").pack(pady=10)

    listbox = Listbox(selection_window, selectmode=MULTIPLE, width=50, height=15)
    for category in categories:
        listbox.insert(tk.END, category)
    listbox.pack(pady=10)

    tk.Button(selection_window, text="Confirm", command=confirm_selection).pack(pady=10)
    selection_window.wait_window()
    return selected_categories

def upload_files():
    """Upload PDF files."""
    pdf_paths = filedialog.askopenfilenames(
        title="Select PDF Files",
        filetypes=[("PDF Files", "*.pdf")],
        multiple=True
    )
    return pdf_paths

def show_progress_window(total_files):
    """Show a progress window."""
    progress_window = Toplevel()
    progress_window.title("Processing Resumes")
    tk.Label(progress_window, text="Processing resumes...").pack(pady=10)
    progress_bar = ttk.Progressbar(progress_window, length=400, mode="determinate")
    progress_bar.pack(pady=10)
    progress_bar["maximum"] = total_files
    return progress_window, progress_bar

def update_progress(progress_bar, current, total):
    """Update progress bar."""
    progress = (current / total) * 100
    progress_bar["value"] = progress
    progress_bar.update()

def run_app():
    root = tk.Tk()
    root.title("Resume Classifier")
    root.geometry("600x400")

    # Load model and dataset labels
    model, vectorizer, saved_label_encoder = load_model()

    categories = saved_label_encoder.classes_

    def start_classification():
        pdf_paths = upload_files()
        if not pdf_paths:
            messagebox.showwarning("No Files Selected", "Please select at least one PDF file.")
            return

        extracted_data = extract_text_from_pdfs(pdf_paths)
        if not extracted_data:
            messagebox.showerror("Extraction Failed", "Failed to extract text from selected files.")
            return

        selected_categories = select_categories(categories)
        if not selected_categories:
            messagebox.showwarning("No Categories Selected", "Please select at least one category.")
            return

        categorized_data = classify_resumes(extracted_data, model, vectorizer, selected_categories, saved_label_encoder)

        progress_window, progress_bar = show_progress_window(len(pdf_paths))
        try:
            organize_resumes(categorized_data, pdf_paths, selected_categories)
            for i in range(len(pdf_paths)):
                update_progress(progress_bar, i + 1, len(pdf_paths))
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("Error", f"An error occurred during organization: {e}")
            return

        progress_window.destroy()
        messagebox.showinfo("Process Completed", "Resumes sorted successfully.")

    tk.Label(root, text="Resume Classifier", font=("Helvetica", 16)).pack(pady=20)
    tk.Button(root, text="Start Classification", command=start_classification, font=("Helvetica", 14)).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    run_app()
