import tkinter as tk
from tkinter import messagebox
from backend.file_uploader import upload_files
from backend.file_handler import extract_text_from_pdfs, organize_resumes
from backend.classifier import classify_resumes
from backend.progress_handler import update_progress
from frontend.progress_window import show_progress_window
import joblib

# Load pre-trained model and vectorizer
def load_model():
    model_path = "data/models/ensemble_model.pkl"
    with open(model_path, "rb") as file:
        data = joblib.load(file)
    return data["model"], data["vectorizer"]

def run_app():
    root = tk.Tk()
    root.title("Resume Classifier")
    root.geometry("600x400")

    model, vectorizer = load_model()

    def start_classification():
        # Upload files
        pdf_paths = upload_files()
        if not pdf_paths:
            messagebox.showwarning("No Files Selected", "Please select at least one PDF file.")
            return

        # Extract text
        extracted_data = extract_text_from_pdfs(pdf_paths)
        if not extracted_data:
            messagebox.showerror("Extraction Failed", "Failed to extract text from selected files.")
            return

        # Select categories
        categories = model.classes_
        selected_categories = ["Engineering", "Sales", "Finance"]  # Example; modify as needed
        if not selected_categories:
            messagebox.showwarning("No Categories Selected", "Please select at least one category.")
            return

        # Classification
        categorized_data = classify_resumes(extracted_data, model, vectorizer, selected_categories)

        # Show progress window
        progress_window, progress_bar = show_progress_window(len(pdf_paths))

        # Organize resumes with progress
        for i, pdf_path in enumerate(pdf_paths):
            organize_resumes(categorized_data, pdf_paths)
            update_progress(progress_bar, i + 1, len(pdf_paths))

        progress_window.destroy()
        messagebox.showinfo("Process Completed", "Resumes sorted successfully.")

    # GUI Elements
    tk.Label(root, text="Resume Classifier", font=("Helvetica", 16)).pack(pady=20)
    tk.Button(root, text="Start Classification", command=start_classification, font=("Helvetica", 14)).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    run_app()
