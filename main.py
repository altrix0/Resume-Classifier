import tkinter as tk
from tkinter import messagebox
from backend.app_working.classifier import load_model, classify_resumes, select_categories
from backend.app_working.file_handler import extract_text_from_pdfs, organize_resumes
from backend.app_working.file_uploader import upload_files
from backend.app_working.progress_handler import show_progress_window, update_progress
from frontend.ui_main import run_app

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

        categorized_data = classify_resumes(
            extracted_data, model, vectorizer, selected_categories, saved_label_encoder
        )

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
