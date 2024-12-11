from tkinter import Tk, messagebox
from backend.app_working.classifier import load_model, classify_resumes, select_categories
from backend.app_working.file_handler import extract_text_from_pdfs, organize_resumes
from backend.app_working.file_uploader import upload_files
from backend.app_working.progress_handler import show_progress_window, update_progress
from frontend.ui_main import Window1, Window2, Window3
import time

class ResumeClassifierApp:
    def __init__(self):
        self.root = Tk()
        self.root.title("Resume Classifier")
        self.root.geometry("900x700")
        self.root.configure(bg="#EFF6EF")

        # Initialize backend components
        self.model, self.vectorizer, self.label_encoder = load_model()
        self.categories = self.label_encoder.classes_

        # Initialize frontend windows
        self.window1 = None
        self.window2 = None
        self.window3 = None

        # Hold session-specific data
        self.output_path = "data/outputs"
        self.session_name = None
        self.pdf_files = []
        self.selected_categories = []

        # Start the app with Window 1
        self.launch_window1()

    def clear_window(self):
        """Clear all widgets from the current window."""
        for widget in self.root.winfo_children():
            widget.destroy()

    def launch_window1(self):
        self.clear_window()
        self.window1 = Window1(
            self.root,
            self.output_path,
            self.categories,
            self.handle_window1_next,
        )

    def handle_window1_next(self, output_path, session_name, selected_categories, pdf_files):
        self.output_path = output_path
        self.session_name = session_name
        self.selected_categories = selected_categories
        self.pdf_files = pdf_files

        self.launch_window2()

    def launch_window2(self):
        self.clear_window()
        self.window2 = Window2(
            self.root,
            len(self.pdf_files),
            self.output_path,
            self.pdf_files,
            self.selected_categories,
            self.session_name 
    )


    def start_sorting(self):
        try:
            # Extract text
            extracted_data = extract_text_from_pdfs(self.pdf_files)
            if not extracted_data:
                raise Exception("Failed to extract text from selected files.")

            # Classify resumes
            categorized_data = classify_resumes(
                extracted_data,
                self.model,
                self.vectorizer,
                self.selected_categories,
                self.label_encoder,
            )

            # Organize resumes (remove the progress_callback argument)
            session_folder = organize_resumes(
                categorized_data,
                self.pdf_files,
                self.selected_categories,
                self.output_path,
                self.session_name
            )

            # Launch success window after sorting
            self.launch_window3(session_folder)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def launch_window3(self, session_folder):
        self.clear_window()
        self.window3 = Window3(self.root, session_folder)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ResumeClassifierApp()
    app.run()