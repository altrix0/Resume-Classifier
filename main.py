from tkinter import Tk, messagebox
from threading import Thread
from backend.app_working.classifier import load_model, classify_resumes
from backend.app_working.file_handler import extract_text_from_pdfs, organize_resumes
from frontend.ui_main import Window1, Window2, Window3


class ResumeClassifierApp:
    def __init__(self):
        """Initialize the Resume Classifier application."""
        self.root = Tk()
        self.root.title("Resume Classifier")
        self.root.geometry("900x700")
        self.root.configure(bg="#EFF6EF")

        # Load model and categories
        self.model, self.vectorizer, self.label_encoder = load_model()
        self.categories = self.label_encoder.classes_

        # Initialize variables for session data
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
        """Launch the first window for input and configuration."""
        self.clear_window()
        self.window1 = Window1(
            self.root,
            self.output_path,
            self.categories,
            self.handle_window1_next,
        )

    def handle_window1_next(self, output_path, session_name, selected_categories, pdf_files):
        """Handle the transition from Window 1 to Window 2."""
        self.output_path = output_path
        self.session_name = session_name
        self.selected_categories = selected_categories
        self.pdf_files = pdf_files
        self.launch_window2()

    def launch_window2(self):
        """Launch the second window for the sorting process."""
        self.clear_window()
        # Pass start_sorting directly to Window2, which will invoke it internally
        self.window2 = Window2(self.root, self.start_sorting)

    def start_sorting(self):
        """Handle the resume sorting process."""
        def sorting_task():
            try:
                # Extract text from PDFs
                extracted_data = extract_text_from_pdfs(self.pdf_files)
                if not extracted_data:
                    raise Exception("Failed to extract text from the selected files.")

                # Classify resumes
                categorized_data = classify_resumes(
                    extracted_data,
                    self.model,
                    self.vectorizer,
                    self.selected_categories,
                    self.label_encoder,
                )

                # Organize resumes
                session_folder = organize_resumes(
                    categorized_data,
                    self.pdf_files,
                    self.selected_categories,
                    self.output_path,
                    self.session_name,
                )

                # Transition to the final window
                self.launch_window3(session_folder)

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

        # Run sorting logic in a separate thread
        Thread(target=sorting_task).start()

    def launch_window3(self, session_folder):
        """Launch the third window with success message and session details."""
        self.clear_window()
        self.window3 = Window3(self.root, session_folder)

    def run(self):
        """Run the main application loop."""
        self.root.mainloop()


if __name__ == "__main__":
    app = ResumeClassifierApp()
    app.run()
