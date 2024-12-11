import os
import subprocess
from datetime import datetime
from tkinter import Canvas, Entry, Button, PhotoImage, StringVar, messagebox, filedialog
from backend.app_working.classifier import load_model, classify_resumes, select_categories
from backend.app_working.file_handler import extract_text_from_pdfs, organize_resumes
from backend.app_working.file_uploader import upload_files
from backend.app_working.progress_handler import show_progress_window, update_progress
import customtkinter as ctk
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk


from tkinter import Canvas, StringVar, filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk

class Window1:
    def __init__(self, master, output_path, categories, next_callback):
        self.master = master
        self.output_path = StringVar(value=output_path)
        self.session_name = StringVar()
        self.pdf_files = []
        self.categories = categories
        self.selected_categories = []  # To store selected categories
        self.next_callback = next_callback

        # Canvas for background
        self.canvas = Canvas(
            master,
            bg="#EFF6EF",
            height=700,
            width=900,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )
        self.canvas.place(x=0, y=0)

        # Align the image without compressing width and move it slightly out of the frame
        original_image = Image.open("frontend/assets/frame1/image_1.png")
        resized_image = original_image.resize((original_image.width, 750), Image.Resampling.LANCZOS)  # Keep original width
        self.image_image_1 = ImageTk.PhotoImage(resized_image)

        self.image_label = ctk.CTkLabel(
            self.master,
            image=self.image_image_1,
            text="",
        )
        self.image_label.place(x=-30, y=-25)  # Move part of the image outside the frame on the left

        self.create_labels()
        self.create_buttons()
        self.create_text_boxes()

    def create_labels(self):
        # Move headings slightly to the left
        self.canvas.create_text(
            315.0, 21.0,  # Adjusted x position
            anchor="nw",
            text="Resume Classifier",
            fill="#000000",
            font=("Courier Prime", 48 * -1),
        )
        self.canvas.create_text(
            360.0, 80.0,  # Adjusted x position
            anchor="nw",
            text="Effortless Resume Categorization",
            fill="#000000",
            font=("Courier Prime", 22 * -1),
        )
        self.canvas.create_text(
            315.0, 140.0,
            anchor="nw",
            text="Output Directory:",
            fill="#0D0D0D",
            font=("Courier Prime", 20 * -1),
        )
        self.canvas.create_text(
            315.0, 210.0,
            anchor="nw",
            text="Session Name:",
            fill="#0D0D0D",
            font=("Courier Prime", 20 * -1),
        )
        self.canvas.create_text(
            315.0, 290.0,
            anchor="nw",
            text="Upload Resume(s):",
            fill="#0D0D0D",
            font=("Courier Prime", 20 * -1),
        )
        self.canvas.create_text(
            315.0, 370.0,
            anchor="nw",
            text="Select Categories:",
            fill="#0D0D0D",
            font=("Courier Prime", 20 * -1),
        )

    def create_buttons(self):
        self.output_dir_button = ctk.CTkButton(
            master=self.master,
            text="Select Output Directory",
            width=250,
            height=30,
            command=self.select_output_dir,
            fg_color="#000000",  # Background color
            text_color="#ffffff"
        )
        self.output_dir_button.place(x=587, y=140)

        self.upload_button = ctk.CTkButton(
            master=self.master,
            text="Upload",
            width=250,
            height=30,
            command=self.upload_files,
            fg_color="#000000",  # Background color
            text_color="#ffffff"
        )
        self.upload_button.place(x=587, y=290)

        self.next_button = ctk.CTkButton(
            master=self.master,
            text="Next",
            width=100,
            height=40,
            command=self.validate_and_next,
            fg_color="#000000",  # Background color
            text_color="#ffffff"
        )
        self.next_button.place(x=750, y=630)

        self.listbox = ctk.CTkScrollableFrame(
            self.master,
            width=250,
            height=150,
            fg_color="#000000"  # Background color
        )
        self.listbox.place(x=580, y=360)

        # Add categories as checkboxes to the listbox
        self.checkbox_vars = {}  # Store variables linked to checkboxes
        for category in self.categories:
            var = StringVar(value="")  # Default: Unselected (empty string)
            self.checkbox_vars[category] = var
            checkbox = ctk.CTkCheckBox(
                self.listbox,
                text=category,
                variable=var,
                onvalue=category,
                offvalue="",
                width=240,
                hover_color="#EFF6EF",
                fg_color="EFF6EF",
                checkmark_color="#000000"
            )
            checkbox.pack(pady=2, fill="x")

    def create_text_boxes(self):
        self.session_entry = ctk.CTkEntry(
            master=self.master,
            textvariable=self.session_name,
            width=250,
            height=30,
            fg_color="#000000",  # Background color
            text_color="#ffffff"
        )
        self.session_entry.place(x=587, y=210)

    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_path.set(directory)

    def upload_files(self):
        files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if files:
            self.pdf_files.extend(files)

    def validate_and_next(self):
        if not self.session_name.get():
            messagebox.showerror("Error", "Session Name cannot be empty.")
            return

        session_dir = os.path.join(self.output_path.get(), self.session_name.get())
        if os.path.exists(session_dir):
            messagebox.showerror(
                "Error", "Session already exists. Please choose a different name."
            )
            return

        if not self.pdf_files:
            messagebox.showerror("Error", "Please upload at least one resume.")
            return

        # Retrieve selected categories
        self.selected_categories = [
            category for category, var in self.checkbox_vars.items() if var.get()
        ]
        if not self.selected_categories:
            messagebox.showerror("Error", "Please select at least one category.")
            return

        self.next_callback(
            self.output_path.get(),
            self.session_name.get(),
            self.selected_categories,  # Pass selected categories
            self.pdf_files,
        )


class Window2:
    def __init__(self, master, total_files, output_folder, pdf_files, selected_categories, session_name):
        self.master = master
        self.total_files = total_files
        self.sorted_files = 0
        self.output_folder = output_folder
        self.pdf_files = pdf_files
        self.selected_categories = selected_categories
        self.session_name = session_name

        # Initialize the GUI elements
        self.canvas = Canvas(
            master,
            bg="#EFF6EF",
            height=700,
            width=900,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )
        self.canvas.place(x=0, y=0)

         # Align the image without compressing width and move it slightly out of the frame
        original_image = Image.open("frontend/assets/frame1/image_1.png")
        resized_image = original_image.resize((original_image.width, 750), Image.Resampling.LANCZOS)  # Keep original width
        self.image_image_1 = ImageTk.PhotoImage(resized_image)

        self.image_label = ctk.CTkLabel(
            self.master,
            image=self.image_image_1,
            text="",
        )
        self.image_label.place(x=-30, y=-25)  # Move part of the image outside the frame on the left

        self.create_labels()
        self.progress_bar = Progressbar(master, orient="horizontal", length=500, mode="determinate")
        self.progress_bar.place(x=350, y=300)

        self.sort_resumes()  # Start sorting when window is initialized

    def create_labels(self):
        self.canvas.create_text(
            315.0,
            21.0,
            anchor="nw",
            text="Resume Classifier",
            fill="#000000",
            font=("Courier Prime", 48 * -1),
        )
        self.canvas.create_text(
            360.0,
            80.0,
            anchor="nw",
            text="Effortless Resume Categorization",
            fill="#000000",
            font=("Courier Prime", 22 * -1),
        )
        self.canvas.create_text(
            300.0,
            250.0,
            anchor="nw",
            text="Please wait while the resumes are being sorted...",
            fill="#0D0D0D",
            font=("Courier Prime", 18 * -1),
        )

    def update_progress(self):
        self.sorted_files += 1
        self.progress_bar["value"] = (self.sorted_files / self.total_files) * 100
        self.master.update_idletasks()

    def sort_resumes(self):
        try:
            # Extract text from PDFs
            extracted_data = extract_text_from_pdfs(self.pdf_files)
            if not extracted_data:
                raise Exception("Failed to extract text from selected files.")

            # Classify resumes
            categorized_data = classify_resumes(
                extracted_data, load_model()[0], load_model()[1], self.selected_categories, load_model()[2]
            )

            # Organize resumes using session name
            organize_resumes(
                categorized_data,
                self.pdf_files,
                self.selected_categories,
                self.output_folder,
                self.session_name  # Use session name here
            )

            for _ in self.pdf_files:
                self.update_progress()  # Simulate progress update

            self.launch_window3()  # Once sorting is done, go to Window3

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")



    def launch_window3(self):
        # Clear the current window and proceed to the next step
        self.clear_window()
        Window3(self.master, self.output_folder)

    def clear_window(self):
        for widget in self.master.winfo_children():
            widget.destroy()


class Window3:
    def __init__(self, master, session_folder):
        self.master = master

        self.canvas = Canvas(
            master,
            bg="#EFF6EF",
            height=700,
            width=900,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )
        self.canvas.place(x=0, y=0)

       # Align the image without compressing width and move it slightly out of the frame
        original_image = Image.open("frontend/assets/frame1/image_1.png")
        resized_image = original_image.resize((original_image.width, 750), Image.Resampling.LANCZOS)  # Keep original width
        self.image_image_1 = ImageTk.PhotoImage(resized_image)

        self.image_label = ctk.CTkLabel(
            self.master,
            image=self.image_image_1,
            text="",
        )
        self.image_label.place(x=-30, y=-25)  # Move part of the image outside the frame on the left


        self.create_labels()
        self.create_buttons(session_folder)

    def create_labels(self):
        self.canvas.create_text(
            315.0,
            21.0,
            anchor="nw",
            text="Resume Classifier",
            fill="#000000",
            font=("Courier Prime", 48 * -1),
        )
        self.canvas.create_text(
            360.0,
            80.0,
            anchor="nw",
            text="Effortless Resume Categorization",
            fill="#000000",
            font=("Courier Prime", 22 * -1),
        )
        self.canvas.create_text(
            450.0,
            250.0,
            anchor="nw",
            text="Thanks for using the app!",
            fill="#0D0D0D",
            font=("Courier Prime", 18 * -1),
        )

    def create_buttons(self, session_folder):
        self.open_folder_button = ctk.CTkButton(
            master=self.master,
            text="Open Sorted Folder",
            width=200,
            height=40,
            command=lambda: self.open_folder(session_folder),
            fg_color="#000000",  # Background color
            text_color="#ffffff" 
        )
        self.open_folder_button.place(x=495, y=400)

    def open_folder(self, folder_path):
        """Open the folder in the system's file explorer."""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(folder_path)
            elif os.name == 'posix':  # Linux/macOS
                subprocess.Popen(['xdg-open', folder_path])
            else:
                messagebox.showerror("Error", "Unsupported Operating System")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {str(e)}")

