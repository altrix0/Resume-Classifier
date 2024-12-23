import os
import subprocess
from tkinter import Canvas, StringVar, filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
from backend.app_working.classifier import load_model, classify_resumes
from backend.app_working.file_handler import extract_text_from_pdfs, organize_resumes


class Window1:
    """Main window to input session details and start classification."""
    def __init__(self, master, output_path, categories, next_callback):
        self.master = master
        self.output_path = StringVar(value=output_path)
        self.session_name = StringVar()
        self.pdf_files = []
        self.categories = categories
        self.selected_categories = []
        self.next_callback = next_callback

        # Setup UI
        self.create_canvas()
        self.create_labels()
        self.create_buttons()
        self.create_text_boxes()

    def create_canvas(self):
        """Setup the canvas and background image."""
        self.canvas = Canvas(
            self.master, bg="#EFF6EF", height=700, width=900,
            bd=0, highlightthickness=0, relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        image = Image.open("frontend/assets/frame1/image_1.png")
        resized_image = image.resize((image.width, 750), Image.Resampling.LANCZOS)
        self.image_image_1 = ImageTk.PhotoImage(resized_image)
        self.image_label = ctk.CTkLabel(self.master, image=self.image_image_1, text="")
        self.image_label.place(x=-30, y=-25)

    def create_labels(self):
        """Setup text labels on the canvas."""
        self.canvas.create_text(315, 21, anchor="nw", text="Resume Classifier",
                                fill="#000000", font=("Courier Prime", 48 * -1))
        self.canvas.create_text(360, 80, anchor="nw", text="Effortless Resume Categorization",
                                fill="#000000", font=("Courier Prime", 22 * -1))
        self.canvas.create_text(315, 140, anchor="nw", text="Output Directory:",
                                fill="#0D0D0D", font=("Courier Prime", 20 * -1))
        self.canvas.create_text(315, 210, anchor="nw", text="Session Name:",
                                fill="#0D0D0D", font=("Courier Prime", 20 * -1))
        self.canvas.create_text(315, 290, anchor="nw", text="Upload Resume(s):",
                                fill="#0D0D0D", font=("Courier Prime", 20 * -1))
        self.canvas.create_text(315, 370, anchor="nw", text="Select Categories:",
                                fill="#0D0D0D", font=("Courier Prime", 20 * -1))

    def create_buttons(self):
        """Setup buttons for user interaction."""
        self.output_dir_button = ctk.CTkButton(
            master=self.master, text="Select Output Directory",
            width=250, height=30, command=self.select_output_dir,
            fg_color="#000000", text_color="#ffffff"
        )
        self.output_dir_button.place(x=587, y=140)

        self.upload_button = ctk.CTkButton(
            master=self.master, text="Upload",
            width=250, height=30, command=self.upload_files,
            fg_color="#000000", text_color="#ffffff"
        )
        self.upload_button.place(x=587, y=290)

        self.next_button = ctk.CTkButton(
            master=self.master, text="Next",
            width=100, height=40, command=self.validate_and_next,
            fg_color="#000000", text_color="#ffffff"
        )
        self.next_button.place(x=750, y=630)

        self.listbox = ctk.CTkScrollableFrame(
            self.master, width=250, height=150, fg_color="#000000"
        )
        self.listbox.place(x=580, y=360)

        self.checkbox_vars = {}
        for category in self.categories:
            var = StringVar(value="")
            self.checkbox_vars[category] = var
            checkbox = ctk.CTkCheckBox(
                self.listbox, text=category, variable=var,
                onvalue=category, offvalue="",
                fg_color="#EFF6EF", checkmark_color="#000000"
            )
            checkbox.pack(pady=2, fill="x")

    def create_text_boxes(self):
        """Setup input fields for session details."""
        self.session_entry = ctk.CTkEntry(
            master=self.master, textvariable=self.session_name,
            width=250, height=30, fg_color="#000000", text_color="#ffffff"
        )
        self.session_entry.place(x=587, y=210)

    def select_output_dir(self):
        """Select an output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_path.set(directory)

    def upload_files(self):
        """Select resumes for classification."""
        files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if files:
            self.pdf_files.extend(files)

    def validate_and_next(self):
        """Validate inputs and proceed to the next window."""
        if not self.session_name.get():
            messagebox.showerror("Error", "Session Name cannot be empty.")
            return
        session_dir = os.path.join(self.output_path.get(), self.session_name.get())
        if os.path.exists(session_dir):
            messagebox.showerror("Error", "Session already exists. Choose another name.")
            return
        if not self.pdf_files:
            messagebox.showerror("Error", "Please upload at least one resume.")
            return
        self.selected_categories = [cat for cat, var in self.checkbox_vars.items() if var.get()]
        if not self.selected_categories:
            messagebox.showerror("Error", "Please select at least one category.")
            return
        self.next_callback(self.output_path.get(), self.session_name.get(), self.selected_categories, self.pdf_files)


class Window2:
    """Display 'Resumes are being sorted' page."""
    def __init__(self, master, start_sorting):
        self.master = master
        self.start_sorting = start_sorting

        # Start sorting in a non-blocking way
        self.start_sorting()

        self.canvas = Canvas(
            self.master, bg="#EFF6EF", height=700, width=900,
            bd=0, highlightthickness=0, relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        image = Image.open("frontend/assets/frame1/image_1.png")
        resized_image = image.resize((image.width, 750), Image.Resampling.LANCZOS)
        self.image_image_1 = ImageTk.PhotoImage(resized_image)
        self.image_label = ctk.CTkLabel(self.master, image=self.image_image_1, text="")
        self.image_label.place(x=-30, y=-25)

        self.canvas.create_text(315, 21, anchor="nw", text="Resume Classifier",
                                fill="#000000", font=("Courier Prime", 48 * -1))
        self.canvas.create_text(360, 80, anchor="nw", text="Effortless Resume Categorization",
                                fill="#000000", font=("Courier Prime", 22 * -1))
        self.canvas.create_text(355, 300, anchor="nw", text="Resumes are being sorted...Please wait.",
                                fill="#0D0D0D", font=("Courier Prime", 18 * -1))



class Window3:
    """Final window to show success and open folder option."""
    def __init__(self, master, session_folder):
        self.master = master
        self.canvas = Canvas(
            self.master, bg="#EFF6EF", height=700, width=900,
            bd=0, highlightthickness=0, relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        image = Image.open("frontend/assets/frame1/image_1.png")
        resized_image = image.resize((image.width, 750), Image.Resampling.LANCZOS)
        self.image_image_1 = ImageTk.PhotoImage(resized_image)
        self.image_label = ctk.CTkLabel(self.master, image=self.image_image_1, text="")
        self.image_label.place(x=-30, y=-25)

        self.canvas.create_text(315, 21, anchor="nw", text="Resume Classifier",
                                fill="#000000", font=("Courier Prime", 48 * -1))
        self.canvas.create_text(360, 80, anchor="nw", text="Effortless Resume Categorization",
                                fill="#000000", font=("Courier Prime", 22 * -1))
        self.canvas.create_text(450, 250, anchor="nw", text="Thanks for using the app!",
                                fill="#0D0D0D", font=("Courier Prime", 18 * -1))

        self.open_folder_button = ctk.CTkButton(
            master=self.master, text="Open Sorted Folder", width=200, height=40,
            command=lambda: self.open_folder(session_folder),
            fg_color="#000000", text_color="#ffffff"
        )
        self.open_folder_button.place(x=495, y=400)

    def open_folder(self, folder_path):
        """Open the folder in the system's file explorer."""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(folder_path)
            elif os.name == 'posix':  # macOS/Linux
                subprocess.Popen(['xdg-open', folder_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {str(e)}")
