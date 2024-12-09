# # LIST FOLDERS AND FILES

# import customtkinter as ctk
# from tkinter import filedialog, messagebox

# # Sample data structure
# data = {
#     "Category 1": ["Resume1.pdf", "Resume2.pdf"],
#     "Category 2": ["Resume3.pdf", "Resume4.pdf"],
#     "Category 3": ["Resume5.pdf", "Resume6.pdf"]
# }

# class ResumeApp(ctk.CTk):
#     def __init__(self):
#         super().__init__()

#         self.title("Folder and Resume Viewer")
#         self.geometry("500x500")
        
#         # Label for Folder Listbox
#         self.folder_label = ctk.CTkLabel(self, text="Folders (Categories):", font=("Arial", 14))
#         self.folder_label.pack(pady=5)

#         # Folder Listbox
#         self.folder_frame = ctk.CTkScrollableFrame(self, width=400, height=150)
#         self.folder_frame.pack(pady=5, padx=10, fill="x")
        
#         self.folder_buttons = {}
#         self.create_folder_list(data.keys())

#         # Label for Resume Listbox
#         self.resume_label = ctk.CTkLabel(self, text="Resumes in Selected Folder:", font=("Arial", 14))
#         self.resume_label.pack(pady=5)

#         # Resume Listbox
#         self.resume_frame = ctk.CTkScrollableFrame(self, width=400, height=150)
#         self.resume_frame.pack(pady=5, padx=10, fill="x")
        
#         self.resume_buttons = {}
#         self.create_resume_list([])

#         # Export Button
#         self.export_button = ctk.CTkButton(self, text="Export Selected Folder", command=self.export_data)
#         self.export_button.pack(pady=10)

#     def create_folder_list(self, folders):
#         """Create folder buttons dynamically."""
#         for folder in folders:
#             folder_button = ctk.CTkButton(
#                 self.folder_frame, 
#                 text=folder, 
#                 command=lambda f=folder: self.on_folder_select(f)
#             )
#             folder_button.pack(pady=5, padx=10, fill="x")
#             self.folder_buttons[folder] = folder_button

#     def create_resume_list(self, resumes):
#         """Create resume buttons dynamically."""
#         # Clear existing resumes
#         for widget in self.resume_frame.winfo_children():
#             widget.destroy()
#         self.resume_buttons.clear()

#         # Add new resume buttons
#         for resume in resumes:
#             resume_button = ctk.CTkButton(self.resume_frame, text=resume)
#             resume_button.pack(pady=5, padx=10, fill="x")
#             self.resume_buttons[resume] = resume_button

#     def on_folder_select(self, folder):
#         """Handle folder selection and update resumes."""
#         resumes = data.get(folder, [])
#         self.create_resume_list(resumes)

#     def export_data(self):
#         """Export resumes from selected folder."""
#         # Get the selected folder (from active folder button)
#         selected_folder = None
#         for folder, button in self.folder_buttons.items():
#             if button.cget("state") == "active":
#                 selected_folder = folder
#                 break

#         if not selected_folder:
#             messagebox.showwarning("No Folder Selected", "Please select a folder to export.")
#             return

#         resumes = data.get(selected_folder, [])
#         if resumes:
#             # Prompt user to select export location
#             export_path = filedialog.askdirectory(title="Select Export Directory")
#             if export_path:
#                 messagebox.showinfo("Export Success", f"Resumes exported to {export_path}")
#                 # Logic to copy resumes to the selected directory can be added here
#             else:
#                 messagebox.showwarning("Export Cancelled", "No directory selected for export.")
#         else:
#             messagebox.showwarning("No Data", "No resumes to export in the selected folder.")

# # Run the application
# if __name__ == "__main__":
#     ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
#     ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"
#     app = ResumeApp()
#     app.mainloop()


# LISTBOX + CHECKBOX


# import customtkinter as ctk

# # Initialize the CTk Application
# ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
# ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

# class App(ctk.CTk):
#     def __init__(self):
#         super().__init__()

#         self.title("Custom Listbox with Checkboxes")
#         self.geometry("400x500")

#         # Frame for Listbox
#         self.listbox_frame = ctk.CTkScrollableFrame(self, width=300, height=400)
#         self.listbox_frame.pack(pady=20, padx=20, fill="both", expand=True)

#         # Add Checkboxes to the Listbox Frame
#         self.checkboxes = []
#         self.categories = [
#             "Software Developer", 
#             "Data Scientist", 
#             "Project Manager", 
#             "AI Specialist", 
#             "Cybersecurity Analyst"
#         ]
#         self.create_checkboxes(self.categories)

#         # Button to Retrieve Selected Categories
#         self.submit_button = ctk.CTkButton(self, text="Get Selected Categories", command=self.get_selected)
#         self.submit_button.pack(pady=10)

#     def create_checkboxes(self, items):
#         """Creates CTkCheckboxes dynamically based on the provided items list."""
#         for item in items:
#             checkbox = ctk.CTkCheckBox(self.listbox_frame, text=item)
#             checkbox.pack(pady=5, padx=10, anchor="w")  # Adjust alignment as needed
#             self.checkboxes.append(checkbox)

#     def get_selected(self):
#         """Retrieves all selected checkboxes and prints their values."""
#         selected = [cb.cget("text") for cb in self.checkboxes if cb.get()]
#         print("Selected Categories:", selected)


# # Run the Application
# if __name__ == "__main__":
#     app = App()
#     app.mainloop()
