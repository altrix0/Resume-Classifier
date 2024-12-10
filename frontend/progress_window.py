import tkinter as tk
from tkinter import ttk

def show_progress_window(total_files):
    progress_window = tk.Tk()
    progress_window.title("Sorting Progress")
    tk.Label(progress_window, text="Sorting resumes...").pack(pady=10)

    progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
    progress_bar.pack(pady=20)

    return progress_window, progress_bar
