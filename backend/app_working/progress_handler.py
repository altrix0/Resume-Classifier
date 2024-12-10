from tkinter import Toplevel, ttk

def show_progress_window(total_files):
    progress_window = Toplevel()
    progress_window.title("Processing Resumes")
    ttk.Label(progress_window, text="Processing resumes...").pack(pady=10)
    progress_bar = ttk.Progressbar(progress_window, length=400, mode="determinate")
    progress_bar.pack(pady=10)
    progress_bar["maximum"] = total_files
    return progress_window, progress_bar

def update_progress(progress_bar, current, total):
    progress = (current / total) * 100
    progress_bar["value"] = progress
    progress_bar.update()
