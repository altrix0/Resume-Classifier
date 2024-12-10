from tkinter import ttk

def update_progress(progress_bar, current, total):
    progress = (current / total) * 100
    progress_bar["value"] = progress
    progress_bar.update()
