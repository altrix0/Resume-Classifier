from tkinter import filedialog

def upload_files():
    """
    Open a file dialog to select multiple PDF files for upload.

    Returns:
        list: List of selected PDF file paths.
    """
    pdf_paths = filedialog.askopenfilenames(
        title="Select PDF Files",  # Dialog title
        filetypes=[("PDF Files", "*.pdf")],  # Allow only PDF files
        multiple=True  # Enable selecting multiple files
    )
    return pdf_paths
