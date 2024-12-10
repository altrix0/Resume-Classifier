from tkinter import filedialog

def upload_files():
    pdf_paths = filedialog.askopenfilenames(
        title="Select PDF Files",
        filetypes=[("PDF Files", "*.pdf")],
        multiple=True
    )
    return pdf_paths
