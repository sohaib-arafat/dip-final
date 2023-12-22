import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
import cv2
import numpy as np

# Global variable to hold the image
img = None

# Initialize the main window
root = tk.Tk()
root.title("Image Processing Techniques")

# Function to apply the user-defined filter
def apply_user_defined_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# Function to prompt the user for the filter size and coefficients
def prompt_filter_coefficients():
    filter_size = simpledialog.askinteger("Filter Size", "Enter the size of the filter (e.g., 3 for a 3x3 filter):",
                                          parent=root, minvalue=1, maxvalue=11)
    if filter_size:
        coeff_window = tk.Toplevel(root)
        coeff_window.title("Filter Coefficients")
        entries = {}
        for i in range(filter_size):
            for j in range(filter_size):
                entry = tk.Entry(coeff_window, width=5)
                entry.grid(row=i, column=j)
                entries[(i, j)] = entry

        def collect_coefficients():
            try:
                kernel = np.zeros((filter_size, filter_size), dtype=np.float32)
                for i in range(filter_size):
                    for j in range(filter_size):
                        kernel[i, j] = float(entries[(i, j)].get())
                coeff_window.destroy()
                process_image(kernel)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid coefficients.")

        btn_apply = tk.Button(coeff_window, text="Apply Filter", command=collect_coefficients)
        btn_apply.grid(row=filter_size, columnspan=filter_size)

# Function to open an image and apply the user-defined filter
def process_image(kernel=None):
    global img
    if not img:
        return
    processed_img = img.copy()
    if kernel is not None:
        processed_img = apply_user_defined_filter(processed_img, kernel)
    cv2.imshow('Processed Image', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to open an image
def open_image():
    global img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)

# Image and filter buttons
btn_load_image = tk.Button(root, text="Open Image", command=open_image)
btn_load_image.pack()

btn_apply_filter = tk.Button(root, text="Apply Custom Filter", command=prompt_filter_coefficients)
btn_apply_filter.pack()

# Start the GUI event loop
root.mainloop()
