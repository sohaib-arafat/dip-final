import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, IntVar
import cv2
import numpy as np
import threading

root = tk.Tk()
root.title("Image Processing HW 2")

img = None
selected_filters = {
    "User Defined": False,
    "Point Detection": IntVar(),
    "Line Detection": IntVar(),
    "Edge Detection": IntVar(),
    "Laplacian Edge Detection": IntVar(),
    "Thresholding": IntVar(),
    "Horizontal Line Detection": IntVar(),
    "Vertical Line Detection": IntVar(),
    "45-Degree Line Detection": IntVar(),
    "-45-Degree Line Detection": IntVar(),
    "Laplacian of Gaussian": IntVar()
}

def apply_user_defined_filter(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def apply_point_detection(img):
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def apply_line_detection(img):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def apply_edge_detection(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.uint8(gradient_magnitude)

def apply_laplacian_edge_detection(img):
    return cv2.Laplacian(img, cv2.CV_64F)

def apply_thresholding(img):
    _, thresholded_img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    return thresholded_img

def apply_horizontal_line_detection(img):
    kernel = np.array([[-1, -1, -1],
                       [2, 2, 2],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def apply_vertical_line_detection(img):
    kernel = np.array([[-1, 2, -1],
                       [-1, 2, -1],
                       [-1, 2, -1]])
    return cv2.filter2D(img, -1, kernel)

def apply_45_degree_line_detection(img):
    kernel = np.array([[-1, -1, 2],
                       [-1, 2, -1],
                       [2, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def apply_minus_45_degree_line_detection(img):
    kernel = np.array([[2, -1, -1],
                       [-1, 2, -1],
                       [-1, -1, 2]])
    return cv2.filter2D(img, -1, kernel)

def apply_laplacian_of_gaussian(img):
    return cv2.Laplacian(cv2.GaussianBlur(img, (3, 3), 0), cv2.CV_64F)

def display_image(processed_img):
    cv2.imshow('Processed Image', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(kernel_entries):
    global img
    if img is None:
        messagebox.showerror("Error", "Please open an image first.")
        return

    try:
        kernel_values = [float(entry.get()) for row_entries in kernel_entries for entry in row_entries]
        kernel = np.array(kernel_values).reshape((len(kernel_entries), len(kernel_entries[0])))
        processed_img = apply_user_defined_filter(img, kernel)

        for filter_name, var in selected_filters.items():
            if filter_name != "User Defined" and var.get():
                processed_img = apply_filter(processed_img, filter_name)

        display_thread = threading.Thread(target=display_image, args=(processed_img,))
        display_thread.daemon = True
        display_thread.start()

    except ValueError:
        messagebox.showerror("Error", "Invalid input")

def open_image():
    global img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def create_kernel_input_window(size):
    kernel_input_window = tk.Toplevel(root)
    kernel_input_window.title(f"Kernel Input ({size}x{size})")

    kernel_entries = [[tk.Entry(kernel_input_window, width=5) for _ in range(size)] for _ in range(size)]

    for i, row_entries in enumerate(kernel_entries):
        for j, entry in enumerate(row_entries):
            entry.grid(row=i, column=j, padx=5, pady=5)

    apply_button = tk.Button(kernel_input_window, text="Apply Filter", command=lambda: process_image(kernel_entries))
    apply_button.grid(row=size, columnspan=size, pady=10)

def prompt_filter_size():
    size = simpledialog.askinteger("Filter Size", "Enter the size of the filter :",
                                   parent=root, minvalue=1)
    if size is not None:
        create_kernel_input_window(size)

def apply_filter(img, filter_name):
    if filter_name == "Point Detection":
        return apply_point_detection(img)
    elif filter_name == "Line Detection":
        return apply_line_detection(img)
    elif filter_name == "Edge Detection":
        return apply_edge_detection(img)
    elif filter_name == "Laplacian Edge Detection":
        return apply_laplacian_edge_detection(img)
    elif filter_name == "Thresholding":
        return apply_thresholding(img)
    elif filter_name == "Horizontal Line Detection":
        return apply_horizontal_line_detection(img)
    elif filter_name == "Vertical Line Detection":
        return apply_vertical_line_detection(img)
    elif filter_name == "45-Degree Line Detection":
        return apply_45_degree_line_detection(img)
    elif filter_name == "-45-Degree Line Detection":
        return apply_minus_45_degree_line_detection(img)
    elif filter_name == "Laplacian of Gaussian":
        return apply_laplacian_of_gaussian(img)
    else:
        return img

def apply_checkbox_filters():
    global img
    if img is None:
        messagebox.showerror("Error", "Please open an image first.")
        return

    try:
        processed_img = np.copy(img)

        for filter_name, var in selected_filters.items():
            if filter_name != "User Defined" and var.get():
                processed_img = apply_filter(processed_img, filter_name)

        display_thread = threading.Thread(target=display_image, args=(processed_img,))
        display_thread.daemon = True
        display_thread.start()

    except ValueError:
        messagebox.showerror("Error", "Can't apply dilters")

btn_load_image = tk.Button(root, text="Open Image", command=open_image)
btn_load_image.pack()

btn_prompt_filter_size = tk.Button(root, text="Filter Size", command=prompt_filter_size)
btn_prompt_filter_size.pack()

filter_checkboxes = {filter_name: tk.Checkbutton(root, text=filter_name, variable=var) for filter_name, var in selected_filters.items() if filter_name != "User Defined"}
for checkbox in filter_checkboxes.values():
    checkbox.pack()

btn_apply_filters = tk.Button(root, text="Apply Selected Filters", command=apply_checkbox_filters)
btn_apply_filters.pack()

root.mainloop()
