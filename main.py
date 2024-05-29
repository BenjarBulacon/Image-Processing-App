import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.geometry("800x800")  
        self.root.configure(bg="light grey")

        self.saved_images_dir = "saved_images"
        os.makedirs(self.saved_images_dir, exist_ok=True)

        # Create a frame for the buttons and sliders on the left side
        self.control_frame = tk.Frame(root, bg="light grey")
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nw')

        self.load_button = tk.Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=2, padx=10, pady=5)

        self.webcam_button = tk.Button(self.control_frame, text="Webcam Edge Detection", command=self.start_webcam)
        self.webcam_button.grid(row=1, column=2, padx=10, pady=5)

        # Add separator
        self.separator = ttk.Separator(self.control_frame, orient='vertical')
        self.separator.grid(row=0, column=1, rowspan=8, padx=20, pady=10, sticky='ns')

        self.sobel_button = tk.Button(self.control_frame, text="Sobel Edge Detection", command=self.apply_sobel)
        self.sobel_button.grid(row=3, column=0, padx=5, pady=5)

        self.canny_button = tk.Button(self.control_frame, text="Canny Edge Detection", command=self.apply_canny)
        self.canny_button.grid(row=4, column=0, padx=5, pady=5)

        self.threshold_slider = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold Scale")
        self.threshold_slider.grid(row=5, column=0, padx=5, pady=5, sticky='w')

        self.segment_button = tk.Button(self.control_frame, text="Apply Threshold", command=self.apply_threshold)
        self.segment_button.grid(row=6, column=0, padx=5, pady=5, sticky='w')

        self.cluster_slider = tk.Scale(self.control_frame, from_=2, to=10, orient=tk.HORIZONTAL, label="K-Means Scale")
        self.cluster_slider.grid(row=7, column=0, padx=5, pady=5, sticky='w')

        self.kmeans_button = tk.Button(self.control_frame, text="Apply K-Means", command=self.apply_kmeans)
        self.kmeans_button.grid(row=8, column=0, padx=5, pady=5, sticky='w')

        self.original_image_label = tk.Label(root)
        self.original_image_label.grid(row=0, column=2, rowspan=4, padx=10, pady=10)

        self.processed_image_label = tk.Label(root)
        self.processed_image_label.grid(row=0, column=3, rowspan=4, padx=10, pady=10)

        self.save_button = tk.Button(root, text="Save Processed Image", command=self.save_image)
        self.save_button.grid(row=8, column=2, padx=10, pady=10)

        self.show_saved_button = tk.Button(root, text="Show Saved Images", command=self.show_saved_images)
        self.show_saved_button.grid(row=9, column=2, padx=10, pady=10)

        self.original_image = None
        self.processed_image = None
        self.capture = None
        self.webcam_running = False

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.processed_image = None
            self.display_image(self.original_image, self.original_image_label)

    def display_image(self, img, label):
        max_dim = 500
        height, width = img.shape[:2]

        if height > max_dim or width > max_dim:
            scaling_factor = min(max_dim / height, max_dim / width)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        label.config(image=img_tk)
        label.image = img_tk

    def apply_sobel(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.hypot(sobelx, sobely)
        sobel = np.uint8(sobel / sobel.max() * 255)

        self.processed_image = sobel
        self.display_image(sobel, self.processed_image_label)

    def apply_canny(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        threshold1 = self.threshold_slider.get()
        threshold2 = threshold1 * 2
        edges = cv2.Canny(gray, threshold1, threshold2)

        self.processed_image = edges
        self.display_image(edges, self.processed_image_label)

    def apply_threshold(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        threshold = self.threshold_slider.get()
        _, segmented = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        self.processed_image = segmented
        self.display_image(segmented, self.processed_image_label)

    def apply_kmeans(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return

        img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)

        K = self.cluster_slider.get()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented_image = segmented.reshape((img.shape))

        self.processed_image = segmented_image
        self.display_image(segmented_image, self.processed_image_label)

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.update_webcam()

    def update_webcam(self):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.processed_image = edges
            self.display_image(edges, self.processed_image_label)
            self.root.after(10, self.update_webcam)
        else:
            messagebox.showerror("Error", "Failed to access webcam")

    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to save!")
            return

        file_name = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")], initialdir=self.saved_images_dir)
        if file_name:
            cv2.imwrite(file_name, self.processed_image)
            messagebox.showinfo("Saved", f"Image saved to {file_name}")

    def show_saved_images(self):
        saved_images = os.listdir(self.saved_images_dir)
        if not saved_images:
            messagebox.showinfo("No Images", "No saved images found.")
            return

        top = tk.Toplevel(self.root)
        top.title("Saved Images")

        for idx, image_name in enumerate(saved_images):
            image_path = os.path.join(self.saved_images_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (250, 250), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            label = tk.Label(top, image=image)
            label.image = image  
            label.grid(row=idx // 3, column=idx % 3, padx=5, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
