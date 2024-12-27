import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

class XRayImageProcessor:
    def __init__(self, master):
        # Configure root window
        self.master = master
        master.title("X-ray Image Enhancement")
        master.geometry("1400x900")
        master.configure(bg='#f0f0f0')

        # Create main container with scrollbar
        self.container = tk.Frame(master, bg='#f0f0f0')
        self.container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self.container, bg='#f0f0f0')
        self.scrollbar = ttk.Scrollbar(self.container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#f0f0f0')

        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack canvas and scrollbar
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))

        # Create UI components
        self.create_ui_components()

    def create_ui_components(self):
        # Header
        header_frame = ttk.Frame(self.scrollable_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(
            header_frame, 
            text="X-ray Image Enhancement Tool", 
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=10)

        # Upload Section
        upload_frame = ttk.Frame(self.scrollable_frame)
        upload_frame.pack(fill=tk.X, pady=10)
        
        upload_button = ttk.Button(
            upload_frame, 
            text="📤 Upload X-ray Image", 
            command=self.upload_image
        )
        upload_button.pack(expand=True)

        # Image Display Section
        image_display_frame = ttk.Frame(self.scrollable_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Original Image Frame
        original_frame = ttk.LabelFrame(image_display_frame, text="Original Image")
        original_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        self.original_label = ttk.Label(original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)

        # Enhanced Image Frame
        enhanced_frame = ttk.LabelFrame(image_display_frame, text="Enhanced Image")
        enhanced_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)
        
        self.enhanced_label = ttk.Label(enhanced_frame)
        self.enhanced_label.pack(fill=tk.BOTH, expand=True)

        # Filter Controls Section
        filter_controls_frame = ttk.LabelFrame(
            self.scrollable_frame, 
            text="Image Processing Parameters"
        )
        filter_controls_frame.pack(fill=tk.X, pady=10, padx=10)

        # Parameters with sliders and tooltips
        self.filter_parameters = [
            {
                'name': 'Noise Reduction', 
                'var': tk.IntVar(value=3),
                'range': (1, 31),
                'tooltip': "Gaussian blur kernel size. Larger values reduce noise but may blur details."
            },
            {
                'name': 'Low B1', 
                'var': tk.IntVar(value=20),
                'range': (0, 100),
                'tooltip': "Low-pass filter feed-forward coefficient. Controls current input sample weight."
            },
            {
                'name': 'Low A1', 
                'var': tk.IntVar(value=80),
                'range': (0, 100),
                'tooltip': "Low-pass filter feedback coefficient. Influences previous output samples."
            },
            {
                'name': 'High B1', 
                'var': tk.IntVar(value=100),
                'range': (0, 100),
                'tooltip': "High-pass filter feed-forward coefficient. Controls current input sample weight."
            },
            {
                'name': 'High A1', 
                'var': tk.IntVar(value=50),
                'range': (0, 100),
                'tooltip': "High-pass filter feedback coefficient. Influences previous output samples."
            }
        ]

        # Create sliders for each parameters
        for param in self.filter_parameters:
            self.create_parameter_slider(
                filter_controls_frame, 
                param['name'], 
                param['var'], 
                param['range'], 
                param['tooltip']
            )

        # Frequency Response Section
        freq_response_frame = ttk.LabelFrame(
            self.scrollable_frame, 
            text="Frequency Response"
        )
        freq_response_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        # Matplotlib Figure for Frequency Response
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#f0f0f0')
        self.fig.suptitle('Filter Frequency Response')
        
        # Embed matplotlib in Tkinter
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=freq_response_frame)
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize image variables
        self.image_original = None
        self.image = None
        self.enhanced_image = None

    def create_parameter_slider(self, parent, name, variable, range_val, tooltip):
        """Create a slider with label and info button"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5, padx=10)

        # Label
        label = ttk.Label(frame, text=name, width=15)
        label.pack(side=tk.LEFT)

        # Info Button
        info_button = ttk.Button(
            frame, 
            text="ℹ️", 
            width=3, 
            command=lambda: messagebox.showinfo(f"{name} Explanation", tooltip)
        )
        info_button.pack(side=tk.LEFT, padx=(0,5))

        # Slider
        slider = ttk.Scale(
            frame, 
            from_=range_val[0], 
            to=range_val[1], 
            orient=tk.HORIZONTAL, 
            variable=variable, 
            command=self.update_filters
        )
        slider.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Current Value Label
        value_label = ttk.Label(frame, textvariable=variable, width=5)
        value_label.pack(side=tk.LEFT, padx=(5,0))

    def upload_image(self):
        """Open file dialog to upload an image"""
        file_path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return

        try:
            # Read image in grayscale
            self.image_original = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if self.image_original is None:
                messagebox.showerror("Error", "Could not read the image file")
                return

            # Normalize image
            self.image = self.image_original / 255.0

            # Update filters and display images
            self.update_filters()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def apply_iir_filter(self, image, b, a):
        """Apply IIR filter along rows and columns"""
        # Apply IIR filter along rows
        filtered_rows = np.apply_along_axis(lambda row: lfilter(b, a, row), axis=1, arr=image)
        # Apply IIR filter along columns
        filtered_image = np.apply_along_axis(lambda col: lfilter(b, a, col), axis=0, arr=filtered_rows)
        return filtered_image

    def plot_frequency_response(self):
        """Plot frequency response of low-pass and high-pass filters"""
        if not hasattr(self, 'fig'):
            return

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Get current filter coefficients
        low_b1 = self.filter_parameters[1]['var'].get() / 100.0
        low_a1 = self.filter_parameters[2]['var'].get() / 100.0
        high_b1 = self.filter_parameters[3]['var'].get() / 100.0
        high_a1 = self.filter_parameters[4]['var'].get() / 100.0

        # Low-pass filter coefficients
        b_low = [low_b1, low_b1]
        a_low = [1.0, -low_a1]

        # High-pass filter coefficients
        b_high = [1.0, -high_b1]
        a_high = [1.0, -high_a1]

        # Compute frequency response
        w_low, h_low = freqz(b_low, a_low)
        w_high, h_high = freqz(b_high, a_high)

        # Plot magnitude response for Low-pass filter
        self.ax1.set_title('Low-pass Filter Frequency Response')
        self.ax1.plot(w_low, 20 * np.log10(np.abs(h_low)), 'b')
        self.ax1.set_ylabel('Magnitude [dB]')
        self.ax1.set_xlabel('Frequency [rad/sample]')
        self.ax1.grid(True)

        # Plot magnitude response for High-pass filter
        self.ax2.set_title('High-pass Filter Frequency Response')
        self.ax2.plot(w_high, 20 * np.log10(np.abs(h_high)), 'r')
        self.ax2.set_ylabel('Magnitude [dB]')
        self.ax2.set_xlabel('Frequency [rad/sample]')
        self.ax2.grid(True)

        # Adjust layout and redraw
        plt.tight_layout()
        self.canvas_widget.draw()

    def update_filters(self, val=None):
        """Update image filters based on slider values"""
        if self.image is None:
            return

        # Get noise reduction level (ensure odd)
        noise_level = self.filter_parameters[0]['var'].get()
        if noise_level % 2 == 0:
            noise_level += 1

        # Apply noise reduction
        denoised_image = cv2.GaussianBlur(self.image_original, (noise_level, noise_level), 0)

        # Get current slider values for filters
        low_b1 = self.filter_parameters[1]['var'].get() / 100.0
        low_a1 = self.filter_parameters[2]['var'].get() / 100.0
        high_b1 = self.filter_parameters[3]['var'].get() / 100.0
        high_a1 = self.filter_parameters[4]['var'].get() / 100.0

        # Update filter coefficients
        b_low = [low_b1, low_b1]
        a_low = [1.0, -low_a1]
        b_high = [1.0, -high_b1]
        a_high = [1.0, -high_a1]

        # Apply filters
        low_passed = self.apply_iir_filter(denoised_image / 255.0, b_low, a_low)
        high_passed = self.apply_iir_filter(denoised_image / 255.0, b_high, a_high)

        # Combine results
        enhanced_image = low_passed + high_passed

        # Rescale for visualization
        enhanced_image = np.clip(enhanced_image, 0, 1) * 255.0
        enhanced_image = enhanced_image.astype(np.uint8)

        # Display original image
        original_pil = Image.fromarray(self.image_original)
        original_pil = original_pil.resize((400, 400), Image.LANCZOS)
        original_photo = ImageTk.PhotoImage(original_pil)
        self.original_label.config(image=original_photo)
        self.original_label.image = original_photo

        # Display enhanced image
        enhanced_pil = Image.fromarray(enhanced_image)
        enhanced_pil = enhanced_pil.resize((400, 400), Image.LANCZOS)
        enhanced_photo = ImageTk.PhotoImage(enhanced_pil)
        self.enhanced_label.config(image=enhanced_photo)
        self.enhanced_label.image = enhanced_photo

        # Update frequency response plot
        self.plot_frequency_response()

def main():
    root = tk.Tk()
    app = XRayImageProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()