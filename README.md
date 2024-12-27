Here’s an explanation of the provided code along with a guide to display the `xrayDemo.mp4` video in the `README.md` file on GitHub:

---

### **Code Explanation**

The code is a Python application built using `Tkinter` for graphical user interface (GUI) and other libraries for image processing and plotting. The application is designed to enhance X-ray images using adjustable parameters. Below is a breakdown of its components:

#### **1. Importing Libraries**
The code uses:
- **`tkinter` and `ttk`**: For creating the GUI.
- **`cv2` (OpenCV)**: For image loading, processing, and noise reduction.
- **`numpy`**: For efficient array manipulations.
- **`scipy.signal`**: To apply IIR filters for frequency domain processing.
- **`matplotlib`**: To plot frequency responses of filters.
- **`PIL` (Pillow)**: For handling and resizing images.

#### **2. Class `XRayImageProcessor`**
The main class encapsulates the GUI logic and image processing.

- **Initialization (`__init__`)**:
  - Configures the main window's layout, scrollable frame, and styling.
  - Calls `create_ui_components` to set up UI elements like buttons, labels, and sliders.

- **UI Components**:
  - **Image upload**: A button allows users to load X-ray images.
  - **Image display**: Original and enhanced images are shown side-by-side in the GUI.
  - **Filter controls**: Adjustable sliders control noise reduction and filter coefficients (low-pass and high-pass filters).
  - **Frequency response plot**: Uses `matplotlib` to visualize filter behavior.

- **Event Handlers**:
  - **`upload_image`**: Opens a file dialog to load an X-ray image, normalizes it, and initializes the processing.
  - **`update_filters`**: Applies noise reduction and IIR filters based on slider values and updates the UI.
  - **`plot_frequency_response`**: Visualizes the frequency response of the filters.

- **Helper Functions**:
  - **`apply_iir_filter`**: Applies an IIR filter along rows and columns of the image.
  - **`create_parameter_slider`**: Generates sliders for each adjustable parameter with tooltips for guidance.
