from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def save_image(image, filename):
    """Save an image to the outputs folder."""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    cv2.imwrite(filepath, image)
    return filename

def compress_image(image, cutoff_ratio):
    """Compress the image by retaining a percentage of frequencies."""
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Compute the 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    # Create a circular mask based on cutoff_ratio
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cutoff = int(cutoff_ratio * min(rows, cols))  # Determine cutoff frequency
    cv2.circle(mask, (center_col, center_row), cutoff, 1, -1)

    # Apply the mask
    f_shift_filtered = f_shift * mask

    # Perform the inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    compressed_image = np.fft.ifft2(f_ishift)
    compressed_image = np.abs(compressed_image)

    # Normalize for visualization
    compressed_image = cv2.normalize(compressed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return compressed_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save uploaded image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'image_upload.png')
            file.save(filepath)

            # Load the image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return "Error: Uploaded file is not a valid image.", 400

            # Save the original image for display
            save_image(image, 'original_image.png')

            # Process the image with default compression
            compressed_image = compress_image(image, cutoff_ratio=0.1)
            save_image(compressed_image, 'compressed_image.png')

            return render_template('results.html', filename='image_upload.png')
    return render_template('index.html')

@app.route('/adjust', methods=['POST'])
def adjust():
    """Handle dynamic adjustments for compression."""
    data = request.json
    filename = 'image_upload.png'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return jsonify({'error': 'Uploaded file not found or invalid.'}), 400

    feature = data['feature']
    param = float(data['param'])  # Treat param as the cutoff ratio
    result = None

    if feature == 'compression':
        # Perform compression based on cutoff_ratio
        result = compress_image(image, cutoff_ratio=param)

    # Save the result and return its filename
    output_filename = f"{feature}_adjusted.png"
    save_image(result, output_filename)
    return jsonify({'output': output_filename})

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
