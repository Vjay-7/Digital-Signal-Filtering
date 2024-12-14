import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_fir_filter(image_path, kernel):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    # Apply FIR filter (2D Convolution)
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    return image, filtered_image

# Define FIR filter kernel (example: 3x3 Gaussian kernel)
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]], dtype=np.float32)

# Parameters
image_path = "uploads/image_upload.png"  # Replace with your image path

# Apply the FIR filter
original_image, filtered_image = apply_fir_filter(image_path, kernel)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("FIR Filtered Image")
plt.imshow(filtered_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
