import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_noise(image_path, kernel_size):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # Apply Gaussian Blur for noise reduction
    # The kernel size controls the strength of the filter (larger kernel = more smoothing)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return image, blurred_image

# Parameters
image_path = "uploads/image_upload.png"  # Replace with your image path
kernel_size = 7  # You can vary this value to control the noise reduction level (must be odd)

# Noise reduction
original_image, reduced_noise_image = reduce_noise(image_path, kernel_size)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Noise Reduced Image\n(Kernel Size: {kernel_size})")
plt.imshow(reduced_noise_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
