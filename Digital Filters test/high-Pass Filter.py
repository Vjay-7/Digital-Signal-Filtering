import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_high_pass_filter(image_path, filter_strength):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Apply Gaussian Blur to get a low-pass version of the image
    kernel_size = int(filter_strength) * 2 + 1
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Subtract the blurred image from the original image to create a high-pass filter effect
    high_pass_filtered_image = cv2.subtract(image, blurred_image)
    
    return image, high_pass_filtered_image


# Parameters
image_path = "uploads/image_upload.png"  # Replace with your image path
filter_strength = 11  # Control the strength of the filter (higher = stronger)

# Apply the filter
original_image, filtered_image = apply_high_pass_filter(image_path, filter_strength)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Filtered Image\n(Filter Strength: {filter_strength})")
plt.imshow(filtered_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
