import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel_filter(image_path, filter_strength):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Apply Sobel filter (filter_strength controls the kernel size)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=filter_strength)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=filter_strength)
    
    # Compute the magnitude of gradients
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Normalize to 8-bit values
    filtered_image = np.uint8(np.absolute(gradient_magnitude))
    
    return image, filtered_image


# Parameters
image_path = "uploads/image_upload.png"  # Replace with your image path
filter_strength = 5  # Control the strength of the filter (higher = stronger)

# Apply the filter
original_image, filtered_image = apply_sobel_filter(image_path, filter_strength)

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
