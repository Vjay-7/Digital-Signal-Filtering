import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_laplacian_filter(image_path, filter_strength):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Apply Laplacian filter (filter_strength controls the kernel size)
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=filter_strength)
    
    # Normalize to 8-bit values
    filtered_image = np.uint8(np.absolute(laplacian))
    
    return image, filtered_image


# Parameters
image_path = "uploads/image_upload.png"  # Replace with your image path
filter_strength = 7  # Control the strength of the filter (higher = stronger)

# Apply the filter
original_image, filtered_image = apply_laplacian_filter(image_path, filter_strength)

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
