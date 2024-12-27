import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_features(image_path, kernel_size):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Sobel operator to detect edges (can be used as a feature extraction filter)
    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    # Calculate gradient magnitude
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Normalize the result to 0-255
    feature_image = np.uint8(np.absolute(gradient_magnitude))
    
    return image, feature_image

# Parameters
image_path = "uploads/image_upload.png"  # Replace with your image path
kernel_size = 1  # You can vary this value to control the extraction strength

# Feature extraction
original_image, feature_image = extract_features(image_path, kernel_size)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Feature Extracted Image\n(Kernel Size: {kernel_size})")
plt.imshow(feature_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
