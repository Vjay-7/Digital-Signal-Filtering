import numpy as np
import cv2
import matplotlib.pyplot as plt

def compress_image(image_path, cutoff_ratio):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Compute the 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)  # Shift zero frequency to the center
    
    # Get the dimensions of the image
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Create a mask with a circular low-pass filter
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cutoff = int(cutoff_ratio * min(rows, cols))  # Determine cutoff frequency
    cv2.circle(mask, (center_col, center_row), cutoff, 1, -1)

    # Apply the mask
    f_shift_filtered = f_shift * mask

    # Perform the inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    compressed_image = np.fft.ifft2(f_ishift)
    compressed_image = np.abs(compressed_image)

    return image, compressed_image

# Parameters
image_path = "uploads/image_upload.png"  # Updated image path
cutoff_ratio = 0.6  # Ratio of frequencies to retain (0.2 = 20%)

# Compress the image
original_image, compressed_image = compress_image(image_path, cutoff_ratio)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Compressed Image\n(Cutoff Ratio: {cutoff_ratio})")
plt.imshow(compressed_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
