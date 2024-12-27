# X-ray Image Enhancement Tool

This Python application leverages **Digital Signal Processing (DSP)** techniques to enhance X-ray images by applying customizable digital filters. It is designed as an interactive tool for both practical image enhancement and educational exploration of DSP principles.

## Key Features

- **Digital Signal Filtering**:
  - Apply **Gaussian Blur** for noise reduction (spatial domain filtering).
  - Use **IIR Filters** (low-pass and high-pass) for frequency-based signal enhancement.
- **Interactive DSP Controls**:
  - Adjust filter parameters in real time and observe the effect on the image.
  - Understand the relationship between filter coefficients and frequency response.
- **Frequency Response Visualization**:
  - Plot low-pass and high-pass filter responses to learn how they influence the image.

## How It Works

1. **Upload an X-ray Image**: Select an X-ray image file to process.
2. **Adjust Noise Reduction**: Modify the Gaussian blur kernel size to remove unwanted high-frequency noise.
3. **Enhance Details**: Use the low-pass and high-pass filter sliders to emphasize or suppress specific frequency components.
4. **Visualize Frequency Response**: Observe the impact of filter coefficients on the signal's frequency behavior in real time.

## Demo

![X-ray Image Enhancement Demo](xrayDemo.mp4)

## Why DSP?

Digital Signal Processing (DSP) is a cornerstone of modern signal analysis and enhancement. This application showcases DSP principles applied to 2D signals (images), emphasizing how filtering techniques can:
- Reduce noise while preserving signal fidelity.
- Enhance specific details, such as edges, using high-pass filtering.
- Smooth large structures by attenuating high frequencies with low-pass filtering.

Explore the power of DSP with this interactive tool!
