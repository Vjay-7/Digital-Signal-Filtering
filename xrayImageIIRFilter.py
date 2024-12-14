import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

class XRayImageProcessor:
    def __init__(self, master):
        # Configure root window
        self.master = master
        master.title("X-ray Image Enhancement")