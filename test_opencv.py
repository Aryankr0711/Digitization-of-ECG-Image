import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_synthetic(image_path):
    image = cv2.imread(image_path)
    # image is BGR
    image = cv2.resize(image, (5600, 1700), interpolation=cv2.INTER_LINEAR)
    image = image[:1696, :5600]
    
    # We want to extract the black/blue lines.
    # Red grid has high R, so if we just look at the Blue channel, red grid is dark?
    # No, if the grid is pink, B might be high too.
    # Let's convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find dark pixels (the signal)
    # The grid is usually lighter than the black signal.
    # Let's see the histogram or just threshold at 100
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    cv2.imwrite("mask_test.png", mask)
    print("Mask saved to mask_test.png")
    
    # We can also use color: black/dark blue has low R, G, B.
    # Red has high R, low G, B.
    
    mask_color = cv2.inRange(image, np.array([0, 0, 0]), np.array([120, 120, 120]))
    cv2.imwrite("mask_color_test.png", mask_color)
    print("Color mask saved to mask_color_test.png")

extract_synthetic(r"D:\mdm_proj\Digitization-of-ECG-Image\train\129883643\129883643-0001.png")
