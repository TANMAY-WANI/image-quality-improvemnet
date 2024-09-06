import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def dynamic_histogram_equalization(img):
    # Apply a simple histogram equalization based on dynamic ranges
    min_val, max_val = np.min(img), np.max(img)
    img_normalized = (img - min_val) / (max_val - min_val) * 255
    img_equalized = cv2.equalizeHist(np.uint8(img_normalized))
    return img_equalized

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, 0)  # Read image in grayscale
            
            equalized_img = dynamic_histogram_equalization(img)
            
            # Display and save the plot
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
            plt.subplot(1, 2, 2), plt.imshow(equalized_img, cmap='gray'), plt.title('DHE Image')
            
            output_img_dir = os.path.join(output_folder, 'Dynamic_Histogram_Equalization_DHE')
            os.makedirs(output_img_dir, exist_ok=True)
            plt.savefig(os.path.join(output_img_dir, filename))
            plt.close()

# Process all images in the specified folder
process_images('input_dataset', 'HE_output')
