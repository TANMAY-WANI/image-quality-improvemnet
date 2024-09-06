import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def histogram_equalization_normalization(img):
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_equalized = cv2.equalizeHist(np.uint8(img_normalized))
    return img_equalized

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, 0)  # Read image in grayscale
            
            equalized_img = histogram_equalization_normalization(img)
            
            # Display and save the plot
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
            plt.subplot(1, 2, 2), plt.imshow(equalized_img, cmap='gray'), plt.title('Equalized Image')
            
            output_img_dir = os.path.join(output_folder, 'Histogram_Equalization_Using_OpenCV_Normalization')
            os.makedirs(output_img_dir, exist_ok=True)
            plt.savefig(os.path.join(output_img_dir, filename))
            plt.close()

# Process all images in the specified folder
process_images('input_dataset', 'HE_output')
