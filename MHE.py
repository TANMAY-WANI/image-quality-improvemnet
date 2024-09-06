import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def multi_histogram_equalization(img):
    # Apply histogram equalization in multiple segments
    num_segments = 4
    height, width = img.shape
    segment_height = height // num_segments
    img_equalized = np.zeros_like(img)
    
    for i in range(num_segments):
        start_y = i * segment_height
        end_y = (i + 1) * segment_height if i < num_segments - 1 else height
        segment = img[start_y:end_y, :]
        equalized_segment = cv2.equalizeHist(segment)
        img_equalized[start_y:end_y, :] = equalized_segment
    
    return img_equalized

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, 0)  # Read image in grayscale
            
            equalized_img = multi_histogram_equalization(img)
            
            # Display and save the plot
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
            plt.subplot(1, 2, 2), plt.imshow(equalized_img, cmap='gray'), plt.title('MHE Image')
            
            output_img_dir = os.path.join(output_folder, 'Multi_Histogram_Equalization_MHE')
            os.makedirs(output_img_dir, exist_ok=True)
            plt.savefig(os.path.join(output_img_dir, filename))
            plt.close()

# Process all images in the specified folder
process_images('input_dataset', 'HE_output')
