import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def histogram_equalization_yuv(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_equalized

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)  # Read image in color
            
            equalized_img = histogram_equalization_yuv(img)
            
            # Display and save the plot
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
            plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)), plt.title('Equalized Image')
            
            output_img_dir = os.path.join(output_folder, 'Histogram_Equalization_on_Color_Images_YUV')
            os.makedirs(output_img_dir, exist_ok=True)
            plt.savefig(os.path.join(output_img_dir, filename))
            plt.close()

# Process all images in the specified folder
process_images('input_dataset', 'HE_output')
