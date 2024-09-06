import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def dualistic_sub_image_histogram_equalization(img):
    median = np.median(img)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    # Split the bins for lower and upper parts
    lower_bins = bins[:int(median)+1]  # Include median in lower_bins
    upper_bins = bins[int(median):]  # From median to the end in upper_bins
    
    # Split histogram at the median value
    lower_hist = hist[:int(median)]
    upper_hist = hist[int(median):]
    
    # Equalize each part separately
    cdf_lower = np.cumsum(lower_hist)
    cdf_upper = np.cumsum(upper_hist)
    
    # Normalize CDFs to the range [0, 255]
    cdf_lower = (cdf_lower - cdf_lower.min()) * 255 / (cdf_lower.max() - cdf_lower.min())
    cdf_upper = (cdf_upper - cdf_upper.min()) * 255 / (cdf_upper.max() - cdf_upper.min())
    
    img_equalized = img.copy()
    
    # Interpolate for the lower and upper parts separately
    img_equalized[img < median] = np.interp(img[img < median], lower_bins[:-1], cdf_lower)
    img_equalized[img >= median] = np.interp(img[img >= median], upper_bins[:-1], cdf_upper)
    
    return img_equalized

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, 0)  # Read image in grayscale
            
            equalized_img = dualistic_sub_image_histogram_equalization(img)
            
            # Display and save the plot
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
            plt.subplot(1, 2, 2), plt.imshow(equalized_img, cmap='gray'), plt.title('DSIHE Image')
            
            output_img_dir = os.path.join(output_folder, 'Dualistic_Sub_Image_Histogram_Equalization_DSIHE')
            os.makedirs(output_img_dir, exist_ok=True)
            plt.savefig(os.path.join(output_img_dir, filename))
            plt.close()

# Process all images in the specified folder
process_images('input_dataset', 'HE_output')
