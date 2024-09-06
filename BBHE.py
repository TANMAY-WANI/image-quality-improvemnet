import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def bi_histogram_equalization(img):
    mean = np.mean(img)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    # Split the bins for lower and upper parts
    lower_bins = bins[:int(mean)+1]
    upper_bins = bins[int(mean):]
    
    # Split histogram at the mean value
    lower_hist = hist[:int(mean)]
    upper_hist = hist[int(mean):]
    
    # Equalize each part separately
    cdf_lower = np.cumsum(lower_hist)
    cdf_upper = np.cumsum(upper_hist)
    
    # Normalize CDFs to the range [0, 255]
    cdf_lower = (cdf_lower - cdf_lower.min()) * 255 / (cdf_lower.max() - cdf_lower.min())
    cdf_upper = (cdf_upper - cdf_upper.min()) * 255 / (cdf_upper.max() - cdf_upper.min())
    
    img_equalized = img.copy()
    img_equalized[img < mean] = np.interp(img[img < mean], lower_bins[:-1], cdf_lower)
    img_equalized[img >= mean] = np.interp(img[img >= mean], upper_bins[:-1], cdf_upper + 128)
    
    return img_equalized

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, 0)  # Read image in grayscale
            
            equalized_img = bi_histogram_equalization(img)
            
            # Display and save the plot
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
            plt.subplot(1, 2, 2), plt.imshow(equalized_img, cmap='gray'), plt.title('BBHE Image')
            
            output_img_dir = os.path.join(output_folder, 'Bi_Histogram_Equalization_BBHE')
            os.makedirs(output_img_dir, exist_ok=True)
            plt.savefig(os.path.join(output_img_dir, filename))
            plt.close()

# Process all images in the specified folder
process_images('input_dataset', 'HE_output')
