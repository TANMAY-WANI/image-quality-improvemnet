import cv2
import os

def histogram_equalization(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            # Construct full file path
            img_path = os.path.join(input_folder, filename)
        
            img = cv2.imread(img_path, 0)

            equ_img = cv2.equalizeHist(img)
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, equ_img)
            print(f"Processed and saved: {output_path}")

input_folder = 'input_dataset' 
output_folder = 'output_dataset'       

histogram_equalization(input_folder, output_folder)
