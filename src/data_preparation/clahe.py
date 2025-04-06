import numpy as np
import cv2 as cv
import os

# Set your folder path here
folder_path = "your_folder_path"  # Replace with your actual folder path
output_folder = os.path.join(folder_path, "processed")  # Subfolder for processed images

# Supported image extensions
supported_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Create CLAHE object
clahe = cv.createCLAHE(clipLimit=50.0, tileGridSize=(16, 16))

# Process all image files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(supported_extensions):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Read the image
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read {filename}, skipping...")
                continue
            
            # Apply CLAHE
            processed_img = clahe.apply(img)
            
            # Create output filename
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_clahe{ext}"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save the processed image
            cv.imwrite(output_path, processed_img)
            print(f"Processed: {filename} → saved as {output_filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("\nProcessing complete!")
print(f"Processed images saved in: {output_folder}")
