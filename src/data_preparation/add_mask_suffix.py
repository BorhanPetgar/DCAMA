# Add _mask to the end of the each file in the folder
import os
def add_mask_suffix(folder_path):
    """Add '_mask' suffix to all files in the folder."""
    for filename in os.listdir(folder_path):
        print(filename)
        if not filename.endswith('_mask.png'):
            new_filename = os.path.splitext(filename)[0] + '_mask.png'
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            
if __name__ == "__main__":
    folder_path = '/home/borhan/Documents/code_workspace/vm6_backup/projects/anomaly/repos/DCAMA/ALL/crack_dataset_custom/test_masks'  # Change this to your folder path
    add_mask_suffix(folder_path)
    print(f"Added '_mask' suffix to all files in {folder_path}")