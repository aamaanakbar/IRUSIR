import os
import shutil

# Source directory containing the 100 folders
source_dir = "/home/akbar/Documents/All_codes/Updated_version/robust-unsupervised/out/restored_samples/2024-09-04T161912/composed_tasks/UNP/3/inversions"
# Destination directory where you want to copy and rename the files
destination_dir = "/home/akbar/Documents/All_codes/Updated_version/robust-unsupervised/out/restored_samples/2024-09-04T161912/composed_tasks/UNP/3/inversions/UNP_predw++"

# Ensure the destination directory exists
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Initialize a counter for serial numbering, starting from 0
counter = 0

# Get a sorted list of folder names
folders = sorted(os.listdir(source_dir))

# Iterate over each folder in the sorted list
for folder_name in folders:
    folder_path = os.path.join(source_dir, folder_name)
    
    # Check if it is a directory
    if os.path.isdir(folder_path):
        # Define the path to the file with the common name
        file_path = os.path.join(folder_path, "pred_W++.png")  # Replace with actual file name and extension
        # Check if the file exists in the current folder
        if os.path.exists(file_path):
            # Create a new file name with a serial number, starting from 00
            new_file_name = f"{counter:02d}_pred_W++.png"  # e.g., 00_common_file_name.extension
            new_file_path = os.path.join(destination_dir, new_file_name)

            # Copy the file to the destination directory with the new name
            shutil.copy(file_path, new_file_path)
            print(f"Copied: {file_path} to {new_file_path}")

            # Increment the counter
            counter += 1
