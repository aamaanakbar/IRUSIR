from PIL import Image
import os

# Set the paths to your source and destination folders
source_folder = '/home/akbar/Documents/All_codes/MASTAN BHAI RELATED/IRUSIR/Updated_version/robust-unsupervised/datasets/ffhq_65'
destination_folder = '/home/akbar/Documents/All_codes/MASTAN BHAI RELATED/IRUSIR/Updated_version/robust-unsupervised/datasets/ffhq_65_resize'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Resize function
def resize_image(image_path, output_path, size=(1024, 1024)):
    with Image.open(image_path) as img:
        img_resized = img.resize(size, Image.LANCZOS)
        img_resized.save(output_path)

# Process each image in the source folder
for filename in os.listdir(source_folder):
    # Create full file paths
    input_path = os.path.join(source_folder, filename)
    output_path = os.path.join(destination_folder, filename)
    
    # Check if the file is an image
    try:
        with Image.open(input_path) as img:
            # Resize and save the image
            resize_image(input_path, output_path)
            print(f'Resized and saved {filename}')
    except IOError:
        print(f'Skipped {filename} (not an image file)')

print('All images have been resized.')
