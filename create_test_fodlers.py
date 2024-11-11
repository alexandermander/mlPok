import os
from PIL import Image

def get_folders(path):
    # Get all folders in the specified path
    folders = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folders.append(folder)
    return folders

def create_test_folders(path):
    folders = get_folders(path)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        test_folder = os.path.join(folder_path, 'test')
        
        # Create 'test' folder if it doesn't exist
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        
        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.jpg', '.jpeg', '.webp', '.png'))]
        
        # Convert WEBP images to JPG and save them in the original folder
        for image in images:
            image_path = os.path.join(folder_path, image)
            if os.path.isfile(image_path):  # Make sure it's a file, not a folder
                with Image.open(image_path) as img:
                    if img.format == 'WEBP':
                        img = img.convert('RGB')
                        new_image_path = image_path.replace('.webp', '.jpg')
                        img.save(new_image_path, 'JPEG')
                        os.remove(image_path)  # Remove the original WEBP file

        # Refresh the image list after conversion to include only JPG images
        images = [img for img in os.listdir(folder_path) if img.lower().endswith('.jpg')]
        
        # Move the first 10 images to the 'test' folder
        for i in range(min(10, len(images))):  # Ensure there are at least 10 images
            image = images[i]
            image_path = os.path.join(folder_path, image)
            test_image_path = os.path.join(test_folder, image)
            os.rename(image_path, test_image_path)  # Move image to the 'test' folder

# Define the path to the 'Stellar_Crown' folder
path = './Pokemons/Stellar_Crown'
folders = get_folders(path)
create_test_folders(path)
print(folders)

