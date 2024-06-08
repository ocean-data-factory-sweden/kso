from PIL import Image 
import os
import shutil

def compress_image(max_size_mb=0.9):
    input_path = input("Enter the path to the local directory of your unprocessed images: ")
    output_path = input("Enter the path to the local directory where you want to store the compressed images: ")
    counter = 0
    if not os.path.isdir(input_path):
        print(f"Error: Input directory '{input_path}' does not exist.")
        return
    
    # Get a list of all image files in the input path
    image_files = [f for f in os.listdir(input_path) if f.endswith('.JPG')]
    print("Image files found:", image_files)

    if not image_files:
        print(f"No image files found in '{input_path}'.")
        return

    for image_file in image_files:
        # Construct the full path of the image file
        image_path = os.path.join(input_path, image_file)
        
        # Open the image file
        image = Image.open(image_path)
        
        # Calculate the desired quality to achieve the maximum size in MB
        quality = 85  # Starting quality
        while True:
            # Save the image with the current quality
            output_image_path = os.path.join(output_path, image_file)
            image.save(output_image_path, quality=quality, optimize=True)
            
            # Check the file size of the saved image
            file_size_mb = os.path.getsize(output_image_path) / (1024 * 1024)

            if file_size_mb <= max_size_mb:
                break
            
            quality -= 5
            if quality <= 0:
                break
        counter += 1
        print(f"Image {image_file} compressed successfully. compressed images: {counter}")
    

    return output_path


compress_image(max_size_mb=0.9)


def create_testset():
    input_path = input("Enter the path to the local directory of your compressed images: ")
    testset_path = input("Enter the testset path: ")
    max_files = int(input("Enter the maximum number of files: ")) + 1

    # Create a list to store the names of the copied files
    copied_files = []

    # Empty the testset directory
    shutil.rmtree(testset_path)
    os.mkdir(testset_path)

    # Counter for the number of copied files
    num_copied_files = 0

    # Iterate over the files in the output_path directory
    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            if file_size < 1000000 and num_copied_files < max_files:
                # Copy the file to the testset directory
                shutil.copy(file_path, testset_path)
                # Add the file name to the copied_files list
                copied_files.append(file_name)
                # Increment the counter
                num_copied_files += 1
            elif num_copied_files >= max_files+1:
                print(f"{max_files} files added to your testset.")
                break

    # Create a file called testset.txt and write the names of the copied files
    with open(os.path.join(testset_path, 'testset.txt'), 'w') as file:
        file.write('\n'.join(copied_files))

create_testset()