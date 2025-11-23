import os
from PIL import Image

def resize_image(input_folder, output_folder, target_size=224):

    #Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #Get all image files
    image_files = os.listdir(input_folder)

    print(f"Resizing images to {target_size}x{target_size}")

    successful = 0
    failed = 0

    #Process each image

    for filename in image_files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                input_path = os.path.join(input_folder, filename)
                image = Image.open(input_path)

                resized_image = image.resize((target_size, target_size), Image.LANCZOS)

                output_path = os.path.join(output_folder, filename)

                resized_image.save(output_path)

                print(f"Resized: {filename}")
                successful += 1

            except Exception as e:
                print(f"Error with {filename}: {e}")
                failed += 1


def resize_mask(input_folder, output_folder, target_size):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mask_files = os.listdir(input_folder)

    print(f"Resizing masks to {target_size}x{target_size}")

    successful = 0
    failed = 0

    for filename in mask_files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                input_path = os.path.join(input_folder, filename)
                mask = Image.open(input_path)

                resized_mask = mask.resize((target_size, target_size), Image.NEAREST)

                output_path = os.path.join(output_folder, filename)

                resized_mask.save(output_path)

                print(f"Resized {filename}")
                successful += 1

            except Exception as e:
                print(f"Error with {filename}: {e}")
                failed += 1

if __name__ == '__main__':
    print('Resizing images...')
    resize_image(
        input_folder='data/ECSSD/images',
        output_folder='data/ECSSD/resized_images_128',
        target_size=128
    )

    print("\nResizing ground truth masks")

    resize_mask(
        input_folder='data/ECSSD/ground_truth_mask',
        output_folder='data/ECSSD/resized_masks_128',
        target_size=128
    )
