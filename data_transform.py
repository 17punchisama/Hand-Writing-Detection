import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import imread, imshow, subplots, show, imsave
import glob
     
def read_img(img_path):
    image = imread(img_path)
    images = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return images

def plot_and_save(images, data_generator, output_path, transformation_name, num_augmented=2):
    augmented_images = data_generator.flow(images, batch_size=1)

    for i in range(num_augmented):
        img = augmented_images.__next__()[0].astype('uint8')  # Get the next augmented image
        augmented_filename = f"{os.path.splitext(os.path.basename(output_path))[0]}_{transformation_name}_{i}.jpg"
        augmented_full_path = os.path.join(os.path.dirname(output_path), augmented_filename)
        imsave(augmented_full_path, img)

# Augmentation functions
def apply_augmentation(img_path):
    images = read_img(img_path)

    # Define output path
    output_path = img_path  # Same as input path for saving augmented images

    # Apply augmentations
    rotation_generator = ImageDataGenerator(rotation_range=20)
    plot_and_save(images, rotation_generator, output_path, "rotation")

    shear_generator = ImageDataGenerator(shear_range=0.3)  # Adjust shear_range as needed
    plot_and_save(images, shear_generator, output_path, "shear")

    zoom_generator = ImageDataGenerator(zoom_range=[1.0, 1.5])
    plot_and_save(images, zoom_generator, output_path, "zoom")

# Process all images in the directory
def process_all_images_in_folder(folder_path):
    for subdir, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # You can add more extensions if needed
                img_path = os.path.join(subdir, filename)
                print(f"Processing: {img_path}")
                apply_augmentation(img_path)

# Example usage
if __name__ == "__main__":
    base_folder = 'data_for_valid'
    process_all_images_in_folder(base_folder)
