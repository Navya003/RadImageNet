import tensorflow as tf
import numpy as np
import os
import cv2 # Using OpenCV to save images easily

def convert_fashion_mnist_to_dirs_limited(
    output_base_dir="fashion_mnist_small_dataset",
    num_train_per_class=10000,  # Number of training images to save per class
    num_test_per_class=2000     # Number of test images to save per class
):
    """
    Downloads Fashion-MNIST, processes it, and saves a limited number of images
    per class into a directory structure suitable for the ImageAnalysis pipeline.
    """
    print("Downloading Fashion-MNIST dataset...")
    # Load Fashion-MNIST directly from Keras datasets
    (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.fashion_mnist.load_data()
    print("Fashion-MNIST dataset downloaded.")

    # Define class names for Fashion-MNIST
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # Create output directories
    train_output_dir = os.path.join(output_base_dir, "train")
    test_output_dir = os.path.join(output_base_dir, "test")

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    print(f"Creating directories in: {output_base_dir}")

    # --- Process and save training images (limited per class) ---
    print(f"Processing {num_train_per_class} training images per class...")
    train_counts = {name: 0 for name in class_names}
    total_train_saved = 0

    for i in range(len(x_train_full)):
        image, label = x_train_full[i], y_train_full[i]
        class_name = class_names[label]

        if train_counts[class_name] < num_train_per_class:
            class_dir = os.path.join(train_output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Fashion-MNIST images are 28x28 grayscale.
            # We need to resize to 224x224 and convert to 3 channels (RGB)
            # by stacking the grayscale channel 3 times.
            image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image_rgb = np.stack([image_resized, image_resized, image_resized], axis=-1)

            image_path = os.path.join(class_dir, f"train_img_{i:05d}.png")
            cv2.imwrite(image_path, image_rgb)

            train_counts[class_name] += 1
            total_train_saved += 1

        # Check if we have enough images for all classes
        if all(count >= num_train_per_class for count in train_counts.values()):
            break # Stop processing if all classes have enough images

    print(f"Finished saving {total_train_saved} training images in total.")

    # --- Process and save test images (limited per class) ---
    print(f"Processing {num_test_per_class} test images per class...")
    test_counts = {name: 0 for name in class_names}
    total_test_saved = 0

    for i in range(len(x_test_full)):
        image, label = x_test_full[i], y_test_full[i]
        class_name = class_names[label]

        if test_counts[class_name] < num_test_per_class:
            class_dir = os.path.join(test_output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Resize and convert to 3 channels (RGB)
            image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image_rgb = np.stack([image_resized, image_resized, image_resized], axis=-1)

            image_path = os.path.join(class_dir, f"test_img_{i:05d}.png")
            cv2.imwrite(image_path, image_rgb)

            test_counts[class_name] += 1
            total_test_saved += 1
        
        # Check if we have enough images for all classes
        if all(count >= num_test_per_class for count in test_counts.values()):
            break # Stop processing if all classes have enough images

    print(f"Finished saving {total_test_saved} test images in total.")
    print(f"Fashion-MNIST conversion complete. Data saved to '{output_base_dir}'")
    print(f"You can now use '{output_base_dir}/train' and '{output_base_dir}/test' as input directories.")

if __name__ == "__main__":
    # Make sure you have tensorflow and opencv-python installed in your environment
    # conda install tensorflow py-opencv
    
    # You can adjust these numbers to control the size of your subset
    convert_fashion_mnist_to_dirs_limited(
        num_train_per_class=100, # e.g., 100 images per class for training (1000 total for 10 classes)
        num_test_per_class=20    # e.g., 20 images per class for testing (200 total for 10 classes)
    )
