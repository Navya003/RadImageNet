# import necessary libraries
import tensorflow as tf
import numpy as np
import os
import cv2 as cv

def convert_fashion_mnist_to_dirs(output_base_dir="fashion_mnist_full_dataset"):
    """
    Downloads the Fashion-MNIST dataset and organizes it into
    'train' and 'test' directories, with subdirectories for each class.
    Each image is saved as a PNG file.
    """
    print(f"Starting Fashion-MNIST conversion to: {output_base_dir}")

    # Loading Fashion-MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Defining class names for Fashion-MNIST
    class_names = [
        'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # Creating base output directories
    train_dir = os.path.join(output_base_dir, 'train')
    test_dir = os.path.join(output_base_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Processing training data...")
    
    # Process training data
    for i, (image, label) in enumerate(zip(X_train, y_train)):
        class_name = class_names[label]
        class_output_dir = os.path.join(train_dir, class_name)
        # Ensure class directory exists
        os.makedirs(class_output_dir, exist_ok=True) 

        # Save image directly into the class directory
        # Fashion-MNIST images are grayscale (28x28). Resized to 128x128 and convert to 3 channels for consistency.
        image_resized = cv.resize(image, (128, 128)) 
        image_bgr = cv.cvtColor(image_resized, cv.COLOR_GRAY2BGR) # Convert to 3 channels (BGR for OpenCV)
        
        img_filename = os.path.join(class_output_dir, f"train_img_{i:05d}.png")
        cv.imwrite(img_filename, image_bgr)

        if (i + 1) % 10000 == 0:
            print(f"  Saved {i + 1} training images.")

    print(f"Finished saving {len(X_train)} training images.")

    print("Processing test data...")
    # Process test data
    for i, (image, label) in enumerate(zip(X_test, y_test)):
        class_name = class_names[label]
        class_output_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True) # Ensure class directory exists

        # Save image directly into the class directory
        image_resized = cv.resize(image, (128, 128)) # MODIFIED: Resized to 128x128
        image_bgr = cv.cvtColor(image_resized, cv.COLOR_GRAY2BGR) # Convert to 3 channels (BGR for OpenCV)

        img_filename = os.path.join(class_output_dir, f"test_img_{i:05d}.png")
        cv.imwrite(img_filename, image_bgr) # Save the image

        if (i + 1) % 2000 == 0:
            print(f"  Saved {i + 1} test images.")

    print(f"Finished saving {len(X_test)} test images.")
    print(f"Fashion-MNIST conversion complete. Data saved to '{output_base_dir}'")

if __name__ == "__main__":
    convert_fashion_mnist_to_dirs()
