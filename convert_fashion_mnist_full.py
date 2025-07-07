import tensorflow as tf
import numpy as np
import os
import cv2 # Using OpenCV to save images easily

def convert_fashion_mnist_to_dirs_full(output_base_dir="fashion_mnist_full_dataset"):
    """
    Downloads the full Fashion-MNIST dataset, processes it, and saves all images
    into a directory structure suitable for the ImageAnalysis pipeline.
    """
    print("Downloading Fashion-MNIST dataset...")
    # Load Fashion-MNIST directly from Keras datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print("Fashion-MNIST dataset downloaded.")

    # Define class names for Fashion-MNIST (10 classes)
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

    # --- Process and save ALL training images ---
    print(f"Processing {len(x_train)} training images...")
    for i, (image, label) in enumerate(zip(x_train, y_train)):
        class_name = class_names[label]
        class_dir = os.path.join(train_output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Fashion-MNIST images are 28x28 grayscale.
        # Resize to 224x224 and convert to 3 channels (RGB) by stacking.
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_rgb = np.stack([image_resized, image_resized, image_resized], axis=-1) # Convert to 3 channels

        image_path = os.path.join(class_dir, f"train_img_{i:05d}.png")
        cv2.imwrite(image_path, image_rgb) # Save as PNG

        if (i + 1) % 10000 == 0:
            print(f"  Saved {i + 1} training images.")

    print(f"Finished saving {len(x_train)} training images.")

    # --- Process and save ALL test images ---
    print(f"Processing {len(x_test)} test images...")
    for i, (image, label) in enumerate(zip(x_test, y_test)):
        class_name = class_names[label]
        class_dir = os.path.join(test_output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Resize and convert to 3 channels (RGB)
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_rgb = np.stack([image_resized, image_resized, image_resized], axis=-1)

        image_path = os.path.join(class_dir, f"test_img_{i:05d}.png")
        cv2.imwrite(image_path, image_rgb)

        if (i + 1) % 2000 == 0:
            print(f"  Saved {i + 1} test images.")

    print(f"Finished saving {len(x_test)} test images.")
    print(f"Fashion-MNIST conversion complete. Data saved to '{output_base_dir}'")
    print(f"You can now use '{output_base_dir}/train' and '{output_base_dir}/test' as input directories.")

if __name__ == "__main__":
    # Make sure you have tensorflow and opencv-python installed in your environment
    # conda install tensorflow py-opencv
    convert_fashion_mnist_to_dirs_full()

