import os
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf # Import TensorFlow

class DataLoader:
    def __init__(self, directory):
        self.directory = directory

    def load_train_data(self):
        """
        Load training data paths and labels, returning them as lists.
        The actual image loading and preprocessing will be done later in the tf.data pipeline.
        """
        image_paths = []
        labels = []

        categories = os.listdir(self.directory)
        print(f"DataLoader (train): Found categories in directory: {categories}")
        for category in categories:
            path = os.path.join(self.directory, category)
            if os.path.isdir(path):
                print(f"DataLoader (train): Processing category: {category} at {path}")
                for img_name in os.listdir(path):
                    img_path = os.path.join(path, img_name)
                    if os.path.isfile(img_path):
                        # We only collect paths and labels here.
                        # Preprocessing (cv2.imread, resize, normalize) will be done in the tf.data map function.
                        image_paths.append(img_path)
                        labels.append(category)

        lb = LabelBinarizer()
        encoded_labels = lb.fit_transform(labels)

        # Ensure all 10 classes are present in the LabelBinarizer's classes_
        # This is a critical check for the 9-class issue.
        if len(lb.classes_) != 10:
            print(f"WARNING: LabelBinarizer detected {len(lb.classes_)} unique classes instead of 10 for training data.")
            print(f"Detected classes: {lb.classes_}")
            # You might want to raise an error or handle this more robustly if it's a common issue.

        print(f"DataLoader (train): Total images found: {len(image_paths)}")
        print(f"DataLoader (train): Total labels found: {len(labels)}")
        print(f"DataLoader (train): Number of unique labels after encoding: {encoded_labels.shape[1]}")

        # Return paths, encoded labels, and category names.
        # These will be used to create a tf.data.Dataset in the trainer.
        return image_paths, encoded_labels, lb.classes_.tolist()

    def load_test_data(self):
        """
        Load test data paths and labels, returning them as lists.
        The actual image loading and preprocessing will be done later in the tf.data pipeline.
        """
        image_paths = []
        labels = []

        categories = os.listdir(self.directory)
        print(f"DataLoader (test): Found categories: {categories} in {self.directory}")
        for category in categories:
            path = os.path.join(self.directory, category)
            if os.path.isdir(path):
                print(f"DataLoader (test): Processing category: {category} at {path}")
                for img_name in os.listdir(path):
                    img_path = os.path.join(path, img_name)
                    if os.path.isfile(img_path):
                        image_paths.append(img_path)
                        labels.append(category)

        lb = LabelBinarizer()
        encoded_labels = lb.fit_transform(labels)

        if len(lb.classes_) != 10:
            print(f"WARNING: LabelBinarizer detected {len(lb.classes_)} unique classes instead of 10 for test data.")
            print(f"Detected classes: {lb.classes_}")

        print(f"DataLoader (test): Total images found: {len(image_paths)}")
        print(f"DataLoader (test): Total labels found: {len(labels)}")
        print(f"DataLoader (test): Number of unique labels after encoding: {encoded_labels.shape[1]}")

        # Return paths, encoded labels, and category names.
        # These will be used to create a tf.data.Dataset in the evaluator.
        return image_paths, encoded_labels, lb.classes_.tolist()

    # The preprocess_images method will now be used as a helper function within the tf.data map.
    # It will be defined in trainer.py and evaluator.py to include model-specific preprocessing.
    # So, we remove it from DataLoader as it's no longer responsible for the pixel-level preprocessing.
    # The DataLoader's role is just to find the file paths and their corresponding labels.
    pass # This class now only has __init__ and load_data methods that return paths/labels.

    # Removed preprocess_images and encode_labels as they are handled differently now.
    # encode_labels logic is kept inline in load_data methods.
