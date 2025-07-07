import os
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class DataLoader:
    def __init__(self, directory):
        self.directory = directory

    def load_train_data(self):
        """
        Load training data from subdirectories where each subdirectory is a label (category).
        """
        data = []
        labels = []
        
        categories = os.listdir(self.directory)
        for category in categories:
            path = os.path.join(self.directory, category)
            if os.path.isdir(path):
                for img in os.listdir(path):
                    img_path = os.path.join(path, img)
                    if os.path.isfile(img_path):
                        im = self.preprocess_images(img_path)
                        if im is not None:
                            data.append(im)
                            labels.append(category)
                            
        labels = self.encode_labels(labels)
        return np.array(data, dtype="float32"), labels, categories

    def load_test_data(self):
        """
        Load test data from subdirectories where each subdirectory is a label (category).
        It will return data, one-hot encoded labels, category names, AND image paths.
        """
        data = []
        labels = []
        image_paths = [] # This is already here, just ensure it's returned

        categories = os.listdir(self.directory)
        print(f"DataLoader (test): Found categories: {categories} in {self.directory}")
        for category in categories:
            path = os.path.join(self.directory, category)
            if os.path.isdir(path):
                print(f"DataLoader (test): Processing category: {category} at {path}")
                for img in os.listdir(path):
                    img_path = os.path.join(path, img)
                    if os.path.isfile(img_path):
                        im = self.preprocess_images(img_path)
                        if im is not None:
                            data.append(im)
                            labels.append(category)
                            image_paths.append(img_path) # Append path here

        lb = LabelBinarizer()
        encoded_labels = lb.fit_transform(labels)
        
        print(f"DataLoader (test): Final data count: {len(data)}")
        print(f"DataLoader (test): Final labels count: {len(labels)}")
        
        # CORRECTED: Added image_paths to the return statement
        return np.array(data, dtype="float32"), encoded_labels, lb.classes_.tolist(), image_paths 

    def preprocess_images(self, img_path, img_size=(224, 224)):
        """
        Preprocess a single image: load, resize, and normalize.
        Returns None if image cannot be loaded.
        """
        im = cv.imread(img_path)
        if im is None:
            print(f"WARNING: preprocess_images: Could not load image from {img_path}")
            return None
        im = cv.resize(im, img_size)
        im = np.array(im) / 255.0
        return im

    def encode_labels(self, labels):
        """
        Convert text labels to one-hot encoded format.
        """
        lb = LabelBinarizer()
        encoded_labels = lb.fit_transform(labels)
        return encoded_labels

