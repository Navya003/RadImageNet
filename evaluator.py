import numpy as np
import pandas as pd
import os
from keras.models import load_model

class ModelEvaluator:
    def __init__(self, output_dir, x_test, image_paths):
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, 'models')
        self.final_model_path = os.path.join(self.models_dir, 'final_model.h5')  # Path to the final model
        self.x_test = x_test
        self.image_paths = image_paths

    def evaluate(self, class_names):
        results = []  # Store predictions and probabilities for each image

        # Check if the final model exists
        if not os.path.exists(self.final_model_path):
            print(f"Final model not found at: {self.final_model_path}")
            return

        # Load the final model
        try:
            print(f"Loading final model: {self.final_model_path}")
            model = load_model(self.final_model_path)
            print(f"Model input shape: {model.input_shape}")
            print(f"x_test shape: {self.x_test.shape}")

            # Predict on the test data
            predictions = model.predict(self.x_test)
            print(f"Prediction shape: {predictions.shape}")

        except Exception as e:
            print(f"Error loading or evaluating the final model: {str(e)}")
            return

        # Get predicted classes and probabilities for each image
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_probabilities = np.max(predictions, axis=1)

        # Store the results in a DataFrame
        for i, image_path in enumerate(self.image_paths):
            predicted_label = class_names[predicted_classes[i]]
            result = {
                "image_path": image_path,
                "predicted_class": predicted_label,
                "probability": predicted_probabilities[i]
            }
            results.append(result)

        # Save the results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, 'test_predictions.csv'), index=False)
        print("Test image predictions saved to 'test_predictions.csv'.")
