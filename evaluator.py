import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model # Still needed for generic model loading if full model was saved
# We will explicitly build the model and load weights now.
# from model_builder import ModelBuilder # This import might be needed if you create ModelBuilder inside Evaluator

class ModelEvaluator:
    # MODIFIED: Updated __init__ to accept model_builder and preprocess_input_fn
    def __init__(self, output_dir, x_test, y_test, test_categories, image_paths, preprocess_input_fn, model_builder_instance):
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, 'models')
        self.final_model_path = os.path.join(self.models_dir, 'final_model.h5') # This path now contains only weights
        self.x_test = x_test
        self.y_test = y_test
        self.test_categories = test_categories
        self.image_paths = image_paths
        self.preprocess_input_fn = preprocess_input_fn
        self.model_builder_instance = model_builder_instance # Store the ModelBuilder instance

    def evaluate(self):
        results = []

        if not os.path.exists(self.final_model_path):
            print(f"Final model weights not found at: {self.final_model_path}") # Adjusted message
            return

        try:
            print(f"Loading final model weights from: {self.final_model_path}")

            # MODIFIED: Re-build the model architecture
            # We need to pass num_classes to build_model. Assuming it's available in model_builder_instance
            # and that it was correctly initialized with the num_classes from training data.
            model, _ = self.model_builder_instance.build_model(self.model_builder_instance.num_classes)
            
            # Load the saved weights onto the newly built model
            model.load_weights(self.final_model_path)
            
            print(f"Model input shape: {model.input_shape}")
            print(f"x_test shape: {self.x_test.shape}")
            
            x_test_processed = self.preprocess_input_fn(self.x_test)

            # Ensure model is compiled for evaluation metrics (it should be from build_model, but good to ensure)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            print("Running model.evaluate on test data...")
            loss, accuracy = model.evaluate(x_test_processed, self.y_test, verbose=0)
            print(f"Test Loss: {loss:.3f}")
            print(f"Test Accuracy: {accuracy:.3f}")

            predictions = model.predict(x_test_processed)
            print(f"Prediction shape: {predictions.shape}")

            predicted_classes = np.argmax(predictions, axis=1)
            predicted_probabilities = np.max(predictions, axis=1)

        except Exception as e:
            print(f"Error loading or evaluating the final model: {str(e)}")
            print("Suggestion: To get more detailed error messages during model evaluation, try adding these lines at the beginning of your main.py:")
            print("import tensorflow as tf")
            print("tf.config.run_functions_eagerly(True)")
            print("tf.data.experimental.enable_debug_mode(True)")
            return

        for i in range(len(self.x_test)):
            if predicted_classes[i] < len(self.test_categories):
                predicted_label = self.test_categories[predicted_classes[i]]
            else:
                predicted_label = "Unknown"

            result = {
                "image_path": self.image_paths[i],
                "predicted_class": predicted_label,
                "probability": predicted_probabilities[i]
            }
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, 'test_predictions.csv'), index=False)
        print("Test image predictions saved to 'test_predictions.csv'.")
