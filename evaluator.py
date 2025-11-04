import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf # Import TensorFlow
from tensorflow.keras.models import load_model # Assuming you'd load a model

class ModelEvaluator:
    def __init__(self, output_dir, image_paths_test, y_test, class_names, x_test_original, preprocess_input_fn, model_builder):
        self.output_dir = output_dir
        self.image_paths_test = image_paths_test # Store image paths
        self.y_test = y_test # Store one-hot encoded labels
        self.class_names = class_names
        self.x_test_original = x_test_original # This seems redundant with image_paths_test
        self.preprocess_input_fn = preprocess_input_fn
        self.model_builder = model_builder # Store model_builder instance

        self.eval_results_dir = os.path.join(self.output_dir, 'evaluation_results')
        os.makedirs(self.eval_results_dir, exist_ok=True)

    def _parse_image_function_eval(self, filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (128, 128)) # MODIFIED: Changed to 128x128
        img = self.preprocess_input_fn(img)
        return img, label

    def evaluate(self):
        # Build the model architecture (without training it) to load weights
        # We need to know the input shape and number of classes
        input_shape = (128, 128, 3) # MODIFIED: Set input shape for evaluation model
        num_classes = self.y_test.shape[1]
        
        # Build an untrained model with the correct architecture to load weights into
        # We pass dummy lr and epochs, as they are not used for just building the model architecture for evaluation
        model, _ = self.model_builder.build_model(num_classes)
        
        # Load the best weights saved during training
        model_path = os.path.join(self.output_dir, 'models', 'final_model.weights.h5')
        if not os.path.exists(model_path):
            print(f"Error: Final model weights not found at {model_path}. Cannot evaluate.")
            return

        model.load_weights(model_path)
        print(f"Loaded model weights from: {model_path}")

        # Create tf.data.Dataset for test data
        test_filepaths = tf.constant(self.image_paths_test)
        test_labels = tf.constant(self.y_test, dtype=tf.float32)

        test_dataset = tf.data.Dataset.from_tensor_slices((test_filepaths, test_labels))
        test_dataset = test_dataset.map(self._parse_image_function_eval, num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE) # Using a default batch size for evaluation

        print("Predicting on test data...")
        predictions = model.predict(test_dataset)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        print("\nClassification Report:")
        print(report)

        report_path = os.path.join(self.eval_results_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(self.eval_results_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        plt.close()

        # Save raw predictions (optional)
        predictions_df = pd.DataFrame(predictions, columns=[f'prob_{name}' for name in self.class_names])
        predictions_df['true_label'] = [self.class_names[i] for i in y_true]
        predictions_df['predicted_label'] = [self.class_names[i] for i in y_pred]
        predictions_df.to_csv(os.path.join(self.eval_results_dir, 'predictions.csv'), index=False)
        print(f"Predictions saved to {os.path.join(self.eval_results_dir, 'predictions.csv')}")
