import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

class ModelEvaluator:
    def __init__(self, output_dir, image_paths_test, y_test, class_names, x_test_original, preprocess_input_fn, model_builder):
        self.output_dir = output_dir
        self.image_paths_test = image_paths_test 
        self.y_test = y_test # Store one-hot encoded labels (y_true_one_hot)
        self.class_names = class_names
        self.x_test_original = x_test_original
        self.preprocess_input_fn = preprocess_input_fn
        self.model_builder = model_builder 

        self.eval_results_dir = os.path.join(self.output_dir, 'evaluation_results')
        os.makedirs(self.eval_results_dir, exist_ok=True)
        self.input_shape = (128, 128, 3) # Set input shape once

    def _parse_image_function_eval(self, filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, self.input_shape[:2])
        img = self.preprocess_input_fn(img)
        return img, label

    def _plot_roc_curves(self, predictions, y_true_one_hot):
        """Calculates and plots ROC curves for all classes."""
        plt.figure(figsize=(10, 8))
        
        # Calculate FPR, TPR, and AUC for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(self.eval_results_dir, 'roc_curves.png')
        plt.savefig(roc_path)
        print(f"ROC curves saved to {roc_path}")
        plt.close()

    def evaluate(self):
        num_classes = self.y_test.shape[1]
        
        # Build and load model
        model, _ = self.model_builder.build_model(num_classes)
        model_path = os.path.join(self.output_dir, 'models', 'final_model.weights.h5')
        
        if not os.path.exists(model_path):
            print(f"Error: Final model weights not found at {model_path}. Cannot evaluate.")
            return

        model.load_weights(model_path)
        print(f"Loaded model weights from: {model_path}")

        # Data preparation
        test_filepaths = tf.constant(self.image_paths_test)
        test_labels = tf.constant(self.y_test, dtype=tf.float32)

        test_dataset = tf.data.Dataset.from_tensor_slices((test_filepaths, test_labels))
        test_dataset = test_dataset.map(self._parse_image_function_eval, num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        print("Predicting on test data...")
        predictions = model.predict(test_dataset)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.y_test, axis=1) # The true integer labels

        # --- 1. EXTRACT AND PRINT KEY METRICS ---
        print("\n--- Key Evaluation Metrics ---")
        overall_acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print("------------------------------------")


        # --- 2. CLASSIFICATION REPORT ---
        report = classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
        print("\nClassification Report:")
        print(report)

        report_path = os.path.join(self.eval_results_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
            f.write(f"\nOverall Accuracy: {overall_acc:.4f}")
        print(f"Classification report saved to {report_path}")

        # --- 3. CONFUSION MATRIX PLOT ---
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

        # --- 4. ROC and AUC Plot ---
        self._plot_roc_curves(predictions, self.y_test)

        # --- 5. Save raw predictions (optional) ---
        predictions_df = pd.DataFrame(predictions, columns=[f'prob_{name}' for name in self.class_names])
        predictions_df['true_label'] = [self.class_names[i] for i in y_true]
        predictions_df['predicted_label'] = [self.class_names[i] for i in y_pred]
        predictions_df.to_csv(os.path.join(self.eval_results_dir, 'predictions.csv'), index=False)
        print(f"Predictions saved to {os.path.join(self.eval_results_dir, 'predictions.csv')}")
