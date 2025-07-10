from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import KFold
# Removed ImageDataGenerator as tf.data.Dataset will handle augmentation
import pandas as pd
import os
import numpy as np
import time
import tensorflow as tf # Import TensorFlow

class ModelTrainer:
    # MODIFIED: x_train is now image_paths_train (list of strings)
    def __init__(self, model, image_paths_train, y_train, batch_size, epochs, output_dir, preprocess_input_fn):
        self.model = model
        self.image_paths_train = image_paths_train # Store image paths
        self.y_train = y_train # Store one-hot encoded labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = output_dir
        self.preprocess_input_fn = preprocess_input_fn
        self.acc_per_fold = []
        self.loss_per_fold = []

        self.models_dir = os.path.join(self.output_dir, 'models')
        self.tensorboard_dir = os.path.join(self.output_dir, 'tensorboard_logs')
        self.csv_logs_dir = os.path.join(self.output_dir, 'csv_logs')
        self.create_subdirectories()

    def create_subdirectories(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.csv_logs_dir, exist_ok=True)

    # Helper function to load, preprocess, and augment images for tf.data.Dataset
    def _parse_image_function(self, filepath, label):
        # Load the raw image bytes
        img = tf.io.read_file(filepath)
        # Decode to tensor
        img = tf.image.decode_image(img, channels=3, expand_animations=False) # Ensure 3 channels
        # Resize
        img = tf.image.resize(img, (224, 224))
        # Apply model-specific preprocessing (e.g., EfficientNet's preprocess_input)
        img = self.preprocess_input_fn(img)
        return img, label

    # Helper function for data augmentation (can be integrated into _parse_image_function or kept separate)
    def _augment_image(self, image, label):
        # Apply augmentations similar to ImageDataGenerator
        image = tf.image.random_flip_left_right(image)
        # Add other augmentations as needed (e.g., random rotation, brightness, contrast)
        # For rotation, you might need to use tf.keras.preprocessing.image.random_rotation
        # or implement it manually with tf.image.rotate.
        # For simplicity, we'll keep it basic for now.
        return image, label

    def get_callbacks(self, fold, use_validation_callbacks=True):
        callbacks_list = []

        csv_logger = CSVLogger(os.path.join(self.csv_logs_dir, f'training_log_{fold}.csv'))
        tensor_board = TensorBoard(log_dir=os.path.join(self.tensorboard_dir, f'tensorboard_logs_{fold}'))

        callbacks_list.append(csv_logger)
        callbacks_list.append(tensor_board)

        if use_validation_callbacks:
            # MODIFIED: Changed filepath extension to .weights.h5
            mc_path = os.path.join(self.models_dir, f'model_{fold}.weights.h5')
            model_checkpoint = ModelCheckpoint(filepath=mc_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1, save_weights_only=True)
            early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

            callbacks_list.append(model_checkpoint)
            callbacks_list.append(early_stop)
            callbacks_list.append(reduce_lr)

        return callbacks_list

    def cross_validate(self, n_folds=3):
        acc_per_fold, loss_per_fold, val_acc_per_fold, val_loss_per_fold, time_per_fold = [], [], [], [], []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        total_time = 0

        # Convert image_paths_train and y_train to TensorFlow Tensors for tf.data
        image_paths_tensor = tf.constant(self.image_paths_train)
        y_train_tensor = tf.constant(self.y_train, dtype=tf.float32) # Ensure correct dtype for labels

        for fold_no, (tr_idx, val_idx) in enumerate(kf.split(self.image_paths_train, self.y_train), 1): # Use image_paths_train for splitting
            print(f'Training fold {fold_no}...')

            start_time = time.time()

            # Create train and validation datasets using tf.data.Dataset
            train_filepaths = tf.gather(image_paths_tensor, tr_idx)
            train_labels = tf.gather(y_train_tensor, tr_idx)
            val_filepaths = tf.gather(image_paths_tensor, val_idx)
            val_labels = tf.gather(y_train_tensor, val_idx)

            # Build tf.data.Dataset for training
            train_dataset = tf.data.Dataset.from_tensor_slices((train_filepaths, train_labels))
            train_dataset = train_dataset.map(self._parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
            train_dataset = train_dataset.map(self._augment_image, num_parallel_calls=tf.data.AUTOTUNE) # Apply augmentation
            train_dataset = train_dataset.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

            # Build tf.data.Dataset for validation
            val_dataset = tf.data.Dataset.from_tensor_slices((val_filepaths, val_labels))
            val_dataset = val_dataset.map(self._parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
            val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE) # No augmentation for validation

            history = self.model.fit(train_dataset,
                                     validation_data=val_dataset,
                                     epochs=self.epochs,
                                     verbose=1, callbacks=self.get_callbacks(fold_no, use_validation_callbacks=True))

            acc_per_fold.append(history.history['accuracy'][-1] * 100)
            loss_per_fold.append(history.history['loss'][-1])
            val_acc_per_fold.append(history.history['val_accuracy'][-1]*100)
            val_loss_per_fold.append(history.history['val_loss'][-1])

            end_time = time.time()
            fold_time = end_time - start_time
            total_time += fold_time
            time_per_fold.append(fold_time)

            print(f"Time taken for fold {fold_no}: {fold_time:.3f} seconds")
        print(f"Total training time for {n_folds} folds: {total_time:.3f} seconds")

        self.save_cv_results(acc_per_fold, loss_per_fold, val_acc_per_fold, val_loss_per_fold, time_per_fold, total_time)
        return np.mean(acc_per_fold), np.mean(loss_per_fold), time_per_fold

    def save_cv_results(self, acc_per_fold, loss_per_fold, val_acc_per_fold, val_loss_per_fold, time_per_fold, total_time):
        results_df = pd.DataFrame({
            'Fold': [i + 1 for i in range(len(acc_per_fold))],
            'Training Accuracy': acc_per_fold,
            'Training Loss': loss_per_fold,
            'Validation Accuracy': val_acc_per_fold,
            'Validation Loss': val_loss_per_fold,
            'Mean_acc': np.mean(acc_per_fold),
            'Mean_loss': np.mean(loss_per_fold),
            'Time (seconds)': time_per_fold
        })
        results_df.to_csv(os.path.join(self.output_dir, 'cv_results.csv'), index=False)

        with open(os.path.join(self.output_dir, 'training_time.txt'), 'w') as f:
            f.write(f'Total training time: {total_time:.3f} seconds\n')

    def train_on_entire_dataset(self):
        print("Training the model on the entire dataset...")

        callbacks = self.get_callbacks(fold='entire_dataset', use_validation_callbacks=False)

        # Build tf.data.Dataset for the entire training set
        image_paths_tensor = tf.constant(self.image_paths_train)
        y_train_tensor = tf.constant(self.y_train, dtype=tf.float32)

        train_dataset_full = tf.data.Dataset.from_tensor_slices((image_paths_tensor, y_train_tensor))
        train_dataset_full = train_dataset_full.map(self._parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset_full = train_dataset_full.map(self._augment_image, num_parallel_calls=tf.data.AUTOTUNE) # Apply augmentation
        train_dataset_full = train_dataset_full.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        start_time = time.time()

        history = self.model.fit(train_dataset_full,
                                 epochs=self.epochs,
                                 verbose=1, callbacks=callbacks)

        end_time = time.time()
        total_time = end_time - start_time

        final_acc = history.history['accuracy'][-1] * 100
        final_loss = history.history['loss'][-1]

        print(f"Time taken to train on the entire dataset: {total_time:.3f} seconds")
        print(f"Final training accuracy on the entire dataset: {final_acc:.3f}%")
        print(f"Final training loss on the entire dataset: {final_loss:.3f}")

        # MODIFIED: Changed final model path to .weights.h5
        model_path = os.path.join(self.models_dir, 'final_model.weights.h5')
        self.model.save_weights(model_path)
        print(f"Final model weights saved at: {model_path}")

        self.save_entire_dataset_results(final_acc, final_loss, total_time)

    def save_entire_dataset_results(self, final_acc, final_loss, total_time):
        results_df = pd.DataFrame({"Final Accuracy (%)": [final_acc], "Final Loss": [final_loss], "Training Time (seconds)": [total_time]})

        results_df.to_csv(os.path.join(self.output_dir, 'entire_dataset_results.csv'), index=False)
        with open(os.path.join(self.output_dir, 'entire_training_time.txt'), 'w') as f:
            f.write(f"Final Accuracy: {final_acc:.3f}%\n")
            f.write(f"Final Loss: {final_loss:.3f}\n")
            f.write(f"Total training time: {total_time:.3f} seconds\n")
