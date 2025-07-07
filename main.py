import argparse
from pipeline import Pipeline
import tensorflow as tf

print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Number of GPUs available: {len(physical_devices)}")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs found, using CPU.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run Image Classification Pipeline')

    # MODIFIED: Added 'EfficientNetB0' to choices
    parser.add_argument('--model_name', type=str, required=True, choices=['VGG16', 'VGG19', 'DenseNet121', 'ResNet50', 'EfficientNetB0'], help="Pre-trained model to use")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the training data directory")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to the test data directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory for saving models and results")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--batch_size', type=int,  help="Batch size for training")
    parser.add_argument('--epochs', type=int, help="Number of epochs for training")

    args = parser.parse_args()

    pipeline = Pipeline(
        model_name=args.model_name,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    pipeline.run()
