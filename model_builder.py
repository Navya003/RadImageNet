from tensorflow.keras.applications import VGG16, VGG19, ResNet50, DenseNet121, EfficientNetB0 # ADDED EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf # Import tensorflow to access its applications.

class ModelBuilder:
    def __init__(self, base_model_name, input_shape, num_classes, lr, epochs):
        self.base_model_name = base_model_name
        self.input_shape = input_shape
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        
    def build_model(self, num_classes):
        base_model = None
        preprocess_input_fn = None # Initialize preprocess_input_fn

        if self.base_model_name == "VGG16":
            base_model = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
            preprocess_input_fn = tf.keras.applications.vgg16.preprocess_input # Get specific preprocess_input
        elif self.base_model_name == "VGG19":
            base_model = VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape)
            preprocess_input_fn = tf.keras.applications.vgg19.preprocess_input
        elif self.base_model_name == "ResNet50":
            base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)
            preprocess_input_fn = tf.keras.applications.resnet50.preprocess_input
        elif self.base_model_name == "DenseNet121":
            base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=self.input_shape)
            preprocess_input_fn = tf.keras.applications.densenet.preprocess_input
        elif self.base_model_name == "EfficientNetB0": # ADDED EfficientNetB0 case
            # EfficientNet models typically expect 224x224 for B0, but can be scaled.
            # Ensure your input_shape matches the expected input for the chosen EfficientNet variant.
            # The default input_shape for EfficientNetB0 is (224, 224, 3)
            base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=self.input_shape)
            preprocess_input_fn = tf.keras.applications.efficientnet.preprocess_input
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")

        if preprocess_input_fn is None:
            # Fallback for models without specific preprocess_input, or if you want a generic one
            # For this pipeline, it's crucial that it's set.
            print(f"Warning: No specific preprocess_input function found for {self.base_model_name}. Using generic /255.0 normalization.")
            preprocess_input_fn = lambda x: x / 255.0 # Default to [0,1] normalization

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=output)

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        opt = Adam(learning_rate=self.lr, decay=self.lr/self.epochs)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model, preprocess_input_fn # MODIFIED: Return preprocess_input_fn

