import tensorflow as tf
from tensorflow.keras import layers, models

def build_visual_model(num_classes, input_shape=(224, 224, 3), backbone="EfficientNetB0"):
    """
    Builds a standalone visual bird classification model based on the 
    Visual Bird AI (Chirp Imagery) approach.
    """
    # 1. Base Model (Transfer Learning)
    if backbone == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, 
            weights="imagenet", 
            input_shape=input_shape
        )
    elif backbone == "EfficientNetB3":
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False, 
            weights="imagenet", 
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Backbone {backbone} not supported.")

    # 2. Freeze base model layers initially (optional, can be fine-tuned later)
    base_model.trainable = True # Fine-tuning from the start is common in Kaggle

    # 3. Custom Head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax', name="species_output")
    ])

    return model

def get_optimizer(lr=1e-3):
    return tf.keras.optimizers.Adam(learning_rate=lr)

def get_loss():
    # Use CategoricalCrossentropy if labels are one-hot, 
    # or BinaryCrossentropy if multi-label.
    return tf.keras.losses.CategoricalCrossentropy()
