import tensorflow as tf
from tensorflow.keras import layers, models

def build_fusion_model(num_classes, 
                       img_shape=(224, 224, 3), 
                       perch_dim=1280, 
                       metadata_dim=4,
                       backbone="EfficientNetB0"):
    """
    Builds a multi-modal fusion model.
    Inputs:
    1. Spectrogram Image (Branch B)
    2. Perch Embedding (Branch A)
    3. Metadata [lat, lon, month, day_of_year] (Branch C)
    """

    # --- Branch B: Visual Branch (EfficientNet) ---
    input_img = layers.Input(shape=img_shape, name="input_spectrogram")
    
    if backbone == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=img_shape
        )
    elif backbone == "EfficientNetB3":
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False, weights="imagenet", input_shape=img_shape
        )
    else:
        raise ValueError(f"Backbone {backbone} not supported.")
    
    # We want to use the features from the visual backbone
    x_visual = base_model(input_img)
    x_visual = layers.GlobalAveragePooling2D()(x_visual)
    x_visual = layers.BatchNormalization()(x_visual)

    # --- Branch A: Perch Branch ---
    input_perch = layers.Input(shape=(perch_dim,), name="input_perch_embedding")
    x_perch = layers.Dense(512, activation='relu')(input_perch)
    x_perch = layers.Dropout(0.3)(x_perch)

    # --- Branch C: Metadata Branch ---
    input_meta = layers.Input(shape=(metadata_dim,), name="input_metadata")
    x_meta = layers.Dense(32, activation='relu')(input_meta)
    x_meta = layers.Dense(32, activation='relu')(x_meta)

    # --- Fusion Layer ---
    concat = layers.Concatenate()([x_visual, x_perch, x_meta])
    
    # --- Classification Head ---
    x = layers.Dense(1024, activation='relu')(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(num_classes, activation='softmax', name="species_output")(x)

    model = models.Model(
        inputs=[input_img, input_perch, input_meta], 
        outputs=output,
        name="BirdCLEF_Fusion_Model"
    )

    return model

if __name__ == "__main__":
    # Test building the model
    model = build_fusion_model(num_classes=264)
    model.summary()
