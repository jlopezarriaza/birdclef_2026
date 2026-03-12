import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.audio.spectrograms import SpectrogramGenerator
from src.models.visual_model import build_visual_model

class VisualDataGenerator(tf.keras.utils.Sequence):
    """
    On-the-fly spectrogram generator for training.
    """
    def __init__(self, df, audio_dir, labels, batch_size=32, img_size=224, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.labels = labels # List of all unique species
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.spec_gen = SpectrogramGenerator(img_size=img_size)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_df = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        X = []
        y = []
        
        for _, row in batch_df.iterrows():
            audio_path = os.path.join(self.audio_dir, row['filename'])
            img = self.spec_gen.generate(audio_path)
            
            if img is not None:
                X.append(img / 255.0) # Rescale to [0, 1]
                # One-hot encoding
                label_vec = np.zeros(len(self.labels))
                label_vec[self.labels.index(row['primary_label'])] = 1.0
                y.append(label_vec)
        
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

def train_standalone_visual(limit=None):
    # 1. Setup Data
    train_csv = "data/raw/train.csv"
    audio_dir = "data/raw/train_audio"
    df = pd.read_csv(train_csv)
    
    if limit:
        df = df.head(limit)
        
    unique_species = sorted(df['primary_label'].unique().tolist())
    num_classes = len(unique_species)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['primary_label'])
    
    # 2. Setup Generators
    train_gen = VisualDataGenerator(train_df, audio_dir, unique_species)
    val_gen = VisualDataGenerator(val_df, audio_dir, unique_species, shuffle=False)
    
    # 3. Build and Compile Model
    model = build_visual_model(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # 4. Train
    print(f"Training on {num_classes} species...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("models/visual_baseline.keras", save_best_only=True)
        ]
    )
    
    return model

if __name__ == "__main__":
    # Test with a small subset
    train_standalone_visual(limit=100)
