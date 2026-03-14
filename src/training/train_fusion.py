import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime
from src.audio.spectrograms import SpectrogramGenerator
from src.models.fusion_model import build_fusion_model

class FusionDataGenerator(tf.keras.utils.Sequence):
    """
    Multi-modal data generator: Spectrogram + Perch Embedding + Metadata
    """
    def __init__(self, df, audio_dir, perch_embeddings, labels, 
                 batch_size=32, img_size=224, shuffle=True,
                 precalc_dir=None):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.perch_embeddings = perch_embeddings
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.precalc_dir = precalc_dir
        self.spec_gen = SpectrogramGenerator(img_size=img_size)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def _extract_metadata(self, row):
        # 1. Latitude and Longitude (Standardized)
        lat = row.get('latitude', 0.0) / 90.0
        lon = row.get('longitude', 0.0) / 180.0
        
        # 2. Temporal Features (Month and Day of Year)
        month = 0.5 # Default middle of year
        day_of_year = 0.5
        
        try:
            # Try to extract from URL if it's an iNat timestamp
            if 'url' in row and '?' in str(row['url']):
                ts_str = str(row['url']).split('?')[-1]
                if ts_str.isdigit():
                    dt = datetime.fromtimestamp(int(ts_str))
                    month = dt.month / 12.0
                    day_of_year = dt.timetuple().tm_yday / 366.0
            # Try to extract from filename if it's soundscape format
            elif 'BC2026_Train' in str(row['filename']):
                # BC2026_Train_0039_S22_20211231_201500.ogg
                date_part = str(row['filename']).split('_')[4]
                dt = datetime.strptime(date_part, '%Y%m%d')
                month = dt.month / 12.0
                day_of_year = dt.timetuple().tm_yday / 366.0
        except:
            pass
            
        return np.array([lat, lon, month, day_of_year], dtype=np.float32)

    def _load_or_gen_spec(self, row):
        if self.precalc_dir:
            rel_path = os.path.splitext(row['filename'])[0] + ".png"
            path = os.path.join(self.precalc_dir, rel_path)
            if os.path.exists(path):
                import cv2
                img = cv2.imread(path)
                if img is not None:
                    return img
        
        # Fallback to on-the-fly generation
        audio_path = os.path.join(self.audio_dir, row['filename'])
        return self.spec_gen.generate(audio_path)

    def __getitem__(self, index):
        batch_df = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        
        X_spec = []
        X_perch = []
        X_meta = []
        y = []
        
        for _, row in batch_df.iterrows():
            # 1. Visual Path
            img = self._load_or_gen_spec(row)
            
            if img is not None:
                X_spec.append(img / 255.0)
                
                # 2. Perch Path
                # Use embedding_idx to fetch from pre-loaded array
                idx = int(row['embedding_idx'])
                X_perch.append(self.perch_embeddings[idx])
                
                # 3. Metadata Path
                meta = self._extract_metadata(row)
                X_meta.append(meta)
                
                # 4. Label
                label_vec = np.zeros(len(self.labels))
                label_vec[self.labels.index(row['primary_label'])] = 1.0
                y.append(label_vec)
        
        return [np.array(X_spec), np.array(X_perch), np.array(X_meta)], np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

def train_fusion_model(limit=None):
    # 1. Setup Paths
    processed_dir = "data/processed"
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Cloud Setup: Download data if missing
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print("Data missing. Downloading via Kaggle CLI...")
        os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if os.path.exists(zip_path):
            os.system(f"unzip -qo {zip_path} -d {raw_dir}")
            os.remove(zip_path)

    # Cloud Setup: Download pre-calculated embeddings from GCS if available
    csv_path = os.path.join(processed_dir, "train_with_perch_v1.csv")
    embeddings_path = os.path.join(processed_dir, "perch_v1_embeddings.npz")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(csv_path):
        gcs_bucket = os.getenv("GCS_BUCKET")
        if gcs_bucket:
            print(f"Downloading embeddings from gs://{gcs_bucket}...")
            os.system(f"gsutil cp gs://{gcs_bucket}/processed/perch_v1_embeddings.npz {embeddings_path}")
            os.system(f"gsutil cp gs://{gcs_bucket}/processed/train_with_perch_v1.csv {csv_path}")
        else:
            print("ERROR: Embeddings missing and GCS_BUCKET not set.")
            return

    audio_dir = os.path.join(raw_dir, "train_audio")
    
    # 2. Load Data
    print("Loading metadata and embeddings...")
    df = pd.read_csv(csv_path)
    # The row index in train_with_perch_v1.csv matches the index in the embeddings npz
    df['embedding_idx'] = np.arange(len(df))
    
    with np.load(embeddings_path) as data:
        perch_embeddings = data['embeddings']
        # indices = data['indices'] # Not needed if CSV matches embedding rows
        
    if limit:
        df = df.head(limit)
    
    # Stratified split requires at least 2 samples per class
    counts = df['primary_label'].value_counts()
    df = df[df['primary_label'].isin(counts[counts >= 2].index)]
        
    unique_species = sorted(df['primary_label'].unique().tolist())
    num_classes = len(unique_species)
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['primary_label'])
    
    # 3. Setup Generators
    precalc_dir = os.path.join(processed_dir, "spectrograms")
    if not os.path.exists(precalc_dir):
        print(f"Precalculated spectrograms not found. Generating now for {len(df)} files...")
        from src.audio.precalculate_spectrograms import main as precalculate
        # Using 16 workers for n1-standard-32
        precalculate(limit=limit, workers=16)
        
    print(f"Using precalculated spectrograms from {precalc_dir}")

    train_gen = FusionDataGenerator(train_df, audio_dir, perch_embeddings, unique_species, precalc_dir=precalc_dir)
    val_gen = FusionDataGenerator(val_df, audio_dir, perch_embeddings, unique_species, shuffle=False, precalc_dir=precalc_dir)
    
    # 4. Build and Compile Model
    print(f"Building Fusion Model for {num_classes} species...")
    model = build_fusion_model(num_classes=num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Lower LR for fusion
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # 5. Train
    print("Starting training...")
    model_name = "fusion_model_v1.keras"
    model_path = os.path.join("models", model_name)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]
    
    # Add TensorBoard logging
    gcs_bucket = os.getenv("GCS_BUCKET")
    if gcs_bucket:
        log_dir = f"gs://{gcs_bucket}/logs/fusion_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        log_dir = f"logs/fusion_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs("logs", exist_ok=True)
        
    print(f"TensorBoard logs will be written to {log_dir}")
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch'))

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=callbacks
    )
    
    # Upload final model to GCS
    gcs_bucket = os.getenv("GCS_BUCKET")
    if gcs_bucket:
        print(f"Uploading model to gs://{gcs_bucket}/models/{model_name}...")
        os.system(f"gsutil cp {model_path} gs://{gcs_bucket}/models/{model_name}")
    
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for smoke testing")
    args = parser.parse_args()
    
    # Train on full dataset or limited subset
    train_fusion_model(limit=args.limit)
