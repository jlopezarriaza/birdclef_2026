import os
import numpy as np
import pandas as pd
import tensorflow as tf
from src.training.train_fusion import FusionDataGenerator
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_mini_model(limit_train=500, eval_size=2000):
    """
    Evaluates the fusion model on the unseen portion of the train_audio dataset.
    """
    processed_dir = "data/processed"
    raw_dir = "data/raw"
    model_path = "models/fusion_model_v1.keras"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 1. Load Data
    print("Loading metadata and embeddings...")
    csv_path = os.path.join(processed_dir, "train_with_perch_v1.csv")
    df = pd.read_csv(csv_path)
    df['embedding_idx'] = np.arange(len(df))
    
    with np.load(os.path.join(processed_dir, "perch_v1_embeddings.npz")) as data:
        perch_embeddings = data['embeddings']
    
    # 2. Identify Unseen Data
    # The training used the first 'limit_train' rows. We'll take the next 'eval_size' rows.
    # Note: We should use the same unique_species list used during training.
    # Training unique_species was sorted(df.head(limit)['primary_label'].unique())
    train_df_subset = df.head(limit_train)
    counts = train_df_subset['primary_label'].value_counts()
    train_df_subset = train_df_subset[train_df_subset['primary_label'].isin(counts[counts >= 2].index)]
    unique_species = sorted(train_df_subset['primary_label'].unique().tolist())
    
    print(f"Model was trained on {len(unique_species)} species from the first {limit_train} rows.")
    
    # Now get evaluation data that belongs to THESE species but was not in the first 500 rows
    eval_df = df.iloc[limit_train:]
    eval_df = eval_df[eval_df['primary_label'].isin(unique_species)]
    
    # Take a larger sample if possible, or just all of them
    max_eval = min(len(eval_df), eval_size)
    if max_eval > 0:
        eval_df = eval_df.sample(n=max_eval, random_state=42)
    
    print(f"Evaluating on {len(eval_df)} unseen samples from the same {len(unique_species)} species...")

    if len(eval_df) == 0:
        print("No unseen samples found for these species. Try increasing limit_train or checking species distribution.")
        return

    # 3. Setup Generator
    audio_dir = os.path.join(raw_dir, "train_audio")
    eval_gen = FusionDataGenerator(
        eval_df, 
        audio_dir, 
        perch_embeddings, 
        unique_species, 
        batch_size=32, 
        shuffle=False
    )

    # 4. Load Model and Predict
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Running inference...")
    y_pred_probs = model.predict(eval_gen)
    
    y_true = []
    for i in range(len(eval_gen)):
        _, batch_y = eval_gen[i]
        y_true.extend(batch_y)
    y_true = np.array(y_true)
    y_pred_probs = y_pred_probs[:len(y_true)]

    # 5. Calculate Metrics
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    
    acc = np.mean(y_true_idx == y_pred_idx)
    print(f"\nUnseen Data Accuracy: {acc:.4f}")
    
    # Use labels parameter to avoid ValueError if some species are missing in eval set
    present_labels = np.unique(np.concatenate([y_true_idx, y_pred_idx]))
    present_names = [unique_species[i] for i in present_labels]

    report = classification_report(
        y_true_idx, 
        y_pred_idx, 
        labels=present_labels,
        target_names=present_names, 
        output_dict=True, 
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    print(report_df.head(10))

if __name__ == "__main__":
    evaluate_mini_model()
