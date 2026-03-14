import pandas as pd
import numpy as np
import os

def run_eda():
    train_csv = "data/raw/train.csv"
    ss_csv = "data/raw/train_soundscapes_labels.csv"
    
    if not os.path.exists(train_csv) or not os.path.exists(ss_csv):
        print("Data files missing.")
        return

    df_train = pd.read_csv(train_csv)
    df_ss = pd.read_csv(ss_csv)

    # 1. Species Overlap
    train_species = set(df_train['primary_label'].unique())
    # Soundscape labels are semicolon separated
    ss_species = set(df_ss['primary_label'].str.split(';').explode().unique())
    # Filter out empty or 'nan' strings if any
    ss_species = {s for s in ss_species if isinstance(s, str) and s.strip()}

    print("--- 1. Species Distribution ---")
    print(f"Species in Short Recordings (train_audio): {len(train_species)}")
    print(f"Species in Soundscapes (train_soundscapes): {len(ss_species)}")
    
    only_in_ss = ss_species - train_species
    only_in_train = train_species - ss_species
    intersection = train_species & ss_species
    
    print(f"Species in both: {len(intersection)}")
    print(f"Species ONLY in Soundscapes (Missing from short audio!): {len(only_in_ss)}")
    print(f"Species ONLY in Short Audio (Not in current soundscapes): {len(only_in_train)}")
    if only_in_ss:
        print(f"Examples missing from train_audio: {list(only_in_ss)[:5]}")

    # 2. Label Density (Multi-label analysis)
    print("\n--- 2. Label Density ---")
    ss_label_counts = df_ss['primary_label'].str.split(';').str.len()
    print(f"Soundscape - Avg labels per 5s window: {ss_label_counts.mean():.2f}")
    print(f"Soundscape - Max labels in one window: {ss_label_counts.max()}")
    print(f"Soundscape - % of windows with >1 species: {(ss_label_counts > 1).mean()*100:.1f}%")
    
    # train_audio also has secondary labels
    train_secondary = df_train['secondary_labels'].apply(eval) # it's stored as a string representation of a list
    train_total_labels = 1 + train_secondary.str.len()
    print(f"Short Audio - Avg labels per recording: {train_total_labels.mean():.2f}")
    print(f"Short Audio - % of recordings with secondary labels: {(train_secondary.str.len() > 0).mean()*100:.1f}%")

    # 3. Quality & Source
    print("\n--- 3. Metadata Comparison ---")
    if 'rating' in df_train.columns:
        print(f"Short Audio - Avg Rating (1-5): {df_train['rating'].mean():.2f}")
    
    print(f"Short Audio - Collections: {df_train['collection'].value_counts().to_dict()}")
    
    # 4. Geospatial (The Pantanal check)
    # Soundscapes are from specific Pantanal sites. Let's see how many train samples are near there.
    # Pantanal roughly: Lat -15 to -22, Lon -60 to -54
    pantanal_lat = (-22, -15)
    pantanal_lon = (-60, -54)
    
    in_pantanal = df_train[
        (df_train['latitude'] >= pantanal_lat[0]) & (df_train['latitude'] <= pantanal_lat[1]) &
        (df_train['longitude'] >= pantanal_lon[0]) & (df_train['longitude'] <= pantanal_lon[1])
    ]
    
    print(f"Short Audio - Samples recorded in/near Pantanal: {len(in_pantanal)} ({len(in_pantanal)/len(df_train)*100:.1f}%)")

if __name__ == "__main__":
    run_eda()
