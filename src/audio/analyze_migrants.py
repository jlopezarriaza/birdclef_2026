import pandas as pd
import numpy as np
import os

def analyze_migrants():
    data_dir = "data/raw"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    taxonomy_df = pd.read_csv(os.path.join(data_dir, "taxonomy.csv"))
    
    print("Taxonomy Columns:", taxonomy_df.columns.tolist())
    
    # Since 'family' is missing, we'll look for keywords in scientific names 
    # or just analyze the species codes for known migrants.
    # Common Pantanal migratory genera:
    # Hirundo (Swallows), Tringa (Sandpipers), Calidris (Sandpipers), 
    # Tyrannus (Flycatchers), Progne (Martins)
    migratory_genera = ['Hirundo', 'Tringa', 'Calidris', 'Tyrannus', 'Progne', 'Pandion', 'Cathartes']
    
    def is_migrant(sci_name):
        genus = str(sci_name).split(' ')[0]
        return genus in migratory_genera

    potential_migrants = taxonomy_df[taxonomy_df['scientific_name'].apply(is_migrant)]
    migrant_codes = potential_migrants['primary_label'].unique()
    
    train_migrants = train_df[train_df['primary_label'].isin(migrant_codes)]
    
    print(f"\n--- Migratory Species Analysis ---")
    print(f"Total target species in taxonomy: {len(taxonomy_df)}")
    print(f"Potential migratory species found (by common genera): {len(potential_migrants)}")
    print(f"Migratory species present in train_audio: {train_migrants['primary_label'].nunique()}")
    
    print("\nExample Migratory Species identified:")
    print(potential_migrants[['primary_label', 'scientific_name', 'common_name']].head(10))
    
    # Soundscape Temporal Distribution
    soundscape_labels_path = os.path.join(data_dir, "train_soundscapes_labels.csv")
    if os.path.exists(soundscape_labels_path):
        soundscapes_df = pd.read_csv(soundscape_labels_path)
        
        print("\nAnalyzing Soundscape Temporal Distribution...")
        def extract_month(filename):
            try:
                # BC2026_Train_9991_S22_20240213_220000.ogg
                parts = filename.split('_')
                date_str = parts[4] # 20240213
                return int(date_str[4:6]) # 02
            except:
                return None

        soundscapes_df['month'] = soundscapes_df['filename'].apply(extract_month)
        month_dist = soundscapes_df['month'].value_counts().sort_index()
        
        print("Soundscape Samples per Month:")
        for month, count in month_dist.items():
            print(f"Month {month:02d}: {count} segments")
        
        # Check if any migratory species appear only in certain months
        # (This requires joining labels with the extracted months)
        # Note: 'primary_label' in soundscapes is semicolon separated
        print("\nTip: Incorporating 'Month' as a feature will be critical for distinguishing")
        print("residents from seasonal migrants like Swallows and Sandpipers.")

if __name__ == "__main__":
    analyze_migrants()
