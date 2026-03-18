#!/bin/bash
# Prepare a directory with only the 10,000 light noise files
mkdir -p data/processed/noise_bank_light

echo "Copying 10,000 selected noise files..."
# Extract the filenames from the light registry and copy them
tail -n +2 data/processed/noise_bank_registry_light.csv | cut -d',' -f2 | xargs -I {} cp data/processed/noise_bank/{} data/processed/noise_bank_light/

echo "Upload to GCS with: gsutil -m cp -r data/processed/noise_bank_light/* gs://birdclef-2026-data-birdclef-490003/processed/noise_bank/"
