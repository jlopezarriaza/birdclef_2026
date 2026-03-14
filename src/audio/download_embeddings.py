import os
import argparse
import subprocess

def download_gsutil(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob using gsutil."""
    gs_path = f"gs://{bucket_name}/{source_blob_name}"
    print(f"Downloading {gs_path} to {destination_file_name}...")
    try:
        subprocess.run(["gsutil", "cp", gs_path, destination_file_name], check=True)
        print(f"Downloaded {source_blob_name} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {source_blob_name} using gsutil: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download embeddings from GCS")
    parser.add_argument("--bucket", type=str, default="birdclef-2026-data-birdclef-490003", help="GCS bucket name")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Local directory to save files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files_to_download = [
        "processed/birdnet_embeddings.npz",
        "processed/perch_v1_embeddings.npz",
        "processed/train_with_perch_v1.csv"
    ]

    for remote_path in files_to_download:
        filename = os.path.basename(remote_path)
        local_path = os.path.join(args.output_dir, filename)
        download_gsutil(args.bucket, remote_path, local_path)

if __name__ == "__main__":
    main()
