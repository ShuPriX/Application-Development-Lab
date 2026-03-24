"""Download trained model weights from Modal volume to local directory."""

import modal
import os

VOLUME_NAME = "cineinsight-models"
LOCAL_BASE = os.path.join(os.path.dirname(__file__), "..", "models")

app = modal.App("cineinsight-download")
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(volumes={"/models": volume})
def list_files():
    """List all files in the volume."""
    files = []
    for root, dirs, filenames in os.walk("/models"):
        for f in filenames:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            files.append((path, size))
            print(f"  {path} ({size / 1024 / 1024:.1f} MB)")
    return files


@app.function(volumes={"/models": volume})
def read_file(path: str) -> bytes:
    """Read a file from the volume."""
    with open(path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main():
    print("Listing files in Modal volume...")
    files = list_files.remote()

    if not files:
        print("No files found in volume. Train models first.")
        return

    print(f"\nDownloading {len(files)} files to {LOCAL_BASE}...")

    for remote_path, size in files:
        # /models/bert_sentiment/config.json → models/bert_sentiment/config.json
        relative = remote_path.replace("/models/", "")
        local_path = os.path.join(LOCAL_BASE, relative)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"  Downloading {relative} ({size / 1024 / 1024:.1f} MB)...")
        data = read_file.remote(remote_path)
        with open(local_path, "wb") as f:
            f.write(data)

    print(f"\nAll weights downloaded to {LOCAL_BASE}/")
    print("  bert_sentiment/ - BERT model")
    print("  bilstm_aspect/  - BiLSTM model")
