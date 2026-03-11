import os
import shutil
import zipfile
import argparse
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

DEFAULT_GDRIVE_PATH = r"G:\My Drive\projects\Models\V23_Trading_Backups"

def backup_models(source_dir, gdrive_dest, models=None):
    if not os.path.exists(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        return

    os.makedirs(gdrive_dest, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    zip_filename = f"V23_WEIGHT_VAULT_{date_str}.zip"
    zip_filepath = os.path.join(gdrive_dest, zip_filename)
    
    logger.info(f"Packaging heavy .pth models from: {source_dir}")
    logger.info(f"Target Destination: {zip_filepath}")
    
    # Track what we backup
    backed_up_models = []
    
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            current_model_dir = os.path.basename(root)
            
            # If specific models are requested, skip if we aren't in their directory tree
            if models and not any(m in root for m in models):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                
                # We ONLY vault the heavy .pth files or model structures
                # Lightweight config.json, metadata.json, and summaries are ignored
                # because they should remain in the repo to be checked into GitHub
                if file_path.endswith(".pth") or file.startswith("model_"):
                    rel_path = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, rel_path)
                    logger.info(f"  -> Vaulted: {rel_path}")
                    
                    if current_model_dir not in backed_up_models:
                        backed_up_models.append(current_model_dir)

    if not backed_up_models:
        logger.warning("No models (.pth files) were found to backup! Deleting empty zip.")
        os.remove(zip_filepath)
        return

    logger.info("\n" + "="*50)
    logger.info(f"✅ SUCCESS: The V23 model weights have been securely air-gapped.")
    logger.info(f"📂 Location: {zip_filepath}")
    logger.info("="*50)
    logger.info("\n>>> NEXT STEPS FOR SOURCE CONTROL <<<")
    logger.info("The original `metadata.json`, `config.json`, and markdown summaries")
    logger.info("remain untouched in your local `weights/` directory.")
    logger.info("You may now safely run `git add weights/` to commit the lightweight metadata to GitHub!")

def main():
    parser = argparse.ArgumentParser(description="Secure GDrive Model Weight Backup")
    parser.add_argument("--env", type=str, default="V23_Deploy", help="Source deployment environment folder inside weights/ (default: V23_Deploy)")
    parser.add_argument("--models", nargs="*", help="Optional: List of specific model folders to backup. If empty, backs up everything in the env.")
    parser.add_argument("--dest", type=str, default=DEFAULT_GDRIVE_PATH, help="Google Drive destination path")
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    source_dir = os.path.join(root_dir, "weights", args.env)
    
    backup_models(source_dir, args.dest, args.models)

if __name__ == "__main__":
    main()
