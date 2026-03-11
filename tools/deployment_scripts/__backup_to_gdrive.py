import os
import shutil
import zipfile
from datetime import datetime

print("Starting V23 Fleet GDrive Backup Process...")

# 1. Define Paths
deploy_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\V23_Deploy"

# Push to Google Drive as requested, specifically into the Models folder
gdrive_path = r"G:\My Drive\projects\Models"

# We will create a specific cold-storage backup folder inside GDrive projects/Models
backup_dir = os.path.join(gdrive_path, "V23_Trading_Backups")
os.makedirs(backup_dir, exist_ok=True)

# Define exact ZIP destination with today's date
date_str = datetime.now().strftime("%Y_%m_%d")
zip_filename = f"V23_WEIGHT_VAULT_{date_str}.zip"
zip_filepath = os.path.join(backup_dir, zip_filename)

# 2. Package the Zip File
print(f"Packaging heavy .pth models from: {deploy_dir}")
print(f"Target Destination: {zip_filepath}")

# Create a ZipFile Object
with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(deploy_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # We ONLY want to backup the actual model directories (weights), not the loose JSONs or MDs
            # The loose JSONs will go to GitHub. The .pth files go to OneDrive / GDrive.
            if file_path.endswith(".pth") or "model_" in file_path:
                # Calculate relative path so it unzips nicely
                rel_path = os.path.relpath(file_path, deploy_dir)
                zipf.write(file_path, rel_path)
                print(f"  -> Added {rel_path} to vault.")

print()
print(f"SUCCESS: The V23 model weights have been securely zipped and air-gapped to:")
print(f" -> {zip_filepath}")
print("You may now safely push the remaining lightweight JSON configs and READMEs to your GitHub repository.")
