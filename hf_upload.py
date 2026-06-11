from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

LOCAL_MODEL_PATH = "clinical_trial_model"
HF_REPO_ID = os.getenv("HF_REPO")

if __name__ == "__main__":
    api = HfApi()
    print(f"Uploading '{LOCAL_MODEL_PATH}' to {HF_REPO_ID}")
    api.upload_folder(
        folder_path=LOCAL_MODEL_PATH,
        repo_id=HF_REPO_ID,
        commit_message="Initial upload"
    )
    print("Upload successful")
