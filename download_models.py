import os
import dropbox
from dropbox.exceptions import ApiError

ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")

MODELS = {
    "laneswithcontrast.pt": "https://www.dropbox.com/scl/fi/s8kr86q8348dsag2w9vj0/laneswithcontrast.pt?rlkey=ef14zhfzw9af782vu0at7l5tr&st=uh1cd51g&dl=0",
    "obstacles.pt": "https://www.dropbox.com/scl/fi/vzeibnyy4tusydr2e6bzq/obstacles.pt?rlkey=km796g4ok2ahl9hugnfbrgy7m&st=756yeoca&dl=0"
}

def download_models():
    raw_token = os.environ.get("DROPBOX_ACCESS_TOKEN", "").strip().strip("'\"")
    print(f"Token length received: {len(raw_token)}")

    # Use the scrubbed token
    dbx = dropbox.Dropbox(raw_token)
    os.makedirs("models", exist_ok=True)

    for name, url in MODELS.items():
        save_path = os.path.join("models", name)
        
        if os.path.exists(save_path):
            print(f"Skipping {name}, already exists.")
            continue

        print(f"Downloading {name} via Dropbox API...")
        try:
            # This API call handles the shared link logic correctly
            metadata, response = dbx.sharing_get_shared_link_file(url=url)
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded {name}.")
        except ApiError as e:
            print(f"Failed to download {name}: {e}")

if __name__ == "__main__":
    download_models()