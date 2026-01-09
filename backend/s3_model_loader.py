import os
import boto3
from dotenv import load_dotenv
from tensorflow.keras.models import load_model as load_keras_model
import torch

# Load environment variables
load_dotenv()

BUCKET = os.getenv("S3_BUCKET_NAME")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
LOCAL_MODEL_DIR = "./models_cache"  # where models will be saved locally

# Create local folder if it doesn't exist
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Initialize S3 client
s3 = boto3.client("s3", region_name=REGION)


def download_model_from_s3(filename: str) -> str:
    """
    Downloads a file from S3 to the local cache folder.
    Returns the local path of the file.
    """
    local_path = os.path.join(LOCAL_MODEL_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from S3...")
        s3.download_file(BUCKET, filename, local_path)
        print(f"Downloaded {filename} to {local_path}")
    else:
        print(f"{filename} already exists locally. Using cached version.")
    return local_path


def load_cnn_model(filename: str):
    """
    Loads a Keras CNN model (.h5) from S3.
    Returns the loaded Keras model.
    """
    local_path = download_model_from_s3(filename)
    model = load_keras_model(local_path, compile=False)
    print(f"CNN model {filename} loaded successfully")
    return model


def load_torch_model(filename: str, device: str = None):
    """
    Loads a PyTorch model (.pth) from S3.
    Returns the loaded PyTorch model.
    """
    local_path = download_model_from_s3(filename)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(local_path, map_location=device)
    model.eval()
    print(f"PyTorch model {filename} loaded successfully on {device}")
    return model
