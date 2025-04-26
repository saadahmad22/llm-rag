from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
MODEL_PATH = BASE_DIR / "model" / "llama-3.2-3b-instruct-q8_0.gguf"


# Model settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.95

# Vector store settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200