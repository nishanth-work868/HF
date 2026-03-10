import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file into the environment

APP_NAME = "Company RAG API"
API_PREFIX = "/api"

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL_DEFAULT = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct:novita")

# HuggingFace Inference API
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Auto-purge conversations older than this many days (0 = disabled)
CONVERSATION_RETENTION_DAYS = int(os.getenv("CONVERSATION_RETENTION_DAYS", "10"))

# File limits
MAX_FILE_SIZE = 50 * 1024 * 1024

# CORS — set ALLOWED_ORIGINS to your production domain(s), comma-separated
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500"
).split(",")

# FAISS paths (relative to the backend directory)
FAISS_INDEX_PATH = "faiss_store/index.faiss"
FAISS_META_PATH = "faiss_store/metadata.json"

# Number of chunks to retrieve per query (higher = better recall, slower LLM)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))