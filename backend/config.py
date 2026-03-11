import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file into the environment

APP_NAME = "Company RAG API"
API_PREFIX = "/api"

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Strip legacy ":provider" suffix (e.g. "model:hf-inference") — newer huggingface_hub
# rejects colons in repo IDs; the provider is passed separately instead.
_raw_chat_model = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct:hf-inference")
if ":" in _raw_chat_model.split("/")[-1]:
    _model_part, _provider_part = _raw_chat_model.rsplit(":", 1)
else:
    _model_part, _provider_part = _raw_chat_model, "hf-inference"

CHAT_MODEL_DEFAULT = _model_part
_SUPPORTED_PROVIDERS = {"fal-ai", "hf-inference", "replicate", "sambanova", "together"}
_configured_provider = os.getenv("CHAT_PROVIDER", _provider_part)
CHAT_PROVIDER = _configured_provider if _configured_provider in _SUPPORTED_PROVIDERS else "hf-inference"

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