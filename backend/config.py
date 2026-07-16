import os
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent

load_dotenv(BACKEND_DIR / ".env", override=True)  # Load backend/.env reliably

APP_NAME = "Company RAG API"
API_PREFIX = "/api"

# Inference backends
# INFERENCE_PROVIDER controls answer generation backend: "local", "ollama", or "lmstudio".
INFERENCE_PROVIDER = os.getenv("INFERENCE_PROVIDER", "ollama").strip().lower()

# EMBEDDING_PROVIDER controls embedding backend: "local", "ollama", or "lmstudio".
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "distilbert-base-uncased")
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "").strip()
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# CHAT_MODEL may be either "model-id" or legacy "model-id:provider".
# Provider suffixes are ignored for local inference.
_raw_chat_model = os.getenv("CHAT_MODEL", "distilgpt2")
if ":" in _raw_chat_model.split("/")[-1]:
    _model_part = _raw_chat_model.rsplit(":", 1)[0]
else:
    _model_part = _raw_chat_model

CHAT_MODEL_DEFAULT = _model_part.strip()
CHAT_MODEL_PATH = os.getenv("CHAT_MODEL_PATH", "").strip()

# Ollama (Local API)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b").strip()
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

# LM Studio (OpenAI-compatible local API - kept for backward compatibility)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

LM_STUDIO_CHAT_MODEL = os.getenv("LM_STUDIO_CHAT_MODEL", CHAT_MODEL_DEFAULT).strip()
LM_STUDIO_EMBED_MODEL = os.getenv("LM_STUDIO_EMBED_MODEL", "").strip()

LM_STUDIO_TIMEOUT_SECONDS = int(os.getenv("LM_STUDIO_TIMEOUT_SECONDS", "60"))

# If True, allow downloading from Hugging Face when a model is not present locally.
ALLOW_MODEL_DOWNLOADS = os.getenv("ALLOW_MODEL_DOWNLOADS", "false").strip().lower() in {
    "1", "true", "yes", "on"
}

# Auto-purge conversations older than this many days (0 = disabled)
CONVERSATION_RETENTION_DAYS = int(os.getenv("CONVERSATION_RETENTION_DAYS", "10"))

# File limits
MAX_FILE_SIZE = 50 * 1024 * 1024

# CORS — set ALLOWED_ORIGINS to your production domain(s), comma-separated
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500,null"
).split(",")

# FAISS paths (relative to the backend directory)
FAISS_INDEX_PATH = "faiss_store/index.faiss"
FAISS_META_PATH = "faiss_store/metadata.json"

# Number of chunks to retrieve per query (higher = better recall, slower LLM)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))

# Minimum semantic similarity required for a FAISS hit to be accepted directly.
RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.08"))

# Minimum lexical overlap ratio required for keyword-assisted retrieval.
RAG_MIN_KEYWORD_SCORE = float(os.getenv("RAG_MIN_KEYWORD_SCORE", "0.25"))

# How much keyword overlap should boost ranking alongside semantic similarity.
RAG_KEYWORD_WEIGHT = float(os.getenv("RAG_KEYWORD_WEIGHT", "0.20"))

# Number of keyword matches to consider while hybrid-ranking candidates.
RAG_KEYWORD_TOP_K = int(os.getenv("RAG_KEYWORD_TOP_K", "8"))

# Chunking settings used when new documents are uploaded and indexed.
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "250"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))