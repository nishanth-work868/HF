from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from contextlib import asynccontextmanager

from pathlib import Path
from config import APP_NAME, API_PREFIX, ALLOWED_ORIGINS

from routers import query, upload

# Ensure logs directory exists before creating FileHandler
Path("logs").mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/rag.log"),
        logging.StreamHandler()
    ]
)

_cleanup_logger = logging.getLogger("cleanup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


@router.get("/health")
def health():
    return {"status": "ok"}


app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    docs_url=f"{API_PREFIX}/docs",
    openapi_url=f"{API_PREFIX}/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(query.router, prefix=API_PREFIX, tags=["RAG"])
app.include_router(upload.router, prefix=API_PREFIX, tags=["Upload"])


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return FileResponse(str(Path(__file__).resolve().parent.parent / "index.html"))