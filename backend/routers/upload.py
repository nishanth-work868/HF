from fastapi import APIRouter, UploadFile, File, HTTPException, status
from services.rag_service import upload_document, clear_index
from config import MAX_FILE_SIZE
import logging

logger = logging.getLogger("upload_router")

router = APIRouter()

@router.post("/upload")
def upload(file: UploadFile = File(...)):
    # Ensure the file isn't larger than the maximum allowed size before processing
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"Upload rejected: File size {file_size} exceeds maximum {MAX_FILE_SIZE}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB."
        )
        
    return upload_document(file)

@router.delete("/clear-index")
def clear():
    """Clear the entire FAISS index. Use this before re-uploading documents."""
    return clear_index()