from fastapi import APIRouter
from models.schemas import QueryRequest, ConversationResponse
from services.rag_service import handle_query

router = APIRouter()

@router.post("/query", response_model=ConversationResponse)
def query_rag(request: QueryRequest):
    return handle_query(request)