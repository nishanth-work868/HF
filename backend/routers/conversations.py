from fastapi import APIRouter
from services.rag_service import (
    get_conversations,
    get_conversation,
    delete_conversation
)

router = APIRouter()

@router.get("/conversations")
def list_conversations():
    return get_conversations()

@router.get("/conversations/{conversation_id}")
def fetch_conversation(conversation_id: str):
    return get_conversation(conversation_id)

@router.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str):
    return delete_conversation(conversation_id)