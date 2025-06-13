#!/usr/bin/env python3
"""FastAPI web UI for AI-Prepping."""

import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import asyncio

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config
from scripts.chat import RAGChatbot
from scripts.chunk_utils import TextChunk


# Initialize FastAPI app
app = FastAPI(title="AI-Prepping", description="Offline RAG Chatbot")

# Templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Global chatbot instance
chatbot = None


class ChatRequest(BaseModel):
    """Chat request model."""
    question: str
    show_sources: bool = True


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    sources: List[dict] = []
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup."""
    global chatbot
    config = get_config()
    try:
        chatbot = RAGChatbot(config)
        print("✓ Chatbot initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize chatbot: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "AI-Prepping Offline Chatbot"
    })


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle chat requests."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Get answer from chatbot
        answer, chunks = await asyncio.to_thread(
            chatbot.answer_question, request.question
        )
        
        # Format sources
        sources = []
        if request.show_sources and chunks:
            for i, chunk in enumerate(chunks[:5]):  # Limit to top 5
                sources.append({
                    "title": chunk.metadata.get('title', 'Unknown'),
                    "score": round(chunk.metadata.get('score', 0.0), 2),
                    "preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
                })
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def status():
    """Get system status."""
    if not chatbot:
        return {"status": "error", "message": "Chatbot not initialized"}
    
    try:
        config = get_config()
        return {
            "status": "ready",
            "model": config.llm_model,
            "index_size": chatbot.index.ntotal,
            "chunks_loaded": len(chatbot.chunks),
            "memory_mode": "low" if config.is_low_memory else "normal"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/stats")
async def stats():
    """Get detailed statistics."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    config = get_config()
    return {
        "system": {
            "memory_mode": "low" if config.is_low_memory else "normal",
            "max_memory_gb": config.get('system.max_memory_gb', 32)
        },
        "models": {
            "llm": config.llm_model,
            "embeddings": config.get('models.embeddings.model')
        },
        "index": {
            "total_vectors": chatbot.index.ntotal,
            "total_chunks": len(chatbot.chunks),
            "dimension": config.get('models.embeddings.dimension', 1024)
        },
        "retrieval": {
            "top_k": config.get('retrieval.top_k', 5),
            "score_threshold": config.get('retrieval.score_threshold', 0.7)
        }
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )