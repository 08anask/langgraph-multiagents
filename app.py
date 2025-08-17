import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from helpers.graph_manager import GraphManager

# Instantiate once (loads models, builds graph, enables memory)
graph = GraphManager()

app = FastAPI(title="RAG + Internet Manager Graph API", version="1.0.0")

# CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    thread_id: str = Field(..., description="Conversation/thread key for short-term memory")
    query: Optional[str] = Field(None, description="User query")
    user_reply: Optional[str] = Field(None, description='Free-form reply like "yes" or "no" for consent')
    user_consent_internet: Optional[bool] = Field(None, description="Explicit boolean consent to use the web")

class QueryResponse(BaseModel):
    final: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not (req.query or req.user_reply is not None or req.user_consent_internet is not None):
        raise HTTPException(status_code=400, detail="Send at least one of: query, user_reply, user_consent_internet")

    
    final = graph.invoke(
        thread_id=req.thread_id,
        query=req.query,
        user_reply=req.user_reply,
        user_consent_internet=req.user_consent_internet,
    )
    if final is None:
        raise HTTPException(status_code=500, detail="No final payload produced by the graph")
    return QueryResponse(final=final)
    """except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))"""

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)
