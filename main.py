from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import httpx
import uuid
from datetime import datetime
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Pydantic Models
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens for response")
    temperature: Optional[float] = Field(0.7, description="Response temperature")

class ChatResponse(BaseModel):
    response: str = Field(..., description="LLM response")
    session_id: str = Field(..., description="Chat session ID")
    message_count: int = Field(..., description="Total messages in session")

class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: datetime
    last_updated: datetime

# In-memory storage for chat history
chat_sessions: Dict[str, List[Message]] = {}
session_metadata: Dict[str, Dict] = {}

# LLM Configuration
class LLMConfig:
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "deepseek/deepseek-chat"
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

llm_config = LLMConfig()

# FastAPI app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 FastAPI LLM Chatbot starting up...")
    print(f"📡 Using model: {llm_config.model}")
    print(f"🔗 Base URL: {llm_config.base_url}")
    yield
    # Shutdown
    print("🛑 FastAPI LLM Chatbot shutting down...")

app = FastAPI(
    title="LLM Chatbot API",
    description="FastAPI endpoint for LLM chatbot with chat history support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get HTTP client
async def get_http_client():
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client

# Helper functions
def create_session_id() -> str:
    """Generate a new session ID"""
    return str(uuid.uuid4())

def get_or_create_session(session_id: Optional[str]) -> str:
    """Get existing session or create new one"""
    if session_id and session_id in chat_sessions:
        return session_id
    
    new_session_id = create_session_id()
    chat_sessions[new_session_id] = []
    session_metadata[new_session_id] = {
        "created_at": datetime.now(),
        "last_updated": datetime.now()
    }
    return new_session_id

def add_message_to_session(session_id: str, message: Message):
    """Add message to session history"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    chat_sessions[session_id].append(message)
    session_metadata[session_id]["last_updated"] = datetime.now()

def format_messages_for_llm(messages: List[Message]) -> List[Dict]:
    """Format messages for LLM API"""
    return [{"role": msg.role, "content": msg.content} for msg in messages]

async def call_llm_api(
    messages: List[Dict], 
    client: httpx.AsyncClient,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """Call the LLM API and return response"""
    
    headers = {
        "Authorization": f"Bearer {llm_config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "FastAPI LLM Chatbot"
    }
    
    payload = {
        "model": llm_config.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        response = await client.post(
            f"{llm_config.base_url}/chat/completions",
            json=payload,
            headers=headers
        )
        
        if response.status_code != 200:
            error_detail = response.text
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LLM API error: {error_detail}"
            )
        
        response_data = response.json()
        
        if "choices" not in response_data or not response_data["choices"]:
            raise HTTPException(
                status_code=500,
                detail="Invalid response from LLM API"
            )
        
        return response_data["choices"][0]["message"]["content"]
    
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Request to LLM API failed: {str(e)}"
        )

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "LLM Chatbot API is running",
        "model": llm_config.model,
        "active_sessions": len(chat_sessions)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client: httpx.AsyncClient = Depends(get_http_client)
):
    """Main chat endpoint"""
    
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        
        # Add user message to history
        user_message = Message(role="user", content=request.message)
        add_message_to_session(session_id, user_message)
        
        # Get full conversation history
        session_messages = chat_sessions[session_id]
        
        # Format messages for LLM API
        llm_messages = format_messages_for_llm(session_messages)
        
        # Call LLM API
        llm_response = await call_llm_api(
            messages=llm_messages,
            client=client,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Add assistant response to history
        assistant_message = Message(role="assistant", content=llm_response)
        add_message_to_session(session_id, assistant_message)
        
        return ChatResponse(
            response=llm_response,
            session_id=session_id,
            message_count=len(chat_sessions[session_id])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions", response_model=List[SessionInfo])
async def get_sessions():
    """Get all active sessions"""
    sessions = []
    for session_id, metadata in session_metadata.items():
        sessions.append(SessionInfo(
            session_id=session_id,
            message_count=len(chat_sessions.get(session_id, [])),
            created_at=metadata["created_at"],
            last_updated=metadata["last_updated"]
        ))
    return sessions

@app.get("/sessions/{session_id}/history", response_model=List[Message])
async def get_session_history(session_id: str):
    """Get chat history for a specific session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chat_sessions[session_id]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_sessions[session_id]
    del session_metadata[session_id]
    
    return {"message": f"Session {session_id} deleted successfully"}

@app.delete("/sessions")
async def clear_all_sessions():
    """Clear all sessions"""
    session_count = len(chat_sessions)
    chat_sessions.clear()
    session_metadata.clear()
    
    return {"message": f"Cleared {session_count} sessions"}

@app.post("/sessions/{session_id}/clear")
async def clear_session_history(session_id: str):
    """Clear history for a specific session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    message_count = len(chat_sessions[session_id])
    chat_sessions[session_id] = []
    session_metadata[session_id]["last_updated"] = datetime.now()
    
    return {"message": f"Cleared {message_count} messages from session {session_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )