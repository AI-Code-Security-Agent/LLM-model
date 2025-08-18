# main.py  (FastAPI) — Multi-LLM with HF + OpenAI-compat providers + Response Logging

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
import os, json, uuid, httpx, asyncio, logging
from dotenv import load_dotenv, find_dotenv
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        # Uncomment below to also log to file
        # logging.FileHandler('api_responses.log')
    ]
)
logger = logging.getLogger("LLM_API")

load_dotenv(find_dotenv(), override=True)

# -----------------------------
# Pydantic DTOs
# -----------------------------
class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    messages: Optional[List[Message]] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    # NEW
    model_id: Optional[str] = None
    provider: Optional[str] = None  # openrouter | hf_inference | openai_compat

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_count: int
    model_id: str
    provider: str

class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: datetime
    last_updated: datetime

class PublicModel(BaseModel):
    id: str
    provider: str
    label: str
    stream: bool = False
    base_url: Optional[str] = None

# -----------------------------
# Helper function to log responses
# -----------------------------
def log_chat_response(session_id: str, user_message: str, ai_response: str, model_id: str, provider: str, response_time: float = None):
    """Log chat interactions to terminal"""
    separator = "=" * 80
    logger.info(f"\n{separator}")
    logger.info(f"🤖 CHAT RESPONSE GENERATED")
    logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🔑 Session ID: {session_id}")
    logger.info(f"🧠 Model: {model_id} ({provider})")
    if response_time:
        logger.info(f"⏱️  Response Time: {response_time:.2f}s")
    logger.info(f"👤 User Message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
    logger.info(f"🤖 AI Response: {ai_response[:200]}{'...' if len(ai_response) > 200 else ''}")
    logger.info(f"📊 Response Length: {len(ai_response)} characters")
    logger.info(f"{separator}\n")

def log_stream_start(session_id: str, user_message: str, model_id: str, provider: str):
    """Log start of streaming response"""
    logger.info(f"🔄 STREAMING STARTED - Session: {session_id} | Model: {model_id} ({provider}) | Message: {user_message[:50]}...")

def log_stream_complete(session_id: str, total_tokens: int, response_time: float = None):
    """Log completion of streaming response"""
    time_info = f" | Time: {response_time:.2f}s" if response_time else ""
    logger.info(f"✅ STREAMING COMPLETED - Session: {session_id} | Tokens: {total_tokens}{time_info}")

# -----------------------------
# In-memory session fallback
# -----------------------------
chat_sessions: Dict[str, List[Message]] = {}
session_metadata: Dict[str, Dict] = {}

def create_session_id() -> str:
    return str(uuid.uuid4())

def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in chat_sessions:
        return session_id
    sid = create_session_id()
    chat_sessions[sid] = []
    session_metadata[sid] = {"created_at": datetime.now(), "last_updated": datetime.now()}
    logger.info(f"📝 New session created: {sid}")
    return sid

def add_message_to_session(sid: str, msg: Message):
    chat_sessions.setdefault(sid, []).append(msg)
    session_metadata[sid]["last_updated"] = datetime.now()

def format_messages_for_llm(messages: List[Message]) -> List[Dict]:
    return [{"role": m.role, "content": m.content} for m in messages]

def build_conversation_context(current_message: str, history: List[Message]) -> List[Dict]:
    llm_messages = format_messages_for_llm(history)
    llm_messages.append({"role": "user", "content": current_message})
    return llm_messages

# -----------------------------
# Model registry
# -----------------------------
def _load_models() -> List[PublicModel]:
    raw = os.getenv("ALLOWED_MODELS_JSON", "").strip()
    if not raw:
        print("ALLOWED_MODELS_JSON missing or empty")
        return []
    try:
        # tolerate backslash continuations and trailing comma
        cleaned = re.sub(r'\\\s*\n', '', raw)            # remove line-continuation backslashes
        cleaned = re.sub(r',\s*]', ']', cleaned)         # remove trailing comma before ]
        arr = json.loads(cleaned)
        models = [PublicModel(**x) for x in arr]
        return models
    except Exception as e:
        print("Failed to parse ALLOWED_MODELS_JSON:", e)
        print("Raw value was:", raw)
        return []

ALLOWED_MODELS: List[PublicModel] = _load_models()
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openrouter")
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", ALLOWED_MODELS[0].id if ALLOWED_MODELS else "")

def resolve_model(model_id: Optional[str], provider: Optional[str]) -> PublicModel:
    # If both provided, enforce exact match
    if model_id and provider:
        for m in ALLOWED_MODELS:
            if m.id == model_id and m.provider == provider:
                return m
        raise HTTPException(400, "Requested model/provider not allowed")
    # If only model_id
    if model_id:
        for m in ALLOWED_MODELS:
            if m.id == model_id:
                return m
        raise HTTPException(400, "Requested model_id not allowed")
    # If only provider
    if provider:
        for m in ALLOWED_MODELS:
            if m.provider == provider:
                return m
        raise HTTPException(400, "No allowed model for requested provider")
    # Default
    for m in ALLOWED_MODELS:
        if m.id == DEFAULT_MODEL_ID and m.provider == DEFAULT_PROVIDER:
            return m
    if ALLOWED_MODELS:
        return ALLOWED_MODELS[0]
    raise HTTPException(500, "No models configured")

# -----------------------------
# Providers
# -----------------------------
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

HEADERS_OR = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "FastAPI LLM Chatbot",
}
HEADERS_HF = {
    "Authorization": f"Bearer {HF_API_KEY}" if HF_API_KEY else "",
    "Content-Type": "application/json",
}
# For openai-compat, we reuse OpenRouter headers (same shape)
HEADERS_OAICOMPAT = {
    "Content-Type": "application/json",
}

async def provider_openrouter_chat(client: httpx.AsyncClient, model: PublicModel, messages: List[Dict], max_tokens: int, temperature: float) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(400, "OPENROUTER_API_KEY missing")
    payload = {
        "model": model.id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    
    start_time = datetime.now()
    resp = await client.post(f"{OPENROUTER_BASE}/chat/completions", headers=HEADERS_OR, json=payload, timeout=120)
    response_time = (datetime.now() - start_time).total_seconds()
    
    if resp.status_code != 200:
        logger.error(f"❌ OpenRouter API Error: {resp.status_code} - {resp.text}")
        raise HTTPException(resp.status_code, f"OpenRouter error: {resp.text}")
    data = resp.json()
    try:
        response_text = data["choices"][0]["message"]["content"]
        logger.info(f"✅ OpenRouter response received in {response_time:.2f}s | Model: {model.id} | Length: {len(response_text)} chars")
        return response_text
    except Exception:
        logger.error(f"❌ Malformed OpenRouter response: {data}")
        raise HTTPException(500, "Malformed OpenRouter response")

async def provider_openrouter_stream(client: httpx.AsyncClient, model: PublicModel, messages: List[Dict], max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    if not OPENROUTER_API_KEY:
        yield f"event: error\ndata: {json.dumps({'detail':'OPENROUTER_API_KEY missing'})}\n\n"
        return
    payload = {
        "model": model.id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    
    token_count = 0
    start_time = datetime.now()
    
    async with client.stream("POST", f"{OPENROUTER_BASE}/chat/completions", headers=HEADERS_OR, json=payload, timeout=None) as resp:
        if resp.status_code != 200:
            detail = await resp.aread()
            logger.error(f"❌ OpenRouter Stream Error: {resp.status_code} - {detail.decode()}")
            yield f"event: error\ndata: {json.dumps({'detail': detail.decode(errors='ignore')})}\n\n"
            return
            
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                response_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ OpenRouter stream completed | Tokens: {token_count} | Time: {response_time:.2f}s")
                yield "data: [DONE]\n\n"
                break
            try:
                obj = json.loads(data)
                delta = obj.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    token_count += 1
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
            except Exception:
                continue

async def provider_hf_inference_chat(client: httpx.AsyncClient, model: PublicModel, messages: List[Dict], max_tokens: int, temperature: float) -> str:
    """
    Hugging Face Inference API (serverless). No SSE streaming.
    We take the last user message and optionally use the history.
    """
    if not HF_API_KEY:
        raise HTTPException(400, "HF_API_KEY missing")

    # simple prompt assembly
    # (You can format like ChatML if your model expects it)
    prompt = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        prefix = "User: " if role == "user" else "Assistant: "
        prompt += f"{prefix}{content}\n"
    prompt += "Assistant:"

    url = f"https://api-inference.huggingface.co/models/{model.id}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
    }
    
    start_time = datetime.now()
    resp = await client.post(url, headers=HEADERS_HF, json=payload, timeout=120)
    response_time = (datetime.now() - start_time).total_seconds()
    
    if resp.status_code != 200:
        logger.error(f"❌ HuggingFace API Error: {resp.status_code} - {resp.text}")
        raise HTTPException(resp.status_code, f"HuggingFace error: {resp.text}")
    data = resp.json()
    """
    HF returns a list of dicts like:
    [{"generated_text": "User: ... Assistant: ..."}]  OR
    [{"summary_text": "..."}] depending on task. For text-generation, generated_text.
    """
    try:
        out = data[0].get("generated_text", "")
        # Strip the prompt if model echoes
        if out.startswith(prompt):
            out = out[len(prompt):]
        out = out.strip()
        logger.info(f"✅ HuggingFace response received in {response_time:.2f}s | Model: {model.id} | Length: {len(out)} chars")
        return out
    except Exception:
        logger.error(f"❌ Malformed HuggingFace response: {data}")
        raise HTTPException(500, f"Malformed HF response: {data}")

async def provider_hf_inference_stream(client: httpx.AsyncClient, model: PublicModel, messages: List[Dict], max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    """
    Simulate streaming: call sync endpoint and then emit tokens/chunks.
    """
    try:
        start_time = datetime.now()
        text = await provider_hf_inference_chat(client, model, messages, max_tokens, temperature)
        # naive tokenization by spaces to simulate stream
        token_count = 0
        for piece in text.split(" "):
            if piece:
                token_count += 1
                await asyncio.sleep(0)  # yield to loop
                yield f"event: token\ndata: {json.dumps({'token': piece + ' '})}\n\n"
        response_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ HuggingFace stream completed | Tokens: {token_count} | Time: {response_time:.2f}s")
        yield "data: [DONE]\n\n"
    except HTTPException as e:
        logger.error(f"❌ HuggingFace stream error: {e.detail}")
        yield f"event: error\ndata: {json.dumps({'detail': str(e.detail)})}\n\n"

# -----------------------------
# HTTP client dependency
# -----------------------------
async def get_http_client():
    async with httpx.AsyncClient(timeout=120) as client:
        yield client

# -----------------------------
# App bootstrap
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FastAPI LLM Chatbot starting up...")
    print(f"🧩 Models: {[m.label for m in ALLOWED_MODELS]}")
    logger.info("🔧 Response logging is enabled")
    yield
    print("🛑 Shutdown")

app = FastAPI(title="LLM Chatbot API (Multi-LLM)", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# -----------------------------
# Health & Models
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "LLM Chatbot API is running",
        "default_model_id": DEFAULT_MODEL_ID,
        "default_provider": DEFAULT_PROVIDER,
        "active_sessions": len(chat_sessions),
    }

@app.get("/models", response_model=List[PublicModel])
async def list_models():
    return ALLOWED_MODELS

# -----------------------------
# Core Chat (non-stream)
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, client: httpx.AsyncClient = Depends(get_http_client)):
    start_time = datetime.now()
    
    sid = get_or_create_session(req.session_id)
    history = req.messages or chat_sessions.get(sid, [])
    llm_messages = build_conversation_context(req.message, history)

    # resolve model
    model = resolve_model(req.model_id, req.provider)
    logger.info(f"🔄 Processing chat request | Session: {sid} | Model: {model.id} ({model.provider})")

    # route to provider
    if model.provider == "openrouter":
        text = await provider_openrouter_chat(client, model, llm_messages, req.max_tokens or 1000, req.temperature or 0.7)
    elif model.provider == "hf_inference":
        text = await provider_hf_inference_chat(client, model, llm_messages, req.max_tokens or 1000, req.temperature or 0.7)
    else:
        raise HTTPException(400, f"Unknown provider: {model.provider}")

    # persist fallback memory
    if not req.messages:
        add_message_to_session(sid, Message(role="user", content=req.message))
        add_message_to_session(sid, Message(role="assistant", content=text))

    total = len(history) + 2
    response_time = (datetime.now() - start_time).total_seconds()
    
    # Log the response
    log_chat_response(sid, req.message, text, model.id, model.provider, response_time)
    
    return ChatResponse(response=text, session_id=sid, message_count=total, model_id=model.id, provider=model.provider)

# -----------------------------
# Streaming Chat (SSE)
# -----------------------------
async def _dispatch_stream(client: httpx.AsyncClient, model: PublicModel, messages: List[Dict], max_tokens: int, temperature: float):
    if model.provider == "openrouter":
        async for chunk in provider_openrouter_stream(client, model, messages, max_tokens, temperature):
            yield chunk
    elif model.provider == "hf_inference":
        async for chunk in provider_hf_inference_stream(client, model, messages, max_tokens, temperature):
            yield chunk
    else:
        yield f"event: error\ndata: {json.dumps({'detail':'Unknown provider'})}\n\n"

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, client: httpx.AsyncClient = Depends(get_http_client)):
    start_time = datetime.now()
    
    sid = get_or_create_session(req.session_id)
    history = req.messages or chat_sessions.get(sid, [])
    llm_messages = build_conversation_context(req.message, history)
    model = resolve_model(req.model_id, req.provider)
    
    # Log stream start
    log_stream_start(sid, req.message, model.id, model.provider)
    
    collected_tokens = []
    token_count = 0

    async def gen():
        nonlocal token_count
        if not req.messages:
            add_message_to_session(sid, Message(role="user", content=req.message))
        
        async for sse in _dispatch_stream(client, model, llm_messages, req.max_tokens or 1000, req.temperature or 0.7):
            # Count and collect tokens for logging
            if sse.startswith("event: token"):
                try:
                    # Extract token from SSE data
                    data_part = sse.split("data: ")[1].strip()
                    token_data = json.loads(data_part)
                    token = token_data.get("token", "")
                    if token:
                        collected_tokens.append(token)
                        token_count += 1
                except:
                    pass
            
            yield sse
            
        # Log completion
        response_time = (datetime.now() - start_time).total_seconds()
        full_response = "".join(collected_tokens).strip()
        
        # Save to session if needed
        if not req.messages and full_response:
            add_message_to_session(sid, Message(role="assistant", content=full_response))
        
        # Log the complete streaming response
        log_chat_response(sid, req.message, full_response, model.id, model.provider, response_time)
        log_stream_complete(sid, token_count, response_time)
        
        # meta
        total = len(history) + 2
        yield f"event: meta\ndata: {json.dumps({'session_id': sid, 'message_count': total, 'model_id': model.id, 'provider': model.provider})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

# -----------------------------
# Sessions (same as before)
# -----------------------------
@app.get("/sessions", response_model=List[SessionInfo])
async def get_sessions():
    out = []
    for sid, meta in session_metadata.items():
        out.append(SessionInfo(session_id=sid, message_count=len(chat_sessions.get(sid, [])), created_at=meta["created_at"], last_updated=meta["last_updated"]))
    return out

@app.get("/sessions/{session_id}/history", response_model=List[Message])
async def get_session_history(session_id: str):
    if session_id not in chat_sessions: raise HTTPException(404, "Session not found")
    return chat_sessions[session_id]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in chat_sessions: raise HTTPException(404, "Session not found")
    del chat_sessions[session_id]; del session_metadata[session_id]
    logger.info(f"🗑️  Session deleted: {session_id}")
    return {"message": f"Session {session_id} deleted"}

@app.delete("/sessions")
async def clear_all_sessions():
    n = len(chat_sessions); chat_sessions.clear(); session_metadata.clear()
    logger.info(f"🗑️  All sessions cleared: {n} sessions")
    return {"message": f"Cleared {n} sessions"}

@app.post("/sessions/{session_id}/clear")
async def clear_session_history(session_id: str):
    if session_id not in chat_sessions: raise HTTPException(404, "Session not found")
    cnt = len(chat_sessions[session_id]); chat_sessions[session_id]=[]; session_metadata[session_id]["last_updated"]=datetime.now()
    logger.info(f"🗑️  Session history cleared: {session_id} ({cnt} messages)")
    return {"message": f"Cleared {cnt} messages from session {session_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)