from fastapi import FastAPI, Body
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from llm.openai_llm import AzureOpenAILLM
from llm.openRouter import openrouter
from agents.intent_router import IntentLLMAgent
from agents.nl2sql_agant import NL2SQLAgent   # ensure filename is *agent*, not *agant*
import os, traceback, json

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

app = FastAPI(title="Opportunity Chatbot")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,  # set True only if you use cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== API models for UI ====
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str = Field(..., description="Message text")
    ts: Optional[str] = Field(None, description="ISO8601 timestamp (optional)")

class ChatRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    history: List[ChatMessage] = Field(default_factory=list, description="Ordered chat history")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata from UI")

class ChatResponse(BaseModel):
    type: str = Field(..., description="Response kind (analysis, small_talk, db_error, etc.)")
    intent: Optional[Dict[str, Any]] = None
    input: Optional[str] = None
    result: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    source: Optional[str] = None
    params: Optional[List[Any]] = None
    chart_path: Optional[str] = None
    used_retrieval: Optional[bool] = None
    error: Optional[str] = None
    trace: Optional[str] = None
# ============================


# Utility endpoints for UI team
@app.get("/healthz")
def healthz():
    return {"ok": True}

# @app.get("/api-def")
# def api_def():
#     return {
#         "request_schema": ChatRequest.model_json_schema(),
#         "response_schema": ChatResponse.model_json_schema(),
#         "notes": "Use POST /chat/{session_id} with ChatRequest payload."
#     }

# llm = AzureOpenAILLM()
llm = openrouter()
intent_agent = IntentLLMAgent(llm)
nl2sql_agent = NL2SQLAgent(llm)

@app.post("/chat/{session_id}", response_model=ChatResponse)
def chat(session_id: str, payload: ChatRequest = Body(...)):

    q = payload.query or ""
    history = [m.model_dump() for m in (payload.history or [])]
    # Standardized context passed into agents
    ctx: Dict[str, Any] = {
        "session_id": session_id,
        "history": history,
        "meta": payload.meta or {}
    }

    try:
        try:
            intent = intent_agent.run(q, ctx)  # preferred (with context/history)
        except TypeError:
            # Backward compatibility if IntentLLMAgent doesn't accept context
            intent = intent_agent.run(q)
        label = (intent.get("label") or "").strip().upper()
        if DEBUG:
            print("[chat] message:", q)
            print("[chat] intent:", intent)

        # Fallback if LLM returned something unexpected
        if label not in {"DB_QUERY", "SMALL_TALK"}:
            label = "DB_QUERY" if any(k in q.lower() for k in ("opportunity","item","supplier")) else "SMALL_TALK"

        if label == "DB_QUERY":
            res = nl2sql_agent.run(q, ctx)
        else:
            res = {"type": "small_talk", "result": "Ask me about opportunities, items, suppliers, etc."}

        if DEBUG:
            print("[chat] result:", json.dumps(res, default=str) if res else "<None>")

        # NEVER return None/empty
        if not res:
            return {"type": "internal_empty", "message": "agent is busy please try again ", "intent": intent, "input": q}

        # Ensure we always include intent + input in the response for UI
        merged = dict(res or {})
        merged.setdefault("type", "analysis")
        merged["intent"] = intent
        merged["input"] = q
        return merged

    except Exception as e:
        if DEBUG:
            return {"type": "error", "error": str(e), "trace": traceback.format_exc(), "input": q}
        return {"type": "error", "error": str(e)}

   