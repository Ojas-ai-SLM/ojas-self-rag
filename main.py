import os
import shutil
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List

from rag_pipeline import build_graph
from vector_store import add_documents_to_index, get_embeddings, load_or_build_index

load_dotenv()

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

state = {}
executor = ThreadPoolExecutor(max_workers=1)


def _blocking_init():
    """All heavy synchronous work runs here in a thread — never blocks the event loop."""
    print("\n[1/4] Loading HuggingFace embeddings...")
    embeddings = get_embeddings()
    print("  [1/4] Done.")

    print("\n[2/4] Loading FAISS vector store...")
    store = load_or_build_index(embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 5})
    print("  [2/4] Done.")

    print("\n[3/4] Initializing Groq LLM...")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is not set!")
    llm = ChatGroq(
        api_key=groq_key,
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
    )
    print("  [3/4] Done.")

    print("\n[4/4] Compiling LangGraph pipeline...")
    graph = build_graph(retriever, llm)
    print("  [4/4] Done.")

    return embeddings, store, llm, graph


async def initialize_pipeline():
    """Offload all blocking work to a thread so the event loop stays free."""
    try:
        loop = asyncio.get_event_loop()
        embeddings, store, llm, graph = await loop.run_in_executor(
            executor, _blocking_init
        )
        state["store"] = store
        state["embeddings"] = embeddings
        state["graph"] = graph
        state["llm"] = llm
        state["ready"] = True
        print("\n✓ OJAS.AI pipeline is ready!\n")
    except Exception as e:
        print(f"\n✗ Pipeline FAILED: {e}")
        traceback.print_exc()
        state["ready"] = False
        state["error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("OJAS.AI — Server started, loading pipeline...")
    print("=" * 50)
    asyncio.create_task(initialize_pipeline())
    yield
    executor.shutdown(wait=False)
    print("OJAS.AI — Shutting down")


app = FastAPI(
    title="Ojas.ai Self-RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str

class SourceInfo(BaseModel):
    document: str
    page: int

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []


@app.get("/health")
async def health():
    if state.get("error"):
        return {"status": "error", "pipeline_ready": False, "error": state["error"]}
    return {"status": "ok" if state.get("ready") else "loading", "pipeline_ready": state.get("ready", False)}


# --- Guardrail patterns (simple checks before LLM) ---
GREETINGS = [
    "hi", "hello", "hey", "hii", "hiii", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "whats up", "what's up", "sup", "namaste", "namaskar",
]

OFF_TOPIC_KEYWORDS = [
    "cricket", "football", "movie", "film", "stock", "crypto",
    "bitcoin", "coding", "programming", "python", "javascript",
    "recipe", "cook", "politics", "election", "war",
]

ABOUT_OJAS = [
    "who are you", "what are you", "what is ojas", "tell me about yourself",
    "what can you do", "help", "how does this work", "what do you know",
]

def get_guardrail_response(question: str):
    """Simple pattern-based guardrails for greetings, about, and off-topic.

    Note: Harmful query detection is handled by the LLM-based safety_check node
    in the graph pipeline for more robust and flexible filtering.
    """
    q = question.lower().strip().rstrip("?!")

    # Greeting
    if q in GREETINGS or any(q.startswith(g) for g in GREETINGS):
        return (
            "Namaste! 🙏 I am Ojas.ai, your Ayurvedic wellness assistant. "
            "I can help you with:\n"
            "- Information about Ayurvedic herbs and their uses\n"
            "- Understanding your dosha (Vata, Pitta, Kapha)\n"
            "- Ayurvedic remedies for common ailments\n"
            "- Diet and lifestyle recommendations\n\n"
            "How can I assist you on your wellness journey today?"
        )

    # About Ojas
    if any(phrase in q for phrase in ABOUT_OJAS):
        return (
            "I am Ojas.ai, an AI assistant specialized in Ayurveda — the ancient Indian science of life. "
            "My knowledge comes from classical Ayurvedic texts including the Charaka Samhita, "
            "Sushruta Samhita, and other traditional sources.\n\n"
            "I can answer questions about herbs, doshas, remedies, diet, and Ayurvedic treatments. "
            "Please consult a qualified Ayurvedic practitioner for personalized medical advice."
        )

    # Off-topic (harmless but not Ayurveda-related)
    if any(keyword in q for keyword in OFF_TOPIC_KEYWORDS):
        return (
            "I specialize in Ayurveda and holistic wellness. "
            "I am not able to help with that topic, but I would be happy to answer any questions "
            "about Ayurvedic herbs, remedies, doshas, or natural wellness. "
            "What Ayurvedic topic can I help you with? Can you please ask an Ayurveda-related question?"
        )

    return None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if state.get("error"):
        raise HTTPException(status_code=500, detail=f"Pipeline error: {state['error']}")
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Pipeline is still loading, try again in a moment.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Check guardrails before hitting the RAG pipeline
    guardrail_response = get_guardrail_response(req.question)
    if guardrail_response:
        return ChatResponse(answer=guardrail_response)

    initial = {
        "question": req.question,
        # Safety check
        "is_safe": True,
        "safety_response": "",
        # Retrieval
        "retrieval_query": "",
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "sources": [],
        "issup": "",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
    }

    result = await asyncio.get_event_loop().run_in_executor(
        executor,
        lambda: state["graph"].invoke(initial, config={"recursion_limit": 80})
    )

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", [])
    )


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    save_path = DOCS_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks_added = add_documents_to_index(str(save_path), state["store"], state["embeddings"])
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {str(e)}")

    return {"message": f"Uploaded and indexed '{file.filename}'", "chunks_added": chunks_added}


@app.get("/documents")
async def list_documents():
    pdfs = sorted([f.name for f in DOCS_DIR.glob("*.pdf")])
    return {"documents": pdfs, "count": len(pdfs)}