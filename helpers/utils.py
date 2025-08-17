import os
import re
import glob
import json
import warnings
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pypdf.errors import PdfReadWarning
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults

warnings.filterwarnings("ignore", category=PdfReadWarning)
warnings.filterwarnings(
    "ignore",
    message="This package (`duckduckgo_search`) has been renamed to `ddgs`!",
    category=RuntimeWarning,
    module="langchain_community.utilities.duckduckgo_search",
)

# --- Config / Globals ---
load_dotenv()
DATA_DIR = "./data"
FAISS_DIR = "./faiss_bge_index"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

faiss_store: Optional[FAISS] = None

# --- Data loading ---
def list_pdf_paths(folder: str = DATA_DIR) -> List[str]:
    os.makedirs(folder, exist_ok=True)
    return sorted(glob.glob(os.path.join(folder, "*.pdf")))

def load_pdf_as_documents(path: str) -> List[Document]:
    return PyPDFLoader(path).load()

def load_all_pdfs_as_documents(folder: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []
    for pdf in list_pdf_paths(folder):
        try:
            docs.extend(load_pdf_as_documents(pdf))
        except Exception as e:
            print(f"[WARN] Could not read {pdf}: {e}")
    return docs

def combine_documents_text(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

# --- FAISS init / rebuild (robust) ---
def init_faiss(docs: List[Document], force_rebuild: bool = False):
    global faiss_store

    def _split_docs(d: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        return splitter.split_documents(d) if d else []

    def _build_from_docs(d: List[Document]) -> FAISS:
        chunks = _split_docs(d)
        if not chunks:
            raise ValueError("No text chunks were produced from documents.")
        texts = [c.page_content for c in chunks]
        metas = [getattr(c, "metadata", {}) for c in chunks]
        vs = FAISS.from_texts(texts=texts, embedding=hf_embeddings, metadatas=metas)
        os.makedirs(FAISS_DIR, exist_ok=True)
        vs.save_local(folder_path=FAISS_DIR)
        return vs

    if force_rebuild:
        if not docs:
            raise ValueError("force_rebuild=True but no docs provided.")
        faiss_store = _build_from_docs(docs)
        return

    if os.path.exists(FAISS_DIR):
        try:
            faiss_store = FAISS.load_local(
                folder_path=FAISS_DIR,
                embeddings=hf_embeddings,
                allow_dangerous_deserialization=True,
            )
            return
        except Exception as e:
            if docs:
                print(f"[FAISS] Load failed ({e}). Rebuilding index…")
                faiss_store = _build_from_docs(docs)
                return
            raise RuntimeError("FAISS load failed and no docs to rebuild.") from e
    else:
        if not docs:
            raise ValueError("No FAISS index and no docs to create one.")
        faiss_store = _build_from_docs(docs)

# --- Small helpers ---
def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {"response": cleaned}

def looks_uncertain(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in [
        "i don't know", "i do not know", "cannot answer", "not sure",
        "no information", "insufficient information", "lack of information"
    ])

def professional_clarification(query: str) -> str:
    return (
        "I’m not confident I can answer that from the current results. "
        f"Could you clarify what you’d like to know about “{query}”, or rephrase the question? "
        "For example, specify the aspect, time frame, or context."
    )

def interpret_consent(reply: Optional[str]) -> Optional[bool]:
    if not reply:
        return None
    r = reply.strip().lower()
    yes = {
        "yes", "y", "yeah", "yep", "sure", "ok", "okay", "please do",
        "go ahead", "proceed", "yes please", "sure thing", "alright", "aye"
    }
    no = {"no", "n", "nope", "don't", "do not", "not now", "stop", "cancel", "no thanks", "no thank you"}
    if r in yes:
        return True
    if r in no:
        return False
    if any(p in r for p in ["go ahead", "proceed", "please do", "yes"]):
        return True
    if any(p in r for p in ["don't", "do not", "not now", "stop", "cancel", "no"]):
        return False
    return None

# --- Web Query Rewrite ---
class WebQueryRewrite(BaseModel):
    search_query: str = Field(..., description="Optimized web search query")
    rationale: str = Field(..., description="Brief reasoning for debugging")

_llm_rewrite = llm.with_structured_output(WebQueryRewrite)

def rewrite_query_for_web(user_query: str) -> WebQueryRewrite:
    system = (
        "Rewrite the user's text into a single, concise web search query suitable for DuckDuckGo/Google. "
        "Prefer key entities, optional helpful operators (quotes, site:, intitle:, filetype:) when clearly useful; "
        "avoid hallucinated constraints; keep it short."
    )
    user = f'Original query: "{user_query}"\nRewrite it into one optimal search query.'
    try:
        return _llm_rewrite.invoke([HumanMessage(content=system), HumanMessage(content=user)])
    except Exception:
        return WebQueryRewrite(search_query=user_query, rationale="LLM rewrite failed; used original query.")

# --- DDG helper (thin wrapper) ---
def ddg_search(query: str) -> List[Dict[str, Any]]:
    tool = DuckDuckGoSearchResults(output_format="list")
    try:
        results = tool.invoke(query)
        return results if isinstance(results, list) else []
    except Exception:
        return []
