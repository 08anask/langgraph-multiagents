from typing import Dict, Any, List, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Literal

from . import utils as U
from .utils import (
    llm,
    extract_json, looks_uncertain, professional_clarification,
    rewrite_query_for_web, ddg_search,
    list_pdf_paths, load_pdf_as_documents, load_all_pdfs_as_documents, combine_documents_text
)

# ---------- Agent 1: Summarize single PDF ----------
def run_agent1_summarize(query_text: Optional[str]) -> Dict[str, Any]:
    pdfs = list_pdf_paths()
    if len(pdfs) != 1:
        return {"status": "400", "data": {"user_query": query_text or "", "error": f"Agent1 expects exactly one PDF in ./data, found {len(pdfs)}."}}

    docs = load_pdf_as_documents(pdfs[0])
    doc_text = combine_documents_text(docs)
    prompt = (
        "Summarize in 3–4 sentences and extract the top 5 keywords.\n\n"
        f"{doc_text}\n\n"
        "Respond strictly in JSON with keys: summary, keywords."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    parsed = extract_json(resp.content if isinstance(resp.content, str) else "")
    return {"status": "200", "data": {"user_query": query_text or "", **parsed}}

# ---------- Agent 2: RAG over PDFs ----------
def run_agent2_query(query_text: Optional[str]) -> Dict[str, Any]:
    q = (query_text or "").strip()
    docs = load_all_pdfs_as_documents()
    if not docs:
        return {"status": "400", "data": {"user_query": q, "error": "Agent2 expected PDFs in ./data but found none."}}

    # Build/load index; this will set U.faiss_store
    U.init_faiss(docs)

    if U.faiss_store is None:
        return {"status": "500", "data": {"user_query": q, "error": "FAISS store not initialized."}}

    # Prefer scored retrieval where available
    retrieved, scores = [], []
    try:
        results = U.faiss_store.similarity_search_with_score(q, k=4)
        for d, s in results:
            retrieved.append(d.page_content)
            scores.append(s)
    except Exception:
        results = U.faiss_store.similarity_search(q, k=4)
        retrieved = [d.page_content for d in results]
        scores = []

    has_docs = len(retrieved) > 0
    strong_evidence = has_docs and (not scores or (min(scores) <= 1.0))

    prompt = (
        "You are given a user query and snippets retrieved from local PDFs. "
        "Answer the query using ONLY those snippets. "
        "If the snippets do not contain the necessary information, DO NOT invent an answer. "
        "Provide a brief, professional clarification asking the user to refine the question.\n\n"
        f"QUERY:\n{q}\n\n"
        f"RETRIEVED_SNIPPETS:\n{retrieved}\n\n"
        'Respond STRICTLY in JSON with keys: "response": string, "is_answered": boolean.'
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    parsed = extract_json(resp.content if isinstance(resp.content, str) else "")
    resp_text = (parsed.get("response") or "").strip()
    is_answered = bool(parsed.get("is_answered") is True)

    if looks_uncertain(resp_text) or not resp_text:
        resp_text = professional_clarification(q)
        is_answered = False

    answerable = bool(is_answered and strong_evidence)
    parsed["response"] = resp_text
    parsed["is_answered"] = is_answered
    parsed["answerable"] = answerable

    return {"status": "200", "data": {"user_query": q, **parsed}}

# ---------- Agent 3: Internet (DDG) ----------
def run_agent3_internet(query_text: Optional[str]) -> Dict[str, Any]:
    q_raw = (query_text or "").strip()
    if not q_raw:
        return {"status": "400", "data": {"user_query": "", "response": "Please provide a query to search.", "source": []}}

    # LLM rewrite (web-friendly)
    rewrite = rewrite_query_for_web(q_raw)
    q_rewritten = (rewrite.search_query or q_raw).strip()

    # Search rewritten first, fall back to raw
    results = ddg_search(q_rewritten)
    if not results:
        results = ddg_search(q_raw)

    top = results[:8]
    urls = [r.get("link") for r in top if r.get("link")]
    blocks = []
    for r in top:
        title = r.get("title", "") or ""
        link = r.get("link", "") or ""
        snippet = r.get("snippet", "") or ""
        blocks.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
    evidence = "\n---\n".join(blocks).strip()

    if not evidence:
        return {
            "status": "200",
            "data": {
                "user_query": q_raw,
                "response": professional_clarification(q_raw),
                "source": urls,
                # "rewrite": {"search_query": q_rewritten, "rationale": rewrite.rationale},
            },
        }

    prompt = (
        "Use ONLY the following DuckDuckGo snippets (titles, URLs, brief summaries) to answer the user’s question. "
        "If the needed information is not present, do NOT answer “I don't know.” "
        "Provide a brief, professional clarification asking the user to refine the question.\n\n"
        f"USER QUERY (original):\n{q_raw}\n\n"
        f"REWRITTEN SEARCH QUERY:\n{q_rewritten}\n\n"
        f"DDG SNIPPETS:\n{evidence}\n\n"
        "Respond strictly in JSON with keys: response, source (source is a list of URLs). "
        "Keep the response concise and professional."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    parsed = extract_json(resp.content if isinstance(resp.content, str) else "")
    response_text = parsed.get("response") or (resp.content if isinstance(resp.content, str) else "")
    if looks_uncertain(response_text):
        response_text = professional_clarification(q_raw)
    sources = parsed.get("source") or urls

    return {
        "status": "200",
        "data": {
            "user_query": q_raw,
            "response": response_text,
            "source": list(dict.fromkeys(sources)),
            "rewrite": {"search_query": q_rewritten, "rationale": rewrite.rationale},
        },
    }
