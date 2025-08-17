from typing import TypedDict, List, Optional, Dict, Any, Tuple, Literal
import json
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from .utils import (
    llm, extract_json, professional_clarification, interpret_consent,
    list_pdf_paths
)
from .agents import run_agent1_summarize, run_agent2_query, run_agent3_internet

# ---------- Shared State ----------
class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: Optional[str]
    resolved_query: Optional[str] 
    task: str
    document: Optional[str]
    manager_decision: Optional[dict]
    selected_agents: List[str]
    user_consent_internet: Optional[bool]
    user_reply: Optional[str]
    awaiting_consent: Optional[bool]
    final: Optional[dict]
    manager_history: List[dict]
    agent_outcomes: List[dict]

# ---------- LLM Routing ----------
class RouteDecision(BaseModel):
    selected: List[Literal["summarize", "query", "internet"]]
    justification: str

_llm_route = llm.with_structured_output(RouteDecision)

def _llm_route_decision(query: str, pdf_count: int) -> dict:
    system = (
        "You are a Manager Agent that routes a user request to one or more agents, in order:\n"
        "- summarize: Summarize the content of exactly one local PDF.\n"
        "- query: Answer a question using one or more local PDFs (RAG).\n"
        "- internet: Answer using web search (DuckDuckGo).\n\n"
        "Return a structured decision with:\n"
        "selected: array of one or more of ['summarize','query','internet'] in execution order\n"
        "justification: a clear, professional explanation focusing on user intent and the tasks needed.\n"
        "Do not mention the number of PDFs in your justification."
    )
    user = (
        f"USER_QUERY: {query}\n"
        f"PDF_COUNT_IN_DATA_DIR: {pdf_count}\n\n"
        "Guidance:\n"
        "- If the user wants a summary of a local PDF and pdf_count = 1, include 'summarize'.\n"
        "- If they want answers based on document(s) and pdf_count >=1 , include 'query'.\n"
        "- Include 'internet' only if the user wants latest info.\n"
        "- Your justification should only discuss intent and task choice, not counts."
    )
    try:
        decision: RouteDecision = _llm_route.invoke([HumanMessage(content=system), HumanMessage(content=user)])
        return {"selected": decision.selected, "justification": decision.justification}
    except Exception:
        if pdf_count == 0:
            return {"selected": ["internet"], "justification": "Fallback: external info likely needed."}
        if pdf_count == 1:
            return {"selected": ["summarize", "query"], "justification": "Fallback: summarize then answer from the PDF."}
        return {"selected": ["query"], "justification": "Fallback: answer from local PDFs."}

class MemoryRewrite(BaseModel):
    rewritten_query: str = Field(..., description="Single, concise query incorporating prior context if needed.")
    used_memory: bool = Field(..., description="True if previous context was used.")
    rationale: str = Field(..., description="Brief reason for rewrite (for logging/debug).")

_llm_memory_rewrite = llm.with_structured_output(MemoryRewrite)

def _rewrite_with_memory(prev_query: Optional[str], curr_query: str) -> MemoryRewrite:
    """
    If curr_query appears to depend on previous turn (pronouns, ellipses),
    rewrite into a self-contained query using prev_query as context.
    Otherwise, return curr_query as-is.
    """
    prev = (prev_query or "").strip()
    curr = (curr_query or "").strip()

    # Quick heuristic to decide if we *likely* need memory
    hint_words = [" it ", " its ", " that ", " this ", " they ", " them ", " those ", " these ", " such "]
    needs_memory = any(w in f" {curr.lower()} " for w in hint_words) or len(curr.split()) <= 4

    system = (
        "You rewrite the user's current message into a single, self-contained query. "
        "If the current message references a previous question or uses pronouns (it/this/that/they/etc.), "
        "incorporate the prior question's context. Keep it short, factual, and free of hallucinated constraints."
    )
    user = (
        f"PREVIOUS_QUESTION:\n{prev or '(none)'}\n\n"
        f"CURRENT_MESSAGE:\n{curr}\n\n"
        "Return JSON fields:\n"
        "- rewritten_query: one line\n"
        "- used_memory: true/false\n"
        "- rationale: brief"
    )
    try:
        if needs_memory and prev:
            return _llm_memory_rewrite.invoke([HumanMessage(content=system), HumanMessage(content=user)])
        else:
            # No rewrite necessary
            return MemoryRewrite(rewritten_query=curr, used_memory=False, rationale="No prior context needed.")
    except Exception:
        # Safe fallback
        return MemoryRewrite(rewritten_query=curr, used_memory=False, rationale="LLM rewrite failed; used current as-is.")
    
def _apply_guardrails(selection: List[str], pdf_count: int) -> Tuple[List[str], List[str]]:
    adjustments: List[str] = []
    ordered, seen = [], set()
    for s in selection:
        if s not in seen:
            ordered.append(s); seen.add(s)

    if pdf_count == 0:
        filtered = [s for s in ordered if s == "internet"]
        if not filtered:
            adjustments.append("Added 'internet' because no local PDFs are available.")
            filtered = ["internet"]
        return filtered, adjustments

    if pdf_count == 1:
        return [s for s in ordered if s in ("summarize", "query", "internet")], adjustments

    filtered = [s for s in ordered if s in ("query", "internet")]
    if not filtered:
        filtered, adj = ["query"], "Replaced 'summarize' with 'query' for multiple PDFs."
        adjustments.append(adj)
    return filtered, adjustments

# ---------- Nodes ----------
def manager_node(state: AgentState):
    state.setdefault("messages", [])
    state.setdefault("manager_history", [])
    state.setdefault("agent_outcomes", [])
    state.setdefault("selected_agents", [])
    state.setdefault("awaiting_consent", False)
    state.setdefault("user_consent_internet", None)
    state.setdefault("user_reply", None)
    state.setdefault("task", "")
    state.setdefault("document", None)
    state.setdefault("manager_decision", None)
    state.setdefault("resolved_query", None)   # <-- NEW

    # Sticky during consent handshake
    if state.get("awaiting_consent") and state.get("selected_agents"):
        return state

    # ---- memory-aware rewrite (NEW) ----
    query = state.get("query") or ""
    prev_query = None
    if state["agent_outcomes"]:
        prev_query = state["agent_outcomes"][-1].get("resolved_query") or state["agent_outcomes"][-1].get("query")

    mem_rewrite = _rewrite_with_memory(prev_query, query)
    resolved_query = mem_rewrite.rewritten_query.strip() or query
    state["resolved_query"] = resolved_query

    pdf_count = len(list_pdf_paths())

    # Base choice
    llm_decision = _llm_route_decision(resolved_query, pdf_count)
    selection_raw = llm_decision.get("selected", [])
    justification = llm_decision.get("justification", "")

    # Guardrails
    selected, adjustments = _apply_guardrails(selection_raw, pdf_count)

    # Memory nudges (unchanged)
    outcomes = state.get("agent_outcomes", [])
    if outcomes:
        last = outcomes[-1]
        rag_failed = (last.get("agent2_answerable") is False)
        prior_consent = state.get("user_consent_internet") is True or (
            state.get("user_reply") is not None and interpret_consent(state.get("user_reply")) is True
        )
        if rag_failed and prior_consent and "internet" not in selected:
            selected = selected + ["internet"]
            adjustments.append("Memory nudge: RAG insufficient earlier; user had consented to web. Added 'internet'.")

        user_refused = state.get("user_consent_internet") is False or (
            state.get("user_reply") is not None and interpret_consent(state.get("user_reply")) is False
        )
        if user_refused:
            selected = [s for s in selected if s != "internet"]
            adjustments.append("Memory nudge: user declined web previously; removed 'internet'.")

    decision_blob = {
        "selected": selection_raw,
        "final_selected": selected,
        "justification": justification,
        "adjustments": adjustments,
        "query": query,
        "resolved_query": resolved_query,                      # <-- NEW (logged)
        "memory_rewrite": {                                    # <-- NEW (logged)
            "used_memory": mem_rewrite.used_memory,
            "rationale": mem_rewrite.rationale,
        },
    }

    state["selected_agents"] = selected
    state["task"] = ",".join(selected) if selected else ""
    state["manager_decision"] = decision_blob

    hist = state.get("manager_history", [])
    hist.append(decision_blob)
    state["manager_history"] = hist

    state["messages"].append(HumanMessage(content=json.dumps({"manager_decision": decision_blob})))
    return state

def executor_node(state: AgentState):
    state.setdefault("messages", [])
    state.setdefault("manager_history", [])
    state.setdefault("agent_outcomes", [])
    state.setdefault("awaiting_consent", False)

    selected = state.get("selected_agents", [])
    # Prefer resolved_query if present
    query = state.get("resolved_query") or state.get("query") or ""
    agent_responses: Dict[str, Any] = {}
    ran_internet = False
    needs_internet_after_rag = False

    for agent in selected:
        if agent == "summarize":
            res = run_agent1_summarize(query)
            agent_responses["Agent1"] = res
        elif agent == "query":
            res = run_agent2_query(query)
            agent_responses["Agent2"] = res
            data = res.get("data", {})
            if not data.get("answerable", True):
                needs_internet_after_rag = True
        elif agent == "internet":
            res = run_agent3_internet(query)
            agent_responses["Agent3"] = res
            ran_internet = True

    if needs_internet_after_rag and not ran_internet:
        consent = state.get("user_consent_internet")
        if consent is None:
            from .utils import interpret_consent as _interp
            consent = _interp(state.get("user_reply"))

        if consent is True:
            res = run_agent3_internet(query)
            agent_responses["Agent3"] = res
            state["awaiting_consent"] = False
            ran_internet = True
        elif consent is False:
            agent_responses["Agent3"] = {
                "status": "200",
                "data": {"user_query": query,
                         "message": "Understood. I won’t use the browser. If you change your mind, say “yes” to proceed with a web search."}
            }
            state["awaiting_consent"] = False
        else:
            agent_responses["Agent3"] = {
                "status": "202",
                "data": {"user_query": query,
                         "message": "I couldn’t answer from the PDFs. Use the browser? Reply “yes” to proceed or “no” to skip."}
            }
            state["awaiting_consent"] = True

    final_payload = {
        "user_query": state.get("query") or "",
        "resolved_query": query,   # <-- surface the normalized query in the API response
        "manager_agent": {
            "decision": state.get("manager_decision", {}).get("justification", ""),
            "selected_agents": state.get("manager_decision", {}).get("final_selected", []),
            "adjustments": state.get("manager_decision", {}).get("adjustments", []),
            "awaiting_consent": state.get("awaiting_consent", False),
        },
        "agent_responses": agent_responses,
    }
    state["final"] = final_payload
    state["messages"].append(HumanMessage(content=json.dumps(final_payload, ensure_ascii=False)))

    outcomes = state.get("agent_outcomes", [])
    outcomes.append({
        "query": state.get("query") or "",
        "resolved_query": query,               # <-- store for next turn
        "agents_run": list(selected),
        "awaiting_consent": state.get("awaiting_consent", False),
        "agent2_answerable": agent_responses.get("Agent2", {}).get("data", {}).get("answerable"),
        "agent3_used": "Agent3" in agent_responses and agent_responses["Agent3"]["status"] == "200",
    })
    state["agent_outcomes"] = outcomes
    return state

def _route_from_manager(_state: AgentState):
    return "executor_node"

# ---------- Graph Manager Class ----------
class GraphManager:
    def __init__(self):
        from langgraph.graph import StateGraph
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("manager_node", manager_node)
        self.workflow.add_node("executor_node", executor_node)

        self.workflow.set_entry_point("manager_node")
        self.workflow.add_conditional_edges("manager_node", _route_from_manager)
        self.workflow.add_edge("executor_node", END)

        # Short-term memory checkpointer
        self.checkpointer = InMemorySaver()
        self.compiled = self.workflow.compile(checkpointer=self.checkpointer)

    def invoke(self,
               thread_id: str,
               query: Optional[str] = None,
               user_reply: Optional[str] = None,
               user_consent_internet: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        Invoke the graph with a delta-state so prior memory (for this thread_id)
        is loaded and merged. Returns the final payload dict.
        """
        delta: Dict[str, Any] = {}
        if query is not None:
            delta["query"] = query
        if user_reply is not None:
            delta["user_reply"] = user_reply
        if user_consent_internet is not None:
            delta["user_consent_internet"] = user_consent_internet

        config = {"configurable": {"thread_id": thread_id}}
        result_state = self.compiled.invoke(delta, config=config)
        return result_state.get("final")
