from langgraph.graph import StateGraph, START, END
from app.agent.state import RAGState
from app.agent.nodes import (
    Initial_Chunks,
    Planner,
    planner_router,
    search_tool,
    Hallucination_Check,
    Hallucination_Check_router,
    generate_answer,
)

# Build the graph once at import time — reused across all requests
graph = StateGraph(RAGState)

graph.add_node("initial_retrieval", Initial_Chunks)
graph.add_node("Planner", Planner)
graph.add_node("search", search_tool)
graph.add_node("Hallucination checker", Hallucination_Check)
graph.add_node("Answer generation", generate_answer)

graph.add_edge(START, "initial_retrieval")
graph.add_edge("initial_retrieval", "Planner")

graph.add_conditional_edges(
    "Planner",
    planner_router,
    {
        "not enough": "search",
        "evident": "Answer generation",
    },
)

graph.add_edge("Answer generation", "Hallucination checker")
graph.add_edge("search", "Planner")

graph.add_conditional_edges(
    "Hallucination checker",
    Hallucination_Check_router,
    {
        "factual": END,
        "repair": "Planner",
    },
)

# Compiled app — single instance, thread-safe for concurrent requests
rag_app = graph.compile()
