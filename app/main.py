import uuid
from agent.graph import return_app  # your compiled graph

async def run_rag_query(query: str, user_id: uuid.UUID) -> str:
    initial_state = {
        "user_id": user_id,
        "query": query,
        "messages": [],

        "chunks": [],
        "search": [],

        "planner_state": "not_enough",   # harmless placeholder, planner overwrites it
        "retrieval_queries": [],
        "wiki_queries": [],
        "tavily_queries": [],

        "executed_retrieval_queries": [],
        "executed_wiki_queries": [],
        "executed_tavily_queries": [],

        "answer": "",

        "need_repair": "repair",         # placeholder, hallucination checker overwrites it
        "hallucination_reason": [],

        "max_tries_planner": 0,
        "max_tries_hallucinator": 0,
    }
    app=return_app()
    result = await app.ainvoke(initial_state)
    return result["answer"]


# Example call site (e.g. in a FastAPI route)
if __name__ == "__main__":
    import asyncio
    answer = asyncio.run(run_rag_query(
        query="What is our company's leave policy?",
        user_id=uuid.UUID("...")
    ))
    print(answer)