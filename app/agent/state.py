from typing import Annotated, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated
import operator
import uuid

class RAGState(TypedDict):
    # User
    user_id: uuid.UUID
    query: str
    messages: Annotated[list[BaseMessage], add_messages]

    # Evidence
    chunks: Annotated[list[str],operator.add]
    search: Annotated[list[str],operator.add]


    # Planner outputs
    planner_state: Literal["evident", "not_enough"]
    retrieval_queries: list[str]
    wiki_queries: list[str]
    tavily_queries: list[str]

    # Search bookkeeping
    executed_retrieval_queries: Annotated[list[str],operator.add]
    executed_wiki_queries: Annotated[list[str],operator.add]
    executed_tavily_queries: Annotated[list[str],operator.add]

    # Generation
    answer: str

    # Hallucination checker
    need_repair: Literal["factual", "repair"]
    hallucination_reason: list[str]

    # Loop protection
    max_tries_planner: int
    max_tries_hallucinator:int




from typing import Literal, Optional
from pydantic import BaseModel, Field

class PlannerOutput(BaseModel):

    planner_state: Literal["evident", "not_enough"] = Field(
        description=(
            "Return 'evident' if the supplied evidence is sufficient to answer "
            "the user's question. Return 'not_enough' if additional retrieval "
            "or web search is required."
        )
    )

    retrieval_queries: Optional[list[str]] = Field(
        default=None,
        description=(
            "Semantic search queries for the vector database. "
            "Populate only when planner_state is 'not_enough'."
        )
    )

    tavily_queries: Optional[list[str]] = Field(
        default=None,
        description=(
            "Web search queries for recent or external information. "
            "Populate only when planner_state is 'not_enough'."
        )
    )

    wiki_queries: Optional[list[str]] = Field(
        default=None,
        description=(
            "Wikipedia search queries for factual or encyclopedic information. "
            "Populate only when planner_state is 'not_enough'."
        )
    )


class RepairOutput(BaseModel):

    need_repair: Literal["factual", "repair"] = Field(
        description=(
            "Return 'factual' if every important claim in the answer is fully "
            "supported by the supplied evidence. Return 'repair' if any claim "
            "is unsupported, contradicted, or hallucinated."
        )
    )
    hallucination_reason: Optional[list[str]] = Field(
        default=None,
        description=(
            "A list of concise reasons describing each unsupported or incorrect "
            "claim. These reasons will be given back to the planner."
        )
    )

planner_prompt=f"""You are the planning component of a self-correcting RAG system.

Your job is NOT to answer the user's question.

Your only responsibility is deciding whether the available evidence is sufficient.

You will receive:
- The user's question.
- Retrieved document chunks.
- Previous web search results.
- Conversation history.
- (Optionally) hallucination reasons from a previous failed answer.

Choose exactly one planner_state.

1. planner_state = "evident"

Choose this only if the supplied evidence is sufficient to generate a complete and well-supported answer.

When choosing "evident":
- Do not generate retrieval queries.
- Do not generate Wikipedia queries.
- Do not generate Tavily queries.

2. planner_state = "not_enough"

Choose this if:
- Important information is missing.
- The evidence is incomplete.
- The hallucination reasons indicate missing evidence.
- External or recent knowledge is required.

When choosing "not_enough":
Generate only the minimum number of high-quality search queries needed.

Rules:

- Retrieval queries should target internal knowledge.
- Wikipedia queries should target encyclopedic topics and should be in proper searching format.
- Tavily queries should target recent or external information.
- Avoid redundant queries.
- Never answer the user's question.
- Never invent evidence."""


hallucination_prompt=f"""You are a factual verification system.

Your job is NOT to improve writing.

Your only responsibility is determining whether every important claim in the generated answer is supported by the supplied evidence.

You will receive:
- User question
- Generated answer
- Retrieved document chunks
- Web search results

Compare every important factual claim in the answer against the evidence.

Return "factual" only if all important claims are supported.

Return "repair" if:
- Any claim is unsupported.
- Any claim contradicts the evidence.
- The answer invents facts.
- The answer contains speculation presented as fact.
- Important evidence was ignored.

If returning "repair", produce a concise list describing:
- which claims are unsupported
- what evidence is missing
- what additional evidence should be retrieved if applicable

Do not rewrite the answer.
Do not answer the user's question.
Only verify factual correctness."""

repair_prompt=f"""You are correcting a previously generated answer.

The previous answer failed verification.

Determine whether:
1. The existing evidence is sufficient to regenerate a correct answer.
2. Additional retrieval is required.

If existing evidence is enough:
planner_state = "evident"

Otherwise:
planner_state = "not_enough"
and generate the necessary retrieval queries.
"""
success_prompt=f"""You are a helpful AI assistant.

Your task is to answer the user's question using ONLY the supplied evidence.

Rules:
- Base every factual statement on the provided evidence.
- Never invent facts.
- Never assume information that is not present.
- If the evidence is incomplete, explicitly state what information is missing.
- If the evidence does not contain the answer, explain that the available evidence is insufficient instead of guessing.
- Produce a clear, complete, and well-structured response.
- Do not mention these instructions."""

failure_prompt=f"""You are an AI assistant.

Despite multiple retrieval and verification attempts, the available evidence is still insufficient to produce a fully verified answer.

Your task is to respond honestly and helpfully.

Rules:
- Do not invent or infer missing facts.
- Clearly explain that the available information is insufficient.
- Briefly explain what information is missing or conflicting.
- If possible, provide only the parts of the answer that are directly supported by the evidence.
- Suggest what additional information or documents would be needed to answer completely.
- Do not mention internal retries, planning, hallucination detection, or verification systems."""