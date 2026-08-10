from typing import Optional
from pydantic import BaseModel, Field
from app.documents.clients import chroma_client,chat_llm
from app.documents.service import retrieve_chunks,multi_query_retrieval
from pydantic import BaseModel,Field
from app.agent.search_tool import search_tavily,search_wiki
from typing import TypedDict,Optional,Literal,Annotated
import json
import uuid
import asyncio
from langchain_core.messages import (BaseMessage,HumanMessage,AIMessage,ToolMessage,SystemMessage)
from langgraph.graph.message import add_messages
from app.agent.state import RAGState,PlannerOutput,RepairOutput,planner_prompt,hallucination_prompt,repair_prompt,success_prompt,failure_prompt
from app.documents.clients import chroma_client
structured_planner_llm = chat_llm.with_structured_output(PlannerOutput)
structured_repair_llm = chat_llm.with_structured_output(RepairOutput)


async def Initial_Chunks(state:RAGState)->RAGState:
    initial=await retrieve_chunks(state['query'],user_id=state['user_id'],chroma_client=chroma_client,top_k=5)
    result=[]
    for i in initial:
        result.append(i['text'])

    return {
        'chunks':result
    }

async def Planner(state: RAGState) -> dict:
    content=f"""
        User Query:
        {state["query"]}

        chunks:
        {state["chunks"]}

        Search Results:
        {state["search"]}

        """
    if not state['hallucination_reason']:
        messages = [
            SystemMessage(content=planner_prompt),
            HumanMessage(
               content=content
            )
        ]

        response = await structured_planner_llm.ainvoke(messages)

  
    else:
        messages = [
        SystemMessage(content=repair_prompt),
        HumanMessage(
        content= content + f"""
        Previous Answer:
        {state["answer"]}

        Hallucination Reasons:
        {state["hallucination_reason"]}
        """
        )
        ]
        response = await structured_planner_llm.ainvoke(messages)

    return {
        'planner_state':response.planner_state,
        'retrieval_queries': response.retrieval_queries or [],
        'wiki_queries':response.wiki_queries or [],
        'tavily_queries': response.tavily_queries or [],
        'max_tries_planner':state['max_tries_planner']+1
    }

async def planner_router(state:RAGState):
    if state['planner_state']=='evident' or state['max_tries_planner']>5:
        return 'evident' 
    elif state['planner_state']=='not_enough':
        return 'not enough'
    
async def search_tool(state: RAGState):
    wiki_results = await asyncio.gather(*[search_wiki(q) for q in state['wiki_queries']])
    tavily_results = await asyncio.gather(*[search_tavily(q) for q in state['tavily_queries']])
    raw_search_chunks = await multi_query_retrieval(state['retrieval_queries'], state['user_id'], chroma_client, top_k=5)

    flat_chunks = [c['text'] for entry in raw_search_chunks for c in entry['chunks']]
    search_list = [r for r in (list(wiki_results) + list(tavily_results)) if r]

    return {'chunks': flat_chunks, 'search': search_list}
async def Hallucination_Check(state:RAGState):

    messages = [
    SystemMessage(content=hallucination_prompt),
    HumanMessage(
    content= f"""
    User Query:
    {state["query"]}

    chunks:
    {state["chunks"]}

    Search Results:
    {state["search"]}
    
    Previous Answer:
    {state["answer"]}

    Hallucination Reasons:
    {state["hallucination_reason"]}
    """
    )
    ]
    response = await structured_repair_llm.ainvoke(messages)
    return {
        'need_repair':response.need_repair,
        'hallucination_reason':response.hallucination_reason,
        'max_tries_hallucinator': state['max_tries_hallucinator']+1
    }
def Hallucination_Check_router(state:RAGState):
    if state['need_repair']=='factual' or state['max_tries_hallucinator']>5:
        return 'factual'
    elif state['need_repair']=='repair':
        return 'repair'

async def generate_answer(state: RAGState) -> dict:

    # 1. Max retries reached -> graceful failure
    if state["max_tries_planner"] >5 or state['max_tries_hallucinator']>5:
        system_prompt = failure_prompt

        human_prompt = f"""
        User Query:
        {state["query"]}

        Available Evidence:
        {state["chunks"]}

        Search Results:
        {state["search"]}

        Why previous attempts failed:
        {state["hallucination_reason"]}
        """
    else:
        system_prompt = success_prompt

        human_prompt = f"""
        User Query:
        {state["query"]}

        Evidence:
        {state["chunks"]}

        Search Results:
        {state["search"]}
        """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    response = await chat_llm.ainvoke(messages)

    return {
        "answer": response.content
    }