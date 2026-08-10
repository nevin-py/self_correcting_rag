
from langgraph.graph import StateGraph, START, END
from app.agent.nodes import *

graph=StateGraph(RAGState)
graph.add_node('initial_retrieval',Initial_Chunks)

graph.add_node('Planner',Planner)
graph.add_node('search',search_tool)
graph.add_node('Hallucination checker',Hallucination_Check)
graph.add_node('Answer generation',generate_answer)
graph.add_edge(START,'initial_retrieval')
graph.add_edge('initial_retrieval','Planner')
graph.add_conditional_edges('Planner',planner_router,{
    'not enough':'search',
    'evident':'Answer generation'
})
graph.add_edge('Answer generation','Hallucination checker')
graph.add_edge('search','Planner')
graph.add_conditional_edges('Hallucination checker',Hallucination_Check_router,{
    'factual':END,
    'repair':'Planner'
})
def return_app():
    app=graph.compile()
    return app
# png = app.get_graph().draw_mermaid_png()

# with open("graph.png", "wb") as f:
#     f.write(png)