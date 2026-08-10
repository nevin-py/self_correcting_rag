from typing import TypedDict,Optional
from langgraph.graph import StateGraph,START,END
class operator(TypedDict):
    num1:int
    num2:int
    num3:int
    num4:int
    final:int
    res1:int
    res2:int
def node_one(state:operator)->operator:
    state['res1']=state['num1']+state['num2']
    return state
def node_two(state:operator)->operator:
    state['res2']=state['num3']+state['num4']
    return state
def decision_node(state:operator):
    if state['res1']>state['res2']:
        return 'subtract'
    else:
        return 'add'
def node_three(state:operator)->operator:
    print('we choose add')
    state['final']=state['res1']+state['res2']
    return state
def node_four(state:operator)->operator:
    print('we choose sub')
    state['final']=state['res1']-state['res2']
    return state
graph=StateGraph(operator)
graph.add_node('set1',node_one)
graph.add_node('set2',node_two)
graph.add_node('set_empty',lambda state:state)
graph.add_node('set3',node_three)
graph.add_node('set4',node_four)
graph.add_conditional_edges('set_empty',decision_node,{
    'add':'set3',
    'subtract':'set4'
})
graph.add_edge(START,'set1')
graph.add_edge('set1','set2')
graph.add_edge('set2','set_empty')
graph.add_edge('set3',END)
graph.add_edge('set4',END)
app=graph.compile()
png = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png)
result=app.invoke({'num1':15,'num2':5,'num3':8,'num4':7})
print(result)