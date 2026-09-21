from state import GraphState
from nodes import retriever_node, grader_node, web_search_node, question_answer_node
from langgraph.graph import StateGraph, END

def decide_if_websearch(state: GraphState) -> str:
    if state["web_search"]:
        return "web_search_node"
    else:
        return "question_answer_node"


graph = StateGraph(GraphState)

graph.add_node("retriever_node", retriever_node)
graph.add_node("grader_node", grader_node)
graph.add_node("web_search_node", web_search_node)
graph.add_node("question_answer_node", question_answer_node)

graph.set_entry_point("retriever_node")

graph.add_edge("retriever_node", "grader_node")

graph.add_conditional_edges("grader_node", decide_if_websearch, {
    "web_search_node":"web_search_node",
    "question_answer_node":"question_answer_node"
})

graph.add_edge("web_search_node", "question_answer_node")
graph.add_edge("question_answer_node", END)

app = graph.compile()

with open("graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())
