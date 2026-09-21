from state import GraphState
from nodes import retriever_node, grader_node, web_search_node, question_answer_node
from tools import hallucination_grader_chain, answersthequestion_grader_chain
from langgraph.graph import StateGraph, END

def decide_if_websearch(state: GraphState) -> str:
    if state["web_search"]:
        return "web_search_node"
    else:
        return "question_answer_node"


def is_answer_useful(state: GraphState) -> str:
    question = state["question"]
    documents = state["documents"]
    answer = state["answer"]

    is_grounded = hallucination_grader_chain.invoke(
        {"documents": documents, "answer": answer}
    )

    if is_grounded:
        answers_the_question = answersthequestion_grader_chain.invoke(
            {"question": question, "answer": answer}
        )

        if answers_the_question:
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"

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

graph.add_conditional_edges("question_answer_node", is_answer_useful, {
    "useful":END,
    "not useful": "web_search_node",
    "not supported": "question_answer_node"
})

app = graph.compile()

with open("graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())
