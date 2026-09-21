from state import GraphState
from ingestion import retriever
from tools import grader_chain
from typing import Dict, Any


def retriever_node(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents}


def grader_node(state: GraphState) -> Dict[str, Any]:
    print("---GRADE---")

    question = state["question"]
    documents = state["documents"]

    relevant_docs = []
    web_search = False

    for doc in documents:
        response = grader_chain.invoke(
            {'question': question, 'document': doc.page_content}
        )

        is_relevant = response.binary_score
        if is_relevant.lower() == "yes":
            relevant_docs.append(doc)
        else:
            web_search = True
            continue

    return {"documents": relevant_docs, "web_search": web_search}