from typing import Any, Dict

from langchain_core.documents import Document

from ingestion import retriever
from state import GraphState
from tools import grader_chain, tavily_search_tool, question_answer_chain


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
            {"question": question, "document": doc.page_content}
        )

        is_relevant = response.binary_score
        if is_relevant.lower() == "yes":
            relevant_docs.append(doc)
        else:
            web_search = True
            continue

    return {"documents": relevant_docs, "web_search": web_search}


def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = tavily_search_tool({"query": question})

    joined_tavilt_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents}


def question_answer_node(state: GraphState) -> Dict[str, Any]:
    print("---REPLY TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    answer = question_answer_chain.invoke({"context": documents, "question": question})

    return {"answer": answer}
