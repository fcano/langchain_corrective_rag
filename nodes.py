from state import GraphState
from ingestion import retriever
from typing import Dict, Any


def retriever_node(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents}
