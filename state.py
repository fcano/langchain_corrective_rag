from typing import TypedDict, List


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Atttributes:
        question: question
        answer: LLM response
        web_search_ whether to add search
        documents: list of documents
    """

    question: str
    answer: str
    web_search: bool
    documents: List[str]
