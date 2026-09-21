from ingestion import retriever
from tools import grader_chain, question_answer_chain
from pprint import pprint


def test_grader_chain_answer_yes() -> None:
    question = "Does Red Hat secure development lifecycle aligns with NIST?"

    docs = retriever.invoke(question)
    # docs[0] is the higher ranked doc
    doc_txt = docs[0].page_content

    doc_is_related = grader_chain.invoke(
        {"question": question, "document": doc_txt}
    )

    #print(doc_txt)
    assert doc_is_related.binary_score == "yes"


def test_grader_chain_answer_no() -> None:
    question = "What does the concept DRY mean in programming?"

    docs = retriever.invoke(question)
    # docs[0] is the higher ranked doc
    doc_txt = docs[0].page_content

    doc_is_related = grader_chain.invoke(
        {"question": question, "document": doc_txt}
    )

    #print(doc_txt)
    assert doc_is_related.binary_score == "no"


def test_question_answer_chain() -> None:
    question = "Does Red Hat secure development lifecycle aligns with NIST?"
    docs = retriever.invoke(question)
    answer = question_answer_chain.invoke({"context": docs, "question": question})
    pprint(answer)
