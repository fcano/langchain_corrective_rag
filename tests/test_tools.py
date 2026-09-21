from ingestion import retriever
from tools import grader_chain, question_answer_chain, hallucination_grader_chain
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


def test_hallucination_grader_chain_answer_yes() -> None:
    question = "Does Red Hat secure development lifecycle aligns with NIST?"
    docs = retriever.invoke(question)
    answer = question_answer_chain.invoke({"context": docs, "question": question})
    result = hallucination_grader_chain.invoke(
        {"documents": docs, "answer": answer}
    )

    assert result.binary_score


def test_hallucination_grader_chain_answer_no() -> None:
    question = "Does Red Hat secure development lifecycle aligns with NIST?"
    docs = retriever.invoke(question)
    answer = question_answer_chain.invoke({"context": docs, "question": question})
    result = hallucination_grader_chain.invoke(
        {"documents": docs, "answer": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec blandit risus et arcu semper porttitor id et ligula. Aenean tincidunt nunc ipsum, non faucibus elit porttitor maximus."}
    )

    assert not result.binary_score