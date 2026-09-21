import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch


load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("MODEL"),
    base_url=os.environ.get("INFERENCE_SERVER_URL"),
    api_key=os.environ.get("MODEL_PROVIDER_API_KEY"),
    temperature=0,
)


class BinaryScore(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question 'yes' or 'no'"
    )


llm_with_structured_output = llm.with_structured_output(BinaryScore)

system_prompt = """
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

grader_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: \n\n {question}"),
    ]
)

grader_chain = grader_prompt_template | llm_with_structured_output

tavily_search_tool = TavilySearch(max_results=3)

question_answer_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)

question_answer_chain = question_answer_prompt_template | llm | StrOutputParser()
