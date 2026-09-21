from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings


urls = [
    "https://www.redhat.com/en/red-hat-eu-cyber-resilience-act",
    "https://www.redhat.com/en/solutions/secure-development-lifecycle",
    "https://www.redhat.com/en/solutions/security-and-compliance-approach",
]

docs = [
    UnstructuredLoader(
        web_url=url, chunking_strategy="basic", max_characters=1000000
    ).load()
    for url in urls
]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings(
    model="bge-large-en-v1.5-f32.gguf",
    base_url="http://192.168.1.190:8082/v1",
    api_key="not-needed",
    # By default, OpenAIEmbeddings tries to use tiktoken to count/truncate
    # tokens the way OpenAI's real embedding models expect — but llama-server's
    # tokenizer for bge-large is different, and this check can raise errors or
    # silently mis-truncate. Setting it to False sends the raw text directly to
    # the server's /v1/embeddings endpoint instead.
    check_embedding_ctx_length=False,
)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag_chroma",
#     embedding=embeddings,
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag_chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()
