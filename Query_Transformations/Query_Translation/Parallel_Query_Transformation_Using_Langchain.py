from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
import logging
import json

load_dotenv()
client = OpenAI()

# #Load Blog Post
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# docs = loader.load()
# print("DOC LOADED")

# #Split into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
# split_docs = text_splitter.split_documents(docs)
# print("SPLITTED DOCS")

#Store in Vector DB
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    embedding=embedder,
    url="http://localhost:6333/",
    collection_name="blog_collection",
    timeout=20.0
)

# vector_store.add_documents(documents=split_docs)

# print("INGESTION DONE")



#RETRIVAL


question = "What are the approaches to Task Decomposition?"

#Using Langchain
llm = ChatOpenAI(temperature=0.5)

retriever_from_llm = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever()
)

#Logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriever_from_llm.invoke(question)
print(len(unique_docs))



context = "\n\n".join(doc.page_content for doc in unique_docs)
context = context[:8000]

SYSTEM_PROMPT = f"""
You are an helpful AI assistant who helps to resolve user's query.
You answer the queries based on the context you have.

context = {context}

"""


answer = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
)

print("OUTPUT -> ",answer.choices[0].message.content)