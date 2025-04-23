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

##Using Langchain
# llm = ChatOpenAI(temperature=0.5)

# retriever_from_llm = MultiQueryRetriever.from_llm(
#     llm=llm,
#     retriever=vector_store.as_retriever()
# )


# #Logging
# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# unique_docs = retriever_from_llm.invoke(question)
# print(len(unique_docs))

# SYSTEM_PROMPT = f"""
# You are an helpful AI assistant who helps to resolve user's query.
# You answer the queries based on the context you have.

# context = {unique_docs}

# """

context_for_queries = found_docs = vector_store.similarity_search(question)

GENERATE_QUERIES_SYSTEM_PROMT = f"""
You are an helpful AI assistant who takes a user query and generates 5 similar queries around it.
These queries are understand the what all things user might want to know.
Use context_for_queries to generate alternative questions around that specific topic.

context_for_queries = {context_for_queries}

Rules - 
    1. You will generate 5 queries similar to user query.
    2. You should return these 5 queries in JSON format only

Example - 
User - Who was Mahatma Gandhi? 
Your Output - {{"queries": [
                    1. Who was Mahatma Gandhi and what contributions he gave?
                    2. What was the fullname of Mahatma Gandhi and where was is born?
                    3. What was the occupation of Mahatma Gandhi?
                    4. What are the things which Mahatma Gandhi liked and believed in?
                    5. Who all were there in family of Mahatma Gandhi?
                ]}}

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMT},
        {"role": "user", "content": question}
    ],
    max_tokens=1000
)

formatted_queries = json.loads(response.choices[0].message.content)

print(formatted_queries)

prompt = ""

for query in formatted_queries["queries"]:
    prompt += query + " " 

print("input prompt -> ",prompt)

found_docs = vector_store.similarity_search(prompt)

print(len(found_docs))

SYSTEM_PROMPT = f"""
You are an helpful AI assistant who helps to resolve user's query.
You answer the queries based on the context you have.

context = {found_docs}

"""


answer = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ],
    max_tokens=1000
)

print("OUTPUT -> ",answer.choices[0].message.content)