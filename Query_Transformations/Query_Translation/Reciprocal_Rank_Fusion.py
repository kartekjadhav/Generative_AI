from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

client = OpenAI()

def reciprocal_rank_fusion(relevant_chunks):
    #Takes List of Lists of documents and returns list of docs based upon ranking.

    doc_ranking = {}
    doc_map = {}
    for docs in relevant_chunks:
        for rank, doc in enumerate(docs, start=1):
            reciprocal = 1 / (rank + 60)
            id = doc.metadata.get("_id", hash(doc.page_content))
            if id not in doc_ranking:
                doc_ranking[id] = 0
                doc_map[id] = doc
            doc_ranking[id] += reciprocal   

    sorted_ids_by_rank = sorted(doc_ranking, key=lambda x: doc_ranking[x], reverse=True)

    ranked_docs = []

    for id in sorted_ids_by_rank:
        ranked_docs.append(doc_map[id])
    
    return ranked_docs

#Loader
web_url = "https://netflixtechblog.com/introducing-impressions-at-netflix-e2b67c88c9fb"
loader = WebBaseLoader(web_url)
docs = loader.load()
print("Doc loading is completed!")

#Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = splitter.split_documents(docs)

print("Doc splitting is completed!")

#Embedding
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    embedding=embedding,
    url="http://localhost:6333",
    collection_name="reciprocal_retrieval",
    timeout=20.0
)

print("Ingestion is completed!")


# RETRIEVAL

question = "what is this blog all about?"


SYSTEM_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
        You are an AI assistant . Your task is to generate five different question from the question given by user to retrieve relevant documents from the vector database.
         By generating different queries you goal is to overcome the distance based similarity search.
         Give out output question seperated by newlines.
        Original Question = {question} 
    """
)

llm = ChatOpenAI(temperature=0.5)

llm_chain = SYSTEM_PROMPT | llm | StrOutputParser() | (lambda x: x.split("\n"))

queries = llm_chain.invoke({"question": question})

print(queries)


relevant_chunks = []

for query in queries:
    chunk = vector_store.similarity_search(query)
    relevant_chunks.append(chunk)

ranked_relevant_chunks = reciprocal_rank_fusion(relevant_chunks)

if len(relevant_chunks) > 0:
    print("RETRIEVED RANKED CHUNKS")

SYSTEM_PROMPT_2 = f"""
You are an AI assistant who resolves user query based on given context.
Context is arranged rank wise, that is the first doc in context list has highest similarity, second little less and so on.
Based on ranking give preference to the docs in context list.

context = {ranked_relevant_chunks}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content":SYSTEM_PROMPT_2},
        {"role": "user", "content": question}
    ]
)

print(response.choices[0].message.content)