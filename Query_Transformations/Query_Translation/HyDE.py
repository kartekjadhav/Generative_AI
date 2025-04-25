from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
client = OpenAI()

web_url = "https://www.healthline.com/nutrition/an-apple-a-day-keeps-the-doctor-away"

# Loader
loader = WebBaseLoader(web_url)
docs = loader.load()
print("DOC LOADING DONE")

#Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = splitter.split_documents(docs)

print("SPLITTING DOCS DONE")

#Vectore Store
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="Apple_Blog",
    embedding=embedding,
    timeout=20.0
)

print("INGESTION DONE!")

#  QUERING

def queryChain(question):
    #Here we generate a response from LLM for the question and do similarity search on that answer.
    #The retrieved documents are then returned.
    hypothetical_document = llm_chain.invoke({"question": question})
    print(len(hypothetical_document))
    print("hypothetical_document -> ",hypothetical_document)

    relevant_chunks = vector_store.similarity_search(hypothetical_document, k=3)
    return relevant_chunks

SYSTEM_PROMPT_1 = PromptTemplate(
    input_variables=["question"],
    template="""
    Generate a concise answer for below question.
    
    Question - {question}
    """
)

llm = ChatOpenAI(temperature=0.5)

llm_chain = SYSTEM_PROMPT_1 | llm | StrOutputParser()

question = "What are benefits of eating apples?"

relevant_chunks = queryChain(question)



SYSTEM_PROMT_2 = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Generate a concise answer for the given question. 
    Utilise the given context to answer the question
    
    Context - {context}
    Question - {question}
    """
)

query_chain = SYSTEM_PROMT_2 | llm | StrOutputParser()

context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

answer = query_chain.invoke({"context": context, "question": question})

print("ANSWER -> ",answer)