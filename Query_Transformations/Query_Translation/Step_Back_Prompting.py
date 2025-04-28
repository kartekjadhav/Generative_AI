from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


embedding = OpenAIEmbeddings()

vector_store = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="Apple_Blog",
    embedding=embedding
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

examples = [
    {
        "input": "When is John Cena's birthday?",
        "output": "What is John Cena's personal history?"
    },
    {
        "input": "Who played the lead role in movie Kill Bill?",
        "output": "What is the cast of movie Kill Bill?"
    },
    {
        "input": "What is 9pm EST to IST?",
        "output": "What is the timing difference between EST and IST?"
    }
]


#Example messages
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
]) 

#Creating FewShotPrompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in world's knowledge. You take user's question and paraphrase into a more abstract step-back question, which is easier to answer. Here are some examples."),
    few_shot_prompt,
    ("human", "{question}")
])

generate_step_back_queries = prompt | ChatOpenAI(temperature=0.5) | StrOutputParser()


#response prompt
template = """
    You are an helpful AI assistant who answers user's question in a comprehensive way. You utilise the give normal_context, step_back_context to generate the answer.
    # {normal_context}
    # {step_back_context}
    # Question {question}
"""

response_prompt = ChatPromptTemplate.from_template(template)

question = input("Enter your question > ").strip()

chain = (
    {
        #Retrieve from user's question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        #Retrieve from step back question
        "step_back_context": generate_step_back_queries | retriever,
        "question": lambda x : x["question"]
    } | response_prompt | ChatOpenAI(temperature=0.5) | StrOutputParser())

answer = chain.invoke({"question": question})

print(answer)
