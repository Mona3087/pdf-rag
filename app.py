import re
import chainlit as cl  # importing chainlit for our app 
from dotenv import load_dotenv
import tiktoken


from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

load_dotenv()
# Define a template for generating chat prompts using a given context and query.
RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}
Answer the question based on {context}, if you can't figure just say "I don't know"
"""

# Initialize the OpenAI chat model specifically for version gpt-3.5-turbo.
openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")
# Obtain the encoding function for the specified model to handle text inputs.
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
# Initialize a model for embedding text, using a smaller, faster version.
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# Load documents from a PDF using the PyMuPDF library, specifically a file named 'facebook_10k.pdf'.
docs = PyMuPDFLoader('facebook_10k.pdf').load()

# Function to compute the number of tokens in a given text using the tiktoken library.
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)
    return len(tokens)


# Define a text splitter that handles character-based splitting with specific overlap and size settings.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 10,
    length_function = tiktoken_len,
)
# Split the loaded documents into chunks using the defined text splitter.
split_chunks = text_splitter.split_documents(docs)

# Create a vector store in memory using the Qdrant library, which stores document embeddings.
qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="Facebook 10k",
)
# Convert the vector store into a retriever that can fetch relevant document chunks.
qdrant_retriever = qdrant_vectorstore.as_retriever()

# Template for generating retrieval-augmented prompts.
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# Chain sequence that handles context retrieval, question posing, and response generation.
retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
)


# Decorator to mark function execution at the start of a user chat session, initializing the chatbot.
@cl.on_chat_start  
async def start_chat():
    msg=cl.Message(content="Firing up the 10k fillings info bot...")
    await msg.send()
    runnable=retrieval_augmented_qa_chain
    msg.content= "Hello, welcome to 10k fillings info bot. What is your query?"
    await msg.update()
    cl.user_session.set("runnable",runnable)

# Decorator to handle incoming messages from users, executing the question-answering sequence.
@cl.on_message  
async def main(message: cl.Message):
    runnable=cl.user_session.get("runnable")
    inputs = {"question": message.content}
    cb = cl.AsyncLangchainCallbackHandler()
    result = await runnable.ainvoke(inputs, callbacks=[cb])
    response = result.get("response")
    pattern = r"content='(.*?)'"
    pattern_new = r'content="(.*?)"'

    # Using re.search to find the first match
    match = re.search(pattern, str(response))
    match_new = re.search(pattern_new, str(response))
    extracted_content = ""
    # Extracting and printing the matched content
    if match:
        extracted_content = match.group(1)
    elif match_new:
        extracted_content = match_new.group(1)
    else:
        print("Error")
    
    await cl.Message(content=extracted_content).send()