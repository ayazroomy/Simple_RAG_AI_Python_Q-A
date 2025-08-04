### Setting Up the Libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel


def initialize_rag_system():
    # Step 1: Set your OpenAI API Key
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')

    
    # Step 2: Load PDF
    pdf_path = "./python_source.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 3: Split into chunks (with overlap for better context)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Step 4: Create Embeddings
    embedding = OpenAIEmbeddings()

    # Step 5: Store in ChromaDB
    persist_directory = "chroma_db_rag"

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )

    vectordb.persist()
    print("✅ Vector store created and persisted.")
    return vectordb, embedding, persist_directory


def get_rag_chain():
    vectordb, embedding, persist_directory = initialize_rag_system()  # Ensure the vector store is initialized
    # Load the vector store
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # Set up retriever + LLM
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(temperature=0)

    # Define your custom prompt template
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are an AI assistant helping users provide answers based on the Context only.
    ONLY use the context provided below. DO NOT use any external knowledge from outside world.

    Focus specifically on giving 'Answer' based on the context if available.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # or "map_reduce" for large docs
            retriever=retriever,  # Your ChromaDB retriever
            chain_type_kwargs={"prompt": custom_prompt}
    )

    return qa_chain



# 🎯 FastAPI setup
app = FastAPI()

qa_chain = get_rag_chain()  # Initialize the RAG chain

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        response = qa_chain.run(request.query)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}