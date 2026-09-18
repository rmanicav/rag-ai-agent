
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from agent import structured_decision, execute_tool_call
from guardrails import guardrail


app = FastAPI(
    title="RAG AI Agent",
    description="AI agent with document retrieval and calculator tools",
    version="1.0.0"
)


class AgentRequest(BaseModel):
    question: str


# Load document
pdf_path = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Create chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_paper"
)


@app.get("/")
def home():
    return {
        "message": "RAG AI Agent is running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "chunks": len(chunks)
    }


@app.post("/agent")
def agent_endpoint(request: AgentRequest):

    allowed, message = guardrail(request.question)

    if not allowed:
        return {
            "status": "blocked",
            "answer": message
        }

    tool_call = structured_decision(request.question)

    result = execute_tool_call(
        tool_call,
        vectorstore
    )

    return {
        "status": "success",
        "tool": tool_call.tool,
        "result": result
    }
