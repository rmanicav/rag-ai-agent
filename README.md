# RAG AI Agent

An end-to-end **Retrieval-Augmented Generation (RAG) and AI Agent** project built in Python.

The project starts with a document-grounded RAG pipeline and extends it into an agent that can select and execute tools based on the user's request. It demonstrates practical patterns for document ingestion, semantic retrieval, reranking, retrieval evaluation, tool use, planning, memory, guardrails, reliability, observability, API development, testing, and containerization.

> **Scope:** This repository is specifically for the **RAG + AI Agent project**. It does not contain general machine-learning study material, fraud-detection MLOps material, or interview-preparation content.

---

## Overview

The system uses the research paper:

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**

as its initial knowledge source.

The application can:

- Load and process a PDF
- Split the document into overlapping chunks
- Generate text embeddings
- Store embeddings in ChromaDB
- Perform semantic similarity search
- Apply retrieval-quality checks
- Rerank retrieved documents
- Preserve source/page metadata
- Generate answers using retrieved context
- Evaluate retrieval quality
- Test hallucination/grounding behavior
- Filter retrieval using metadata
- Combine keyword and vector retrieval
- Route requests to different tools
- Search the knowledge base through an agent
- Perform calculations through a tool
- Execute multiple planned actions
- Maintain conversation state
- Apply basic input guardrails
- Detect simple prompt-injection attempts
- Handle tool failures
- Retry transient failures
- Log agent execution
- Expose the system through FastAPI
- Test the API programmatically
- Provide Docker configuration

---

## Architecture

```text
                           User
                            |
                            v
                       FastAPI API
                            |
                            v
                       Guardrails
                            |
                            v
                    Agent / Planner
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
        RAG Search      Calculator      Direct LLM
             |
             v
        Vector Search
             |
             v
          ChromaDB
             |
             v
          Reranker
             |
             v
       Retrieved Context
             |
             v
             LLM
             |
             v
       Final Response
             |
             v
       Logging / Metrics
```

The RAG retriever is exposed as one of the agent's tools.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| RAG framework | LangChain |
| PDF processing | PyPDF |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` |
| LLM runtime | Hugging Face Transformers |
| API | FastAPI |
| Validation | Pydantic |
| Server | Uvicorn |
| Testing | FastAPI TestClient |
| Containerization | Docker |
| Version control | Git / GitHub |

---

# RAG PIPELINE

## Document Ingestion

The initial knowledge base is created from a PDF research paper.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(pdf_path)
documents = loader.load()
```

`PyPDFLoader` extracts the document content while preserving useful metadata such as page information.

The page metadata is later used for source attribution and retrieval evaluation.

---

## Chunking

Large documents are split into smaller overlapping pieces before embedding.

The main configuration used by the project is:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)
```

The main configuration produced approximately **111 chunks** for the selected paper.

Chunking creates a trade-off:

```text
Smaller chunks
    -> potentially more precise retrieval
    -> more chunks
    -> less surrounding context

Larger chunks
    -> more surrounding context
    -> fewer chunks
    -> potentially more irrelevant content
```

The project experimented with:

```text
400 / 80    -> 205 chunks
800 / 150   -> 111 chunks
1200 / 200  -> 73 chunks
```

The `800 / 150` configuration was used for the main prototype.

---

## Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The model produces **384-dimensional embeddings**.

Conceptually:

```text
Text
  |
  v
Embedding Model
  |
  v
Vector Representation
```

The vectors allow the application to compare semantic similarity between questions and document chunks.

---

## Vector Database

ChromaDB is used as the vector store.

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_paper"
)
```

The vector store provides the retrieval layer used by the RAG pipeline and the agent's document-search tool.

---

## Semantic Retrieval

A basic retrieval operation is:

```python
results = vectorstore.similarity_search(
    "What is Retrieval-Augmented Generation?",
    k=3
)
```

The retrieval flow is:

```text
Question
   |
   v
Question Embedding
   |
   v
Vector Similarity Search
   |
   v
Relevant Chunks
```

Semantic retrieval allows the system to find conceptually related content even when the exact wording differs.

---

## Retrieval Distance

The project inspected Chroma similarity distances to understand retrieval quality.

For the configuration used:

```text
Lower distance = more similar
Higher distance = less similar
```

During experimentation, a relevant query produced distances around:

```text
0.85
0.89
0.91
```

An unrelated query such as:

```text
What is the capital of Australia?
```

produced a much weaker result, around:

```text
1.78
```

These observations were used to experiment with a retrieval-quality gate.

---

## Retrieval-Quality Gate

A simple demonstration threshold of:

```text
maximum distance = 1.2
```

was tested.

Conceptually:

```text
Question
   |
   v
Retrieve Documents
   |
   v
Sufficiently Relevant?
     / \
   Yes  No
    |    |
    v    v
   LLM  Explain that
        the knowledge
        base is insufficient
```

The threshold is **not a universal value**. It depends on the embedding model, distance metric, corpus, and evaluation data.

Production thresholds should be calibrated against a representative evaluation set.

---

## RAG Generation

The RAG process combines the user question with retrieved context:

```text
Question
    +
Retrieved Context
    |
    v
Prompt
    |
    v
LLM
    |
    v
Answer
```

The project uses:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

through Hugging Face Transformers.

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=300,
    temperature=0.1
)
```

The model is intentionally small and open source so the project can run without requiring a paid model API.

---

## Source Attribution

Retrieved chunks retain their page metadata.

Example:

```python
{
    "content": doc.page_content,
    "page": doc.metadata.get("page")
}
```

This makes it possible to associate retrieved information with its source page.

Source attribution is useful for:

- Verification
- Debugging
- Trust
- Research
- Auditing

---

# RETRIEVAL IMPROVEMENT

## Cross-Encoder Reranking

The project adds a second-stage reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The retrieval pipeline becomes:

```text
Question
   |
   v
Vector Search
   |
   v
Top 10 Candidates
   |
   v
Cross-Encoder
   |
   v
Reranked Results
   |
   v
Top 3
```

Vector search provides fast candidate generation, while the cross-encoder provides a more detailed relevance assessment for the candidates.

---

## Metadata Filtering

Document metadata can be used to constrain retrieval.

The project experimented with metadata such as:

```text
document_type = research_paper
topic = RAG
language = English
```

The conceptual flow is:

```text
Question
   |
   v
Metadata Filter
   |
   v
Vector Search
   |
   v
Reranking
   |
   v
Context
```

This becomes especially useful when the knowledge base contains many different documents.

---

## Hybrid Search

The project also explored combining lexical and semantic retrieval.

```text
Keyword Search
      +
Vector Search
      |
      v
Combined Ranking
```

A simple experiment combined TF-IDF retrieval with Chroma similarity.

For a production implementation, a common architecture is:

```text
BM25
  +
Vector Search
  +
Reciprocal Rank Fusion (RRF)
```

Keyword retrieval is useful for exact terms, names, identifiers, and phrases, while vector retrieval is useful for semantic relationships.

---

# RAG EVALUATION

## Retrieval Evaluation

Retrieval should be evaluated independently from LLM generation.

A small manually labeled evaluation set was created:

```python
evaluation_questions = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "relevant_pages": [0, 1]
    },
    {
        "question": "What dataset was used to evaluate the RAG model?",
        "relevant_pages": [3, 5, 6]
    },
    {
        "question": "What are the main components of a RAG model?",
        "relevant_pages": [1, 2]
    }
]
```

This is a small learning/portfolio evaluation set, not a production benchmark.

---

## Precision@K

Precision@K measures the proportion of retrieved results that are relevant.

```text
Precision@K =
Relevant Retrieved Documents / K
```

---

## Recall@K

Recall@K measures how much of the relevant information was retrieved.

```text
Recall@K =
Relevant Retrieved Documents /
Total Relevant Documents
```

---

## Mean Reciprocal Rank

MRR measures how highly the first relevant result appears.

```text
MRR = 1 / Rank of First Relevant Result
```

A relevant document at rank 1 gives:

```text
MRR = 1
```

A relevant document at rank 3 gives:

```text
MRR = 1/3
```

---

## Answer Evaluation

RAG evaluation also needs to consider generated answers.

Important dimensions include:

### Faithfulness / Groundedness

Is the answer supported by the retrieved context?

### Answer Relevance

Does the answer actually answer the question?

### Context Relevance

Is the retrieved context relevant to the question?

Conceptually:

```text
Question
   |
   v
Retrieval Quality
   |
   v
Context Quality
   |
   v
Answer Relevance
   |
   v
Faithfulness
```

A simple word-overlap faithfulness check was explored as a learning exercise. It is not considered sufficient for production evaluation.

More robust production approaches can include:

- LLM-based evaluation
- Natural Language Inference
- Citation verification
- Human evaluation
- Dedicated RAG evaluation frameworks

---

## Hallucination Testing

The project deliberately tested an unrelated question:

```text
What is the capital of Australia?
```

This demonstrated that an LLM can attempt to answer even when the retrieved context does not contain supporting evidence.

The retrieval-quality gate therefore provides an important control:

```text
Strong supporting context
        |
        v
      Answer

Insufficient supporting context
        |
        v
Do not claim the knowledge base supports it
```

RAG reduces hallucination risk but does not guarantee hallucination-free responses.

---

# AI AGENT

## Extending RAG into an Agent

The project extends the RAG pipeline into an agentic workflow.

Traditional RAG:

```text
Question
   |
   v
Retrieve
   |
   v
Generate
```

Agent:

```text
Question
   |
   v
Decide
   |
   v
Choose Tool
   |
   v
Execute Tool
   |
   v
Observe Result
   |
   v
Continue or Answer
```

The RAG retriever becomes one of the agent's tools.

---

## Agent Tools

The prototype provides three logical paths:

```text
search_documents
calculate
direct_llm
```

### `search_documents`

Searches the ChromaDB knowledge base.

### `calculate`

Performs arithmetic.

### `direct_llm`

Handles requests that do not require the document-search or calculator tool.

---

## Document Search Tool

```python
def search_documents(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)

    return [
        {
            "content": doc.page_content,
            "page": doc.metadata.get("page")
        }
        for doc in results
    ]
```

The tool provides a clean interface between the agent and the retrieval layer.

---

## Calculator Tool

The prototype calculator sanitizes basic arithmetic expressions:

```python
import re

def calculate(expression):
    try:
        expression = re.sub(
            r"[^0-9+\-*/().% ]",
            "",
            expression
        )

        return eval(
            expression,
            {"__builtins__": {}},
            {}
        )

    except Exception:
        return "Unable to calculate expression."
```

This is a prototype implementation for learning purposes.

A production application should use a dedicated safe expression parser rather than `eval`.

---

## Tool Selection

The current prototype uses deterministic routing logic to select:

```text
SEARCH
CALCULATE
DIRECT
```

Examples:

```text
"What is Retrieval-Augmented Generation?"
        |
        v
     SEARCH
```

```text
"What is 25 * 40?"
        |
        v
    CALCULATE
```

```text
"What is the capital of Germany?"
        |
        v
     DIRECT
```

This makes the first version predictable and testable.

A future version can replace deterministic routing with native LLM function/tool calling.

---

## Structured Tool Calls

Tool calls are represented using a structured object:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ToolCall:
    tool: str
    argument: Optional[str] = None
```

Example:

```python
ToolCall(
    tool="search_documents",
    argument="What is Retrieval-Augmented Generation?"
)
```

Calculator:

```python
ToolCall(
    tool="calculate",
    argument="25 * 40"
)
```

This separates the decision about **what to call** from the implementation of **how to execute it**.

---

## Tool Execution

The execution flow is:

```text
Agent Decision
      |
      v
Structured ToolCall
      |
      v
Tool Executor
      |
      v
Selected Tool
      |
      v
Tool Result
```

This makes the architecture easier to extend with additional tools.

---

## Planning and Multi-Step Execution

The prototype can identify multiple actions for a combined request.

Example:

```text
What is Retrieval-Augmented Generation,
and what is 25 * 40?
```

Possible plan:

```text
1. SEARCH
2. CALCULATE
```

Execution:

```text
                   User Question
                         |
                         v
                      Planner
                     /       \
                    v         v
                 Search   Calculator
                    |         |
                    +----+----+
                         |
                         v
                      Results
                         |
                         v
                    Final Answer
```

A production agent can extend this with explicit state, step limits, tool permissions, and termination conditions.

---

## Agent Loop

The general agent loop is:

```text
User
 |
 v
Guardrail
 |
 v
Agent
 |
 v
Decision
 |
 v
Tool
 |
 v
Observation
 |
 v
Agent
 |
 +---- More work? ---- YES ----+
 |                             |
 NO                            |
 |                             |
 v                             |
Final Answer <-----------------+
```

This is the foundation of an agentic workflow.

---

# MEMORY, SECURITY, AND RELIABILITY

## Conversation State

Basic conversation state was explored using:

```python
conversation_history = []
```

Messages can be stored as:

```python
conversation_history.append({
    "role": "user",
    "content": question
})
```

and:

```python
conversation_history.append({
    "role": "assistant",
    "content": answer
})
```

This provides basic short-term conversation state.

For production, persistent conversation state could be stored in systems such as Redis or PostgreSQL.

---

## Guardrails

The project includes basic input guardrails.

The current checks include:

- Empty input
- Maximum input length
- Simple prompt-injection patterns

Examples include:

```text
ignore previous instructions
reveal system prompt
show your system prompt
```

The flow is:

```text
User Input
    |
    v
Guardrail
    |
    +---- Block ----> Safe Response
    |
    v
  Agent
```

These are basic protections and should not be treated as a complete production security solution.

---

## Prompt Injection

Prompt injection is particularly important in RAG and agent applications because external content may be untrusted.

A core principle is:

```text
Retrieved documents are DATA.
Retrieved documents are NOT instructions.
```

Untrusted content can come from:

- User input
- Documents
- Web content
- Tool output
- External APIs
- Conversation memory

The application should preserve trusted system/application instructions and restrict what tools are permitted to do.

---

## Tool Security

Tools should have explicit permissions.

Production architecture:

```text
Agent
  |
  v
Permission Check
  |
  +---- Allowed ----> Execute Tool
  |
  +---- Denied -----> Reject
```

Higher-risk tools may require:

- Authentication
- Authorization
- Human approval
- Sandboxing
- Rate limiting
- Resource limits
- Audit logging

---

## Error Handling

Tool failures can occur because of:

- Invalid arguments
- Database failures
- Network errors
- Service outages
- Timeouts
- Model errors

The application should handle failures gracefully.

```text
Agent
  |
  v
Tool
  |
  +---- Success ---> Continue
  |
  +---- Failure ---> Error Handler
                         |
                         v
                  Graceful Response
```

---

## Retry Handling

The prototype includes a basic retry mechanism with a maximum of three attempts.

```text
Tool Call
   |
   v
Attempt 1
   |
   +-- Success ---> Continue
   |
   +-- Failure
          |
          v
       Attempt 2
          |
          +-- Failure
                 |
                 v
              Attempt 3
                 |
                 v
          Graceful Failure
```

Retries should be applied selectively.

```text
Transient failure  -> potentially retry
Invalid input      -> do not retry
Permission failure -> do not retry
Permanent failure  -> do not repeatedly retry
```

Production systems should normally use timeouts, exponential backoff, and appropriate error classification.

---

# OBSERVABILITY

## Logging

The agent uses Python logging to make execution visible.

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("agent")
```

Useful events include:

- Agent start
- Tool selection
- Tool execution
- Retry attempts
- Errors
- Completion
- Latency

---

## Latency

RAG latency can be decomposed into:

```text
Retrieval
    +
Reranking
    +
Generation
    =
Total Latency
```

Important metrics include:

- p50
- p95
- p99
- Retrieval latency
- Reranking latency
- LLM generation latency
- Total request latency

Potential optimizations include:

- Smaller retrieval `k`
- Smaller reranking candidate sets
- Caching
- Batching
- GPU inference
- Smaller models
- Streaming

---

## Production Observability

A production implementation could add:

```text
OpenTelemetry
      |
      v
Distributed Tracing
      |
      v
Metrics
      |
      v
Dashboards
      |
      v
Alerts
```

Useful production metrics include:

- Request count
- Error rate
- Tool-call count
- Tool failure rate
- Retrieval latency
- Reranking latency
- LLM latency
- Token usage
- Cost
- Retrieval quality
- Answer quality
- Agent task completion

---

# API

## FastAPI

The agent is exposed through FastAPI.

Endpoints:

```text
GET  /
GET  /health
POST /agent
```

The API uses Pydantic for request validation.

---

## Root Endpoint

```http
GET /
```

Example:

```json
{
  "message": "RAG AI Agent is running",
  "version": "1.0.0"
}
```

---

## Health Endpoint

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "chunks": 111
}
```

---

## Agent Endpoint

```http
POST /agent
```

Request:

```json
{
  "question": "What is Retrieval-Augmented Generation?"
}
```

Processing:

```text
HTTP Request
     |
     v
Validation
     |
     v
Guardrails
     |
     v
Agent
     |
     v
Tool
     |
     v
Result
     |
     v
HTTP Response
```

---

# TESTING

## API Testing

FastAPI's `TestClient` is used for automated API testing.

```python
from fastapi.testclient import TestClient

client = TestClient(app)

response = client.post(
    "/agent",
    json={"question": "What is 25 * 40?"}
)

assert response.status_code == 200
```

Tests should cover:

- Root endpoint
- Health endpoint
- Calculator behavior
- Document-search behavior
- Tool selection
- Invalid input
- Guardrail behavior
- Error handling

---

# DOCKER

## Dockerfile

The repository contains a Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:

```bash
docker build -t rag-ai-agent .
```

Run:

```bash
docker run -p 8000:8000 rag-ai-agent
```

The Dockerfile was prepared for local deployment. Docker builds were not performed inside the standard Colab environment.

---

## `.dockerignore`

```text
__pycache__
*.pyc
.ipynb_checkpoints
.git
.gitignore
```

---

# PROJECT STRUCTURE

```text
rag-ai-agent/
│
├── app.py
├── agent.py
├── tools.py
├── guardrails.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
│
├── data/
│   └── research_paper.pdf
│
└── tests/
    └── test_agent.py
```

### `app.py`

Contains:

- FastAPI application
- API endpoints
- Request model
- PDF loading
- Chunking
- Embedding initialization
- ChromaDB initialization
- Agent integration

### `agent.py`

Contains:

- Agent decisions
- Structured tool calls
- Planning
- Tool execution

### `tools.py`

Contains:

- Document search
- Calculator

### `guardrails.py`

Contains:

- Input validation
- Basic prompt-injection checks

### `requirements.txt`

Contains the Python dependencies.

### `Dockerfile`

Defines the container image.

---

# INSTALLATION

## Requirements

The main dependencies are:

```text
fastapi
uvicorn
pydantic
langchain
langchain-community
langchain-chroma
langchain-huggingface
langchain-text-splitters
chromadb
sentence-transformers
transformers
torch
pypdf
```

Install:

```bash
pip install -r requirements.txt
```

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/rmanicav/rag-ai-agent.git
```

Enter the project:

```bash
cd rag-ai-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app:app --reload
```

Open:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# EXAMPLES

## Calculator

Request:

```json
{
  "question": "What is 25 * 40?"
}
```

Routing:

```text
CALCULATE
```

Result:

```text
1000
```

---

## RAG Search

Request:

```json
{
  "question": "What is Retrieval-Augmented Generation?"
}
```

Routing:

```text
SEARCH
```

The agent invokes the document-search tool, which retrieves relevant chunks from ChromaDB.

---

## Multi-Tool Request

Request:

```text
What is Retrieval-Augmented Generation,
and what is 25 * 40?
```

Possible plan:

```text
SEARCH
CALCULATE
```

The results can then be combined into a final response.

---

# CURRENT IMPLEMENTATION

The current project intentionally uses deterministic routing for the first agent implementation:

```text
User Question
      |
      v
Rule-Based Decision
      |
      +---- SEARCH
      |
      +---- CALCULATE
      |
      +---- DIRECT
```

This makes the system predictable and easy to test.

The next architectural step is native LLM structured function/tool calling.

---

# LIMITATIONS

This repository is a **portfolio and learning implementation**, not a claim of production readiness.

Current limitations include:

- Rule-based tool selection
- Small local LLM
- Local ChromaDB
- Basic guardrails
- Prototype calculator
- In-memory conversation state
- Basic prompt-injection detection
- Small manually labeled retrieval evaluation set
- Single research-paper knowledge base
- No authentication
- No authorization
- No distributed tracing
- No production deployment
- No automated retraining
- Docker configured but not built inside Colab
- Native LLM function calling is not yet the primary routing mechanism

These limitations are explicitly documented so the scope of the prototype is clear.

---

# PRODUCTION ROADMAP

## RAG

Potential improvements:

- BM25 + vector search
- Reciprocal Rank Fusion
- Query rewriting
- Multi-query retrieval
- Context compression
- Parent-child retrieval
- Adaptive retrieval
- Improved reranking
- Citation verification
- Retrieval caching
- Query caching
- Document versioning
- Automated indexing
- Larger evaluation datasets
- Automated evaluation pipelines

## Agent

Potential improvements:

- Native LLM function calling
- Structured tool schemas
- Better planning
- ReAct-style execution
- Tool permissions
- Tool timeouts
- Parallel tool execution
- Human approval
- Persistent memory
- Long-term memory
- Agent state machines
- Workflow orchestration
- Automated agent evaluation
- Distributed tracing

## Security

Potential improvements:

- Authentication
- Authorization
- Rate limiting
- Stronger prompt-injection defenses
- Tool sandboxing
- Output validation
- Secret management
- Audit logging
- Network restrictions
- Security monitoring

## Infrastructure

Potential improvements:

```text
Docker
   |
   v
CI/CD
   |
   v
Container Registry
   |
   v
Cloud Deployment
   |
   v
Kubernetes / Scaled Infrastructure
```

---

# RAG + AGENT DESIGN PRINCIPLES

The project follows several practical principles.

### Retrieval and generation are separate concerns

A poor retrieval result cannot be fixed reliably by simply asking the LLM to generate a better answer.

### Retrieval should be evaluated independently

Precision, recall, and ranking metrics help identify retrieval problems.

### Retrieved content is untrusted

Documents should be treated as data, not as instructions.

### Tools should be controlled

Agents should not have unrestricted access to powerful operations.

### Deterministic logic has value

Not every part of an AI application needs to be handled by an LLM.

### Observability is essential

Agent systems need visibility into decisions, tools, failures, and latency.

### RAG does not guarantee correctness

Grounding mechanisms reduce risk but require evaluation and monitoring.

---

# FUTURE ARCHITECTURE

The intended evolution is:

```text
Current RAG + Agent Prototype
              |
              v
Native LLM Tool Calling
              |
              v
Improved Agent Planning
              |
              v
Persistent Memory
              |
              v
Advanced RAG Evaluation
              |
              v
OpenTelemetry
              |
              v
Production Security
              |
              v
CI/CD
              |
              v
Cloud Deployment
              |
              v
Scaled Infrastructure
```

---

# CONCLUSION

This repository demonstrates an end-to-end **RAG + AI Agent application** rather than an isolated model experiment.

The implemented system connects:

```text
PDF Ingestion
      ↓
Chunking
      ↓
Embeddings
      ↓
ChromaDB
      ↓
Semantic Retrieval
      ↓
Reranking
      ↓
Retrieval Evaluation
      ↓
Grounded Generation
      ↓
Agent Tool Selection
      ↓
Tool Execution
      ↓
Planning
      ↓
Conversation State
      ↓
Guardrails
      ↓
Error Handling
      ↓
Retries
      ↓
Logging
      ↓
FastAPI
      ↓
Testing
      ↓
Docker
```

The repository is focused specifically on demonstrating practical **Retrieval-Augmented Generation and Agent Engineering** patterns and provides a foundation for extending the prototype into a production-oriented AI application.
