# RAG AI Agent — End-to-End AI Engineering Project

An end-to-end AI Engineering portfolio project that evolves a document-based Retrieval-Augmented Generation (RAG) system into a tool-using AI agent. The project covers the complete path from document ingestion and embeddings to retrieval, reranking, evaluation, hallucination handling, hybrid search, agent planning, tools, memory, guardrails, reliability, observability, FastAPI, testing, and Docker.

The system uses the research paper **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** as its primary knowledge source and uses the open-source `Qwen/Qwen2.5-0.5B-Instruct` model for generation.

---

## What This Project Is

The project demonstrates how modern AI applications are built as systems rather than as a single model.

At a high level:

```text
User
  |
  v
FastAPI
  |
  v
Input Validation / Guardrails
  |
  v
Agent / Planner
  |
  +-------------------+-------------------+
  |                   |                   |
  v                   v                   v
RAG Search        Calculator          Direct LLM
  |                   |
  v                   |
Vector Database       |
  |                   |
  v                   |
Reranking              |
  |                   |
  +---------+----------+
            |
            v
      Tool / Context Results
            |
            v
           LLM
            |
            v
      Final Response
            |
            v
 Logging / Monitoring
```

The project intentionally separates retrieval, reasoning, tools, validation, API handling, and observability so that each part can be tested and improved independently.

---

## Core Capabilities

The application demonstrates:

- PDF document ingestion
- Text extraction
- Chunking and chunk-size experimentation
- Metadata preservation
- Sentence-transformer embeddings
- 384-dimensional vector representations
- ChromaDB vector storage
- Semantic similarity search
- Retrieval-quality thresholds
- Cross-encoder reranking
- Precision@K
- Recall@K
- Mean Reciprocal Rank
- Faithfulness and groundedness concepts
- Answer relevance
- Context relevance
- Hallucination testing
- Metadata filtering
- Hybrid keyword + vector retrieval
- Retrieval latency measurement
- RAG answer generation
- AI agent decision making
- Structured tool calls
- Document-search tool
- Calculator tool
- Direct LLM path
- Multi-step planning
- Multi-tool execution
- Conversation state
- Input guardrails
- Prompt-injection protection
- Tool error handling
- Retry logic
- Logging
- API health checks
- FastAPI REST API
- Automated API testing
- Docker configuration
- Production architecture planning

---

## Why RAG?

Large language models have knowledge encoded in their parameters, but an application often needs to answer questions using a specific knowledge source.

RAG separates the problem into two stages:

```text
Question
   |
   v
Retrieve relevant information
   |
   v
Provide information to the LLM
   |
   v
Generate an answer
```

This makes it possible to build applications around private, domain-specific, or frequently changing information without retraining the language model every time the knowledge base changes.

RAG does not automatically eliminate hallucinations. Retrieval quality, context quality, prompting, model behavior, and answer validation all matter.

---

# RAG PIPELINE

## Document Ingestion

The initial knowledge source is the research paper:

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**

The PDF is loaded with LangChain's `PyPDFLoader`.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(pdf_path)
documents = loader.load()
```

Each document contains both page content and metadata, which later allows retrieved information to be associated with its source page.

---

## Chunking

A complete PDF page or document is generally too large and too broad to retrieve as one unit. The document is therefore divided into overlapping chunks.

The main configuration used in the project is:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)
```

This produced approximately **111 chunks** for the selected document.

Chunking involves a trade-off:

```text
Smaller chunks
    -> more precise retrieval
    -> more chunks
    -> potentially less surrounding context

Larger chunks
    -> more context
    -> fewer chunks
    -> potentially more irrelevant information
```

The project compared:

```text
400 / 80   -> 205 chunks
800 / 150  -> 111 chunks
1200 / 200 -> 73 chunks
```

The `800 / 150` configuration was used for the main prototype.

---

## Embeddings

Text is converted into numerical vectors using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Implementation:

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
[0.12, -0.31, 0.82, ...]
```

Semantically related text should generally produce vectors that are closer in the embedding space.

---

## Vector Database

ChromaDB is used to store and retrieve document embeddings.

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_paper"
)
```

The vector database provides the retrieval layer of the RAG system.

---

## Semantic Retrieval

A basic semantic search looks like:

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
Similarity Search
   |
   v
Relevant Document Chunks
```

Unlike simple keyword matching, semantic retrieval can find text that expresses a related concept using different words.

---

## Retrieval Distance and Quality Gating

Chroma's distance metric is interpreted so that, for this configuration:

```text
Lower distance = more similar
Higher distance = less similar
```

During experimentation, relevant queries produced distances around:

```text
0.85
0.89
0.91
```

An unrelated question such as:

```text
What is the capital of Australia?
```

produced a much weaker best-match distance of approximately:

```text
1.78
```

A simple demonstration threshold of `1.2` was therefore tested.

The important engineering lesson is that a threshold such as `1.2` is **not universal**. It depends on the embedding model, distance metric, document collection, and evaluation data. Production thresholds should be calibrated empirically.

A retrieval gate can follow this pattern:

```text
Question
   |
   v
Retrieve
   |
   v
Is the context sufficiently relevant?
       / \
     Yes  No
      |    |
      v    v
     LLM  Explain that
          knowledge is
          insufficient
```

---

## RAG Generation

Once relevant chunks are retrieved, they are placed into the LLM context.

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

with Hugging Face Transformers.

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=300,
    temperature=0.1
)
```

The small open-source model was selected so the project could be developed without requiring paid API usage.

---

## Source Attribution

Retrieved chunks preserve page metadata:

```python
{
    "content": doc.page_content,
    "page": doc.metadata.get("page")
}
```

This makes it possible to expose source pages with answers.

Source attribution is useful for:

- Trust
- Debugging
- Research
- Human verification
- Auditability

---

# RETRIEVAL QUALITY

## Reranking

Initial vector retrieval is useful for quickly producing candidate documents. A second-stage reranker can improve the ordering of those candidates.

The project uses:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The retrieval architecture becomes:

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
Cross-Encoder Reranker
   |
   v
Top 3 Ranked Results
```

The vector search stage is optimized for candidate retrieval, while the cross-encoder performs a more detailed question-document relevance assessment.

---

## Retrieval Evaluation

Retrieval quality should be measured separately from generation quality.

The project introduced a small manually labeled evaluation set:

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

These labels are a small learning/portfolio evaluation set rather than a production benchmark.

### Precision@K

```text
Precision@K =
Relevant Retrieved Documents / K
```

It measures how many retrieved results are relevant.

### Recall@K

```text
Recall@K =
Relevant Retrieved Documents / Total Relevant Documents
```

It measures how much of the relevant information was successfully retrieved.

### MRR

Mean Reciprocal Rank measures how early the first relevant result appears.

```text
MRR = 1 / Rank of First Relevant Result
```

These metrics help determine whether retrieval itself is working before judging the LLM.

---

## Answer Evaluation

A strong RAG evaluation should consider several dimensions.

### Faithfulness / Groundedness

Does the answer follow the retrieved evidence?

### Answer Relevance

Does the answer actually address the question?

### Context Relevance

Was the retrieved context relevant to the question?

The overall evaluation chain is:

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

A simple word-overlap faithfulness checker was explored as a learning exercise. It demonstrates the idea but is not sufficient for production evaluation.

Production systems can use:

- LLM-as-judge evaluation
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

This exposed an important RAG failure mode: an LLM may still produce an answer even when the retrieved context does not contain supporting evidence.

The retrieval-quality gate therefore becomes an important defensive layer.

The desired behavior is:

```text
Relevant context
    -> Answer using context

Insufficient context
    -> Do not pretend the knowledge base supports the answer
```

RAG reduces hallucination risk; it does not guarantee hallucination-free behavior.

---

# RETRIEVAL IMPROVEMENTS

## Metadata Filtering

Document metadata was used to describe the knowledge source:

```text
document_type = research_paper
topic = RAG
language = English
```

Metadata filtering can narrow the search space before semantic retrieval:

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

This becomes increasingly valuable when the knowledge base contains many documents.

---

## Hybrid Search

Semantic search is powerful, but exact keyword matching is useful for:

- Names
- Product IDs
- Technical terms
- Acronyms
- Exact phrases

The project therefore explored hybrid retrieval:

```text
Keyword Search
      +
Vector Search
      |
      v
Combined Ranking
```

A simple experiment combined TF-IDF with Chroma similarity.

A production implementation could use:

```text
BM25
  +
Vector Search
  +
Reciprocal Rank Fusion (RRF)
```

The key idea is to combine lexical and semantic retrieval rather than relying on only one strategy.

---

## RAG Performance

RAG latency can be decomposed into:

```text
Retrieval Latency
      +
Reranking Latency
      +
Generation Latency
      =
Total Latency
```

Useful metrics include:

- p50 latency
- p95 latency
- p99 latency
- Retrieval latency
- Reranking latency
- Generation latency
- Total request latency

Potential optimizations include:

- Smaller retrieval `k`
- Smaller reranking candidates
- Caching
- Batching
- GPU inference
- Smaller LLMs
- Streaming responses
- Efficient vector indexes

---

# AI AGENT

## From RAG to an Agent

The RAG system was extended into an agent.

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
Choose Action
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

The important architectural change is that the system can now decide what action to take instead of always performing the same retrieval pipeline.

---

## Agent Tools

The prototype provides three logical paths:

```text
search_documents
calculate
direct_llm
```

The document search is a RAG tool.

The calculator is a deterministic tool.

The direct LLM path handles questions that do not require either tool.

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

The tool hides the details of vector retrieval behind a simple interface.

---

## Calculator Tool

The prototype calculator sanitizes a basic arithmetic expression:

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

This is intentionally a learning prototype.

A production implementation should use a dedicated safe expression parser rather than `eval`, even with restricted built-ins.

---

## Agent Decision Making

The current prototype uses deterministic routing logic.

Possible decisions:

```text
SEARCH
CALCULATE
DIRECT
```

Example:

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

This approach makes the first agent implementation easy to understand and test.

The production roadmap replaces this routing with native LLM structured tool/function calling.

---

## Structured Tool Calls

The agent represents an action using a structured object:

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

This separates tool selection from tool execution.

---

## Tool Execution

The architecture is:

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

This modularity makes it easier to add additional tools later.

Potential future tools could include:

- Database queries
- REST APIs
- File operations
- Search services
- Business systems
- Data analysis
- Code execution in a sandbox

---

## Planning and Multi-Step Execution

The prototype can identify multiple actions for a combined question.

Example:

```text
What is Retrieval-Augmented Generation,
and what is 25 * 40?
```

The plan can become:

```text
1. SEARCH
2. CALCULATE
```

The architecture is:

```text
                    User Question
                          |
                          v
                       Planner
                       /     \
                      /       \
                     v         v
                  Search    Calculator
                     |         |
                     +----+----+
                          |
                          v
                       Results
                          |
                          v
                    Final Answer
```

This demonstrates the fundamental idea behind multi-step agent workflows.

---

## Agent Loop

A general agent loop is:

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

A production implementation can add explicit state, step limits, tool permissions, timeouts, and termination conditions.

---

# MEMORY, SECURITY, AND RELIABILITY

## Conversation Memory

Basic short-term state was implemented using:

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

This demonstrates conversation state.

Production memory could use:

- Redis
- PostgreSQL
- A conversation database
- Dedicated memory infrastructure

Long-term memory should be designed carefully and should not automatically retain unnecessary or sensitive information.

---

## Guardrails

Input guardrails are applied before agent execution.

The prototype checks:

- Empty questions
- Excessively long questions
- Basic prompt-injection patterns

Examples of blocked patterns include:

```text
ignore previous instructions
reveal system prompt
show your system prompt
```

Architecture:

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

Guardrails should be treated as one layer of a defense-in-depth security architecture rather than a complete security solution.

---

## Prompt Injection

Prompt injection is especially important in RAG and agent systems because the application may process untrusted text.

A fundamental rule is:

```text
Retrieved documents are DATA.
Retrieved documents are NOT instructions.
```

Untrusted content can originate from:

- User input
- Documents
- Web pages
- Tool results
- External APIs
- Memory

The application must preserve the priority of trusted system/application instructions and constrain what tools can do.

---

## Tool Security

Tools should have explicit permissions.

For production systems:

```text
Agent
 |
 v
Tool Permission Check
 |
 +---- Allowed ----> Execute
 |
 +---- Denied -----> Reject
```

High-risk tools should require additional controls such as:

- Authentication
- Authorization
- Human approval
- Sandboxing
- Rate limits
- Resource limits
- Audit logging

---

## Error Handling

Tools can fail because of:

- Invalid arguments
- Network errors
- Database errors
- Service outages
- Model failures
- Timeouts

The agent should handle failures without crashing the entire application.

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

## Retry Strategy

The prototype includes a basic retry mechanism with a maximum of three attempts.

```text
Tool Call
   |
   v
Attempt 1
   |
   +-- Success --> Continue
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

Retries should be used selectively.

A production system should distinguish between:

```text
Transient error  -> potentially retry
Invalid input    -> do not retry
Permission error -> do not retry
Timeout          -> potentially retry
Permanent error  -> do not repeatedly retry
```

Production retries should normally include exponential backoff and appropriate timeout limits.

---

# OBSERVABILITY AND EVALUATION

## Logging

Python logging is used to make agent execution visible.

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

## Observability

A production AI application should be observable at multiple levels:

```text
Request
  |
  +-- Agent decision
  |
  +-- Retrieval
  |
  +-- Reranking
  |
  +-- Tool calls
  |
  +-- LLM generation
  |
  +-- Final response
```

Important metrics include:

- Request count
- Error rate
- Tool-call count
- Tool failure rate
- Retrieval latency
- Reranking latency
- LLM latency
- p50 latency
- p95 latency
- Token usage
- Cost
- Retrieval quality
- Answer quality
- Task completion rate

Future observability can use OpenTelemetry and centralized metrics/tracing systems.

---

## Agent Evaluation

Agent behavior should be evaluated separately from ordinary API tests.

Example:

```text
Question:
What is Retrieval-Augmented Generation?

Expected tool:
search_documents
```

```text
Question:
What is 25 * 40?

Expected tool:
calculate
```

```text
Question:
What is the capital of Germany?

Expected:
direct_llm
```

A basic metric is:

```text
Tool Selection Accuracy
```

More complete evaluation can measure:

- Task completion
- Tool-selection accuracy
- Tool-call efficiency
- Retrieval quality
- Final-answer quality
- Groundedness
- Failure recovery
- Latency
- Cost

---

# API AND SOFTWARE ENGINEERING

## FastAPI

The agent is exposed as a REST API using FastAPI.

Endpoints:

```text
GET  /
GET  /health
POST /agent
```

The application uses Pydantic request validation.

Example request model:

```python
from pydantic import BaseModel

class AgentRequest(BaseModel):
    question: str
```

---

## Root Endpoint

```http
GET /
```

Example response:

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

A health endpoint provides a simple application-level health signal.

---

## Agent Endpoint

```http
POST /agent
```

Example:

```json
{
  "question": "What is Retrieval-Augmented Generation?"
}
```

The request flows through:

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

## Automated API Testing

FastAPI's `TestClient` is used for automated tests.

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
- Search behavior
- Calculator behavior
- Invalid requests
- Guardrail behavior
- Tool selection
- Error handling

---

# CONTAINERIZATION AND REPOSITORY

## Docker

The project includes a Dockerfile:

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

The Dockerfile was prepared during development, but the Docker build was not executed inside the standard Google Colab environment.

---

## Docker Ignore

```text
__pycache__
*.pyc
.ipynb_checkpoints
.git
.gitignore
```

---

## Project Structure

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

Contains the FastAPI application, API endpoints, request validation, document loading, chunking, embedding initialization, vector-store initialization, and agent integration.

### `agent.py`

Contains agent decision logic, structured tool calls, planning, and tool execution.

### `tools.py`

Contains the document-search and calculator tools.

### `guardrails.py`

Contains input validation and basic prompt-injection checks.

### `requirements.txt`

Contains Python dependencies.

### `Dockerfile`

Defines the application container.

---

## Dependencies

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

## `.gitignore`

Recommended:

```gitignore
__pycache__/
*.py[cod]

.ipynb_checkpoints/

.venv/
venv/
env/

*.log
.pytest_cache/

.cache/
huggingface/
transformers/

chroma/
chroma_db/

.env
*.key
*.pem
```

Secrets, local environments, model caches, and generated vector databases should not be committed to Git.

---

# RUNNING THE PROJECT

## Local Installation

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

Run:

```bash
uvicorn app:app --reload
```

Open the API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Example Calculator Request

```json
{
  "question": "What is 25 * 40?"
}
```

Expected routing:

```text
CALCULATE
```

Expected result:

```text
1000
```

---

## Example RAG Request

```json
{
  "question": "What is Retrieval-Augmented Generation?"
}
```

Expected routing:

```text
SEARCH
```

The application retrieves relevant chunks from ChromaDB and uses them as context.

---

# PRODUCTION ENGINEERING

## Current Prototype vs Production System

The current application is a learning and portfolio prototype.

It intentionally demonstrates the major architecture without claiming to be production-ready.

Current limitations include:

- Rule-based tool selection
- Small local LLM
- Local ChromaDB
- Basic guardrails
- Prototype calculator
- In-memory conversation state
- No authentication
- No authorization
- No distributed deployment
- No production tracing
- Small manual retrieval evaluation set
- Single research-paper knowledge base
- No automated retraining
- Docker configured but not built inside Colab
- Native LLM function calling is not yet the primary routing mechanism

Documenting these limitations is part of the engineering approach.

---

## Production Architecture

A production implementation could evolve into:

```text
                         Client
                           |
                           v
                     API Gateway
                           |
                           v
                    Authentication
                           |
                           v
                      Agent Service
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
         Retriever       Tools       LLM Service
             |
             v
       Vector Database
             |
             v
        Document Store
```

Supporting infrastructure could include:

```text
Redis
PostgreSQL
Object Storage
OpenTelemetry
Metrics
Logging
CI/CD
Container Registry
Kubernetes
Cloud Infrastructure
```

---

## Production RAG Improvements

A production RAG system could add:

- BM25 + vector retrieval
- Reciprocal Rank Fusion
- Query rewriting
- Multi-query retrieval
- Context compression
- Parent-child retrieval
- Adaptive retrieval
- Better reranking
- Citation verification
- Retrieval caching
- Query caching
- Document versioning
- Automated indexing
- Larger evaluation datasets
- Automated evaluation pipelines

---

## Production Agent Improvements

The agent can be extended with:

- Native LLM function calling
- Structured tool schemas
- ReAct-style execution
- Better planning
- Tool permissions
- Tool timeouts
- Parallel tool execution
- Human approval steps
- Persistent memory
- Long-term memory
- Agent state machines
- Workflow orchestration
- Automated agent evaluation
- Distributed tracing

---

## Native Function Calling Roadmap

The current implementation uses deterministic routing.

A more advanced architecture is:

```text
User
 |
 v
LLM
 |
 v
Structured Tool Call
 |
 v
Tool Executor
 |
 v
Tool Result
 |
 v
LLM
 |
 v
Final Answer
```

For example:

```json
{
  "tool": "search_documents",
  "arguments": {
    "query": "What is Retrieval-Augmented Generation?"
  }
}
```

This allows the LLM to select tools based on semantic understanding rather than only keyword rules.

---

## Memory Roadmap

Current:

```text
In-memory conversation_history
```

Future:

```text
User
 |
 v
Agent
 |
 +-- Short-Term State
 |
 +-- Long-Term Memory
 |
 +-- Conversation History
 |
 +-- Task State
```

Potential technologies include Redis and PostgreSQL.

Memory should be governed by explicit retention and privacy rules.

---

## Reliability Architecture

Production systems should consider:

```text
Timeouts
   +
Retries
   +
Exponential Backoff
   +
Fallbacks
   +
Circuit Breakers
   +
Input Validation
   +
Output Validation
   +
Monitoring
```

The correct response to an error depends on the error type.

---

# MODEL ENGINEERING

## Current Model

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model was selected because it is:

- Open source
- Small enough for experimentation
- Suitable for a free development environment
- Easy to run with Hugging Face Transformers

The trade-off is reduced reasoning and tool-calling reliability compared with larger models.

Production model selection should consider:

- Accuracy
- Latency
- Cost
- Context length
- Tool-calling support
- Hosting requirements
- Privacy
- Hardware availability

---

## LLM Reliability

A smaller model can struggle with:

- Complex reasoning
- Long contexts
- Multi-step planning
- Reliable structured outputs
- Tool selection

Therefore the current system intentionally combines deterministic software logic with the LLM.

This is an important engineering principle:

> Use deterministic software where deterministic behavior is required, and use the LLM where language understanding and generation add value.

---

# SECURITY

## Defense in Depth

AI application security should not rely on a single prompt.

A stronger architecture is:

```text
Input Validation
      |
      v
Authentication
      |
      v
Authorization
      |
      v
Prompt-Injection Defense
      |
      v
Tool Permission Checks
      |
      v
Sandboxing
      |
      v
Output Validation
      |
      v
Audit Logging
```

Security controls should be implemented at the application and infrastructure levels.

---

# OVERALL LEARNING AND ENGINEERING COVERAGE

This project builds on a broader AI Engineering foundation.

The machine-learning foundation covered:

- Supervised learning
- Unsupervised learning
- Classification
- Regression
- Training/validation/test splits
- Overfitting
- Underfitting
- Bias and variance
- Regularization
- Cross-validation
- Data leakage
- Feature engineering
- Feature scaling
- Hyperparameter tuning
- Precision
- Recall
- F1
- ROC-AUC
- Average Precision
- Imbalanced datasets
- Missing values
- Outliers
- Decision trees
- Random forests
- Gradient boosting
- XGBoost
- Model persistence
- Model deployment
- MLOps
- Monitoring
- Data drift

Deep-learning coverage included:

- Neural networks
- Forward propagation
- Backpropagation
- Gradient descent
- Loss functions
- BCEWithLogitsLoss
- Optimizers
- Learning-rate scheduling
- Batch normalization
- Dropout
- Early stopping
- Checkpoints
- GPU training
- Transfer learning
- CNNs
- RNNs
- LSTMs
- GRUs
- Transformers
- Attention

LLM engineering coverage included:

- Tokenization
- Embeddings
- Self-attention
- Multi-head attention
- Query/key/value
- Positional encoding
- Encoder/decoder architecture
- Autoregressive generation
- Context windows
- Temperature
- Prompt engineering
- Structured outputs
- Tool calling
- Hallucination
- RAG
- Fine-tuning
- LoRA
- QLoRA
- Quantization
- Inference optimization

The RAG and agent implementation connects these concepts into a single working architecture.

---

# INTERVIEW-READY EXPLANATION

A concise explanation of the project:

> I built an end-to-end AI agent that combines Retrieval-Augmented Generation with tool execution. I started by loading a research paper, splitting it into overlapping chunks, generating embeddings with Sentence Transformers, and storing those embeddings in ChromaDB. I implemented semantic retrieval and then added retrieval-quality thresholds, cross-encoder reranking, metadata filtering, hybrid search, retrieval evaluation, and basic hallucination and faithfulness checks.
>
> I then extended the RAG pipeline into an agent. The agent can select between document retrieval, a calculator, and direct LLM processing. I implemented structured tool calls, multi-step planning, conversation state, guardrails, prompt-injection protection, error handling, retries, logging, evaluation, and a FastAPI REST interface.
>
> The application is modularized into agent, tool, guardrail, and API components and includes automated API tests and Docker configuration. The current implementation is a portfolio prototype. The next production steps would be native LLM function calling, persistent memory, stronger evaluation, OpenTelemetry, authentication and authorization, improved security, CI/CD, and cloud deployment.

---

# KEY INTERVIEW QUESTIONS

### What is RAG?

RAG combines retrieval and generation. Relevant external information is retrieved and provided to the LLM as context before generating an answer.

### Why use embeddings?

Embeddings represent text as vectors that capture semantic relationships, enabling semantic similarity search.

### Why use a vector database?

A vector database provides efficient similarity search over document embeddings.

### Why rerank?

Initial retrieval quickly generates candidates. A reranker can improve the ordering and relevance of those candidates.

### Does RAG eliminate hallucinations?

No. RAG can reduce hallucination risk by providing evidence, but retrieval errors, irrelevant context, model behavior, and prompt design can still produce unsupported answers.

### What is an AI agent?

An AI agent is a system that can decide which actions or tools to use to accomplish a task.

### What is tool calling?

Tool calling allows an AI system to request execution of an external function using structured arguments.

### Why use deterministic routing in this prototype?

It makes the initial agent behavior predictable and testable. Native LLM function calling can be introduced once the basic architecture is understood.

### What is prompt injection?

Prompt injection is an attack in which untrusted content attempts to manipulate an AI system into violating its intended instructions.

### Why are guardrails needed?

Guardrails constrain inputs, outputs, and tool execution to improve reliability and security.

### Why are retries needed?

Retries can recover from transient failures, but they should not be used for permanent failures or invalid requests.

### Why is observability important?

It provides visibility into agent decisions, tool execution, latency, failures, and overall system behavior.

### How would you productionize this system?

I would add native structured tool calling, stronger RAG evaluation, persistent memory, authentication, authorization, tool permissions, timeouts, exponential backoff, distributed tracing, metrics, CI/CD, container orchestration, cloud infrastructure, and stronger security controls.

---

# FINAL SYSTEM VIEW

```text
                                USER
                                  |
                                  v
                           ┌─────────────┐
                           │   FastAPI   │
                           └──────┬──────┘
                                  |
                                  v
                           ┌─────────────┐
                           │ Guardrails  │
                           └──────┬──────┘
                                  |
                                  v
                           ┌─────────────┐
                           │    Agent    │
                           │   Planner   │
                           └──────┬──────┘
                                  |
              +-------------------+-------------------+
              |                   |                   |
              v                   v                   v
       ┌──────────────┐   ┌─────────────┐   ┌─────────────┐
       │  RAG Search  │   │ Calculator  │   │  Direct LLM │
       └──────┬───────┘   └─────────────┘   └─────────────┘
              |
              v
       ┌──────────────┐
       │   ChromaDB   │
       └──────┬───────┘
              |
              v
       ┌──────────────┐
       │   Reranker   │
       └──────┬───────┘
              |
              v
       ┌──────────────┐
       │    Context   │
       └──────┬───────┘
              |
              v
       ┌──────────────┐
       │     Qwen     │
       │     LLM      │
       └──────┬───────┘
              |
              v
        Final Response
              |
              v
       Logging / Metrics
```

---

# ROADMAP

The planned evolution is:

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
Kubernetes / Scaled Infrastructure
```

---

# Conclusion

This project demonstrates the progression from machine-learning fundamentals to a practical AI Engineering system:

```text
Machine Learning
       ↓
Deep Learning
       ↓
Transformers
       ↓
LLMs
       ↓
Embeddings
       ↓
Vector Databases
       ↓
RAG
       ↓
Reranking
       ↓
Retrieval Evaluation
       ↓
Hybrid Search
       ↓
AI Agents
       ↓
Tool Calling
       ↓
Planning
       ↓
Memory
       ↓
Guardrails
       ↓
Reliability
       ↓
Observability
       ↓
FastAPI
       ↓
Testing
       ↓
Docker
       ↓
Production AI Architecture
```

The primary goal is to demonstrate practical AI Engineering through a complete, modular system rather than isolated model experiments.

---

## License

This project is intended for educational and portfolio purposes.
