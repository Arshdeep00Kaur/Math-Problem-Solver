# MathProblemSolver

An intelligent mathematical problem-solving system powered by LangGraph and Google's Gemini AI. The system intelligently classifies math queries and routes them through appropriate solving pipelines, with support for conversation checkpointing and multi-threaded sessions.

## Features

- **Intelligent Query Classification**: Automatically classifies math problems as "simple" or "complex"
- **Multi-Agent Architecture**: Uses LangGraph for stateful, graph-based conversational AI
- **Dual Solving Strategy**: 
  - Simple problems: Solved by LLM (Gemini) with step-by-step reasoning
  - Complex problems: Solved by SymPy for precise symbolic computation
- **Knowledge Base**: Upload mathematical documents to build a searchable knowledge base using RAG
- **Checkpointing**: Persists conversation state for resume and replay capabilities
- **Thread Management**: Supports multiple concurrent conversation threads with unique thread IDs
- **Vector Database**: Qdrant integration for semantic search over mathematical documents

## Architecture

The system uses a graph-based architecture with the following components:

### Nodes
- **Classifier Node** ([nodes/classifier.py](backend/nodes/classifier.py)): Analyzes queries and routes them to appropriate solvers
- **Simple Solver Node** ([nodes/simple.py](backend/nodes/simple.py)): Uses LLM (Gemini) to solve basic to intermediate math problems with natural language reasoning
- **Complex Solver Node**: Uses SymPy for symbolic computation, handling complex calculations and high-precision math

### State Management
- **ChatState** ([models/chat.py](backend/models/chat.py)): TypedDict-based state with message history, routing info, and current question
- **Checkpointing**: Conversation states are persisted for continuity across sessions
- **Thread IDs**: Each conversation thread has a unique identifier for parallel sessions

### API
- **Embeddings API** ([api/embeddings.py](backend/api/embeddings.py)): FastAPI endpoint for uploading mathematical documents to build the knowledge base (not for user queries)

## Problem Categories

### Simple Problems (Solved by LLM)
The system uses Gemini AI to solve these types with natural language reasoning:
- Basic calculations
- Symbolic reasoning
- Language-to-math translation
- Arithmetic problems
- Algebraic problems
- Word problems
- Geometry problems
- Probability and statistics
- Calculus (conceptual or procedural)
- Logical or puzzle-based problems
- Optimization and decision problems

### Complex Problems (Solved by SymPy)
Routed to SymPy for precise symbolic computation:
- Large or long numerical calculations
- Very complex integrals
- High-precision engineering or scientific math
- Advanced symbolic manipulation
- Exact computation requiring symbolic math engine

## Technology Stack

- **LangGraph**: Graph-based conversation orchestration
- **LangChain**: LLM framework and document processing
- **Google Gemini 2.0 Flash**: AI model for classification and solving simple problems
- **SymPy**: Python symbolic mathematics library for complex problem solving
- **Qdrant**: Vector database for knowledge base embeddings
- **FastAPI**: REST API framework
- **Google Generative AI Embeddings**: Document vectorization for knowledge base

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for Qdrant vector database)
- Google API Key (for Gemini AI)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd MathProblemSolver
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

5. **Start Qdrant vector database**
```bash
cd backend/static
docker-compose up -d
```

This will start Qdrant on `http://localhost:6333`

## Usage

### Starting the API Server

```bash
cd backend
python -m uvicorn api.embeddings:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Upload Document to Knowledge Base
```bash
POST /api/doc_embeddings
Content-Type: multipart/form-data

# Upload a text file to build the mathematical knowledge base
# This is for reference material, not for solving user queries
curl -X POST "http://localhost:8000/api/doc_embeddings" \
  -F "file=@math_reference_material.txt"
```

**Response:**
```json
{
  "filename": "math_reference_material.txt",
  "chunks": 42,
  "collection": "math_uploaded_doc"
}
```

**Note:** This endpoint is for building a searchable knowledge base of mathematical concepts, formulas, and reference material. User queries are solved directly by the LLM or SymPy, not by uploading documents.

### Using the Math Solver (Python)

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from nodes.classifier import query_classifier
from nodes.simple import simple
from models.chat import ChatState
import uuid

# Initialize checkpointer for conversation persistence
checkpointer = MemorySaver()

# Build the graph
graph = StateGraph(ChatState)
graph.add_node("classifier", query_classifier)
graph.add_node("simple_solver", simple)

# Define routing logic
graph.set_entry_point("classifier")
graph.add_conditional_edges(
    "classifier",
    lambda state: state["route"],
    {
        "simple": "simple_solver",
        "complex": "complex_solver"  # Add when implemented
    }
)

# Compile with checkpointer
app = graph.compile(checkpointer=checkpointer)

# Create a unique thread ID for this conversation
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# Run a query
result = app.invoke({
    "messages": [],
    "question": "What is the derivative of x^2 + 3x + 5?",
    "route": None
}, config=config)

print(result)

# Continue the conversation using the same thread_id
result2 = app.invoke({
    "messages": result["messages"],
    "question": "Now integrate that result",
    "route": None
}, config=config)
```

## Checkpointing and Thread Management

### Checkpointing
The system uses LangGraph's checkpointing feature to:
- Persist conversation state after each step
- Resume conversations from any checkpoint
- Replay and debug conversation flows
- Implement time-travel debugging

### Thread IDs
Each conversation thread has a unique identifier that:
- Isolates conversations from each other
- Enables parallel processing of multiple users
- Maintains conversation history per thread
- Allows retrieval of past conversations

**Example with multiple threads:**
```python
# User 1 conversation
thread_1 = str(uuid.uuid4())
config_1 = {"configurable": {"thread_id": thread_1}}
result_1 = app.invoke(state_1, config=config_1)

# User 2 conversation (completely isolated)
thread_2 = str(uuid.uuid4())
config_2 = {"configurable": {"thread_id": thread_2}}
result_2 = app.invoke(state_2, config=config_2)

# Resume User 1's conversation later
result_1_continued = app.invoke(state_1_new, config=config_1)
```

## Project Structure

```
MathProblemSolver/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── embeddings.py          # FastAPI document embedding endpoints
│   ├── models/
│   │   └── chat.py                # ChatState TypedDict definition
│   ├── nodes/
│   │   ├── classifier.py          # Query classification node
│   │   └── simple.py              # Simple problem solver node
│   └── static/
│       └── docker-compose.yml     # Qdrant vector database setup
├── venv/                          # Virtual environment
├── .env                           # Environment variables (not in git)
├── .gitignore
└── README.md
```

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google Generative AI API key

### Model Settings
- **Classification Model**: `gemini-2.0-flash-exp` (temperature=0 for consistency)
- **Embedding Model**: `models/gemini-embedding-001`
- **Vector Store**: Qdrant at `http://localhost:6333`
- **Collection Name**: `math_uploaded_doc`

## Development

### Adding New Solver Nodes
1. Create a new Python file in `backend/nodes/`
2. Define a function that takes `ChatState` and returns `ChatState`
3. Add the node to your LangGraph
4. Update the routing logic in the classifier

### Customizing Classification
Edit the classification prompt in [nodes/classifier.py](backend/nodes/classifier.py) to adjust routing logic.



## Troubleshooting

### Qdrant Connection Issues
Ensure Docker is running and Qdrant is accessible:
```bash
curl http://localhost:6333/collections
```

### Google API Errors
Verify your API key is valid and has Generative AI API enabled.

### Import Errors
Make sure you're running from the backend directory or adjust your PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/MathProblemSolver/backend"
```
