# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A LangChain v1.0 chatbot that analyzes mid-level (중니어) developer concerns, built with Streamlit and Google GenAI. The project uses 3000+ developer reflections and retrospectives to provide consultation on career growth challenges.

**중니어 (Mid-level)**: Korean term for developers between junior and senior levels who face unique growth challenges.

## Technology Stack

- **LangChain v1.0+**: Core framework (v1.0.2+) with ReAct Agent pattern
- **LangGraph**: Agent orchestration with tools, middleware, and context injection
- **Google GenAI (Gemini 2.5 Flash Lite)**: LLM provider via `langchain-google-genai` (v3.0.0+)
- **Streamlit**: UI framework (v1.50.0+) with multi-page app and real-time token streaming
- **Pinecone**: Vector database for semantic search
- **FalkorDB**: Graph database for keyword-based relationship analysis
- **Upstage Solar**: Korean-optimized embeddings (4096 dimensions)
- **DuckDuckGo Search**: Web search tool for latest information
- **Python 3.13**: Required minimum version
- **uv**: Dependency management tool

## Development Commands

### Environment Setup
```bash
# Install dependencies with uv
uv sync

# Create .env file with required API keys
cp .env.example .env
# Add: UPSTAGE_API_KEY, PINECONE_API_KEY, GEMINI_API_KEY, PINECONE_INDEX_NAME
# Add: FALKORDB_HOST, FALKORDB_PORT (optional, defaults: localhost, 6379)

# Activate virtual environment (optional, uv handles this)
source .venv/bin/activate  # macOS/Linux
```

### Building Vector Store
```bash
# Build/rebuild Pinecone vector store (run once or when data changes)
python -m scripts.build_vectorstore

# This will:
# - Create Pinecone index if not exists
# - Generate Upstage embeddings for all 3000 CSV records
# - Upload vectors to Pinecone with namespace "20251029_crawling"
# - Verify with sample search
```

### Building Graph Database
```bash
# Start FalkorDB (Docker)
docker run -d -p 6379:6379 falkordb/falkordb:latest

# Build/rebuild FalkorDB graph database (run once or when data changes)
python -m scripts.build_graphdb

# This will:
# - Connect to FalkorDB (localhost:6379)
# - Create graph schema with indexes
# - Fetch metadata from Pinecone namespace "20251029_crawling"
# - Build graph with Document, Keyword, Category nodes
# - Create relationships: HAS_KEYWORD, BELONGS_TO, CO_OCCURS_WITH
# - Display graph statistics

# Test graph queries
python -m utils.graph_queries
```

### Running the Application
```bash
# Run Streamlit app
streamlit run main.py

# The app has multiple pages:
# - main.py: Profile/concern registration
# - pages/chatbot.py: AI consultation with ReAct Agent (✅ Implemented with streaming)
# - pages/search.py: Semantic search (WIP)

# Environment variables required:
# - UPSTAGE_API_KEY: Upstage Solar embeddings
# - PINECONE_API_KEY: Pinecone vector database
# - GOOGLE_API_KEY: Google Gemini API
# - PINECONE_INDEX_NAME: Pinecone index name (default: mid-level-helper)
```

### Testing and Quality
```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_retriever.py

# Lint and format code
ruff check .
ruff format .
```

## Data Structure

**CSV File**: `./data/mid_level_data_unique_3000.csv` (~3000 entries)

Columns (Korean headers):
- `글 제목` (Title): Post title
- `출처` (Source): URL source
- `핵심 키워드` (Keywords): Core keywords (e.g., "성장통, 재택근무")
- `문제점 요약` (Problem Summary): Brief problem description with categorization
- `글 내용 요약` (Content Summary): Detailed content summary

**Common Issue Categories** (from keywords/summaries):
- 성장통 (Growing pains)
- 성장 슬럼프 (Growth slump)
- 경력 정체 (Career stagnation)
- 기술 부채 (Technical debt)
- 개발 문화 (Development culture)

**Graph Database Structure** (FalkorDB):

Node Types:
1. **Document** (문서 노드)
   - Properties: `id`, `title`, `source`, `problem_summary`, `category`
   - Represents each developer case/post

2. **Keyword** (키워드 노드)
   - Properties: `name`
   - Represents extracted keywords from posts

3. **Category** (카테고리 노드)
   - Properties: `name`
   - Represents issue categories

Relationship Types:
1. **HAS_KEYWORD**: (Document)-[HAS_KEYWORD]->(Keyword)
   - Links documents to their keywords

2. **BELONGS_TO**: (Document)-[BELONGS_TO]->(Category)
   - Links documents to their categories

3. **CO_OCCURS_WITH**: (Keyword)-[CO_OCCURS_WITH {weight}]-(Keyword)
   - Links keywords that appear together in documents
   - Property: `weight` (co-occurrence frequency)

## Architecture

### Core Components

**main.py**: Entry point with cached resource initialization
- `@st.cache_resource` for Pinecone index, Upstage client, Gemini LLM
- Session state management: `user_profile`, `user_concerns`, `chat_messages`, `search_results`
- Profile/concern registration forms with validation
- **IMPORTANT**: `get_gemini()` returns cached LLM instance (not pre-instantiated)

**pages/chatbot.py**: ReAct Agent chatbot with progress tracking (✅ Implemented)
- LangGraph `create_agent()` with 5 tools and middleware stack
- Progress streaming using `stream_mode="updates"` for intermediate steps
- Tool execution visualization with `st.status()` (running → complete states)
- Typing effect for final response (word-by-word animation)
- Dynamic system prompt based on user profile
- Chat history management with session state

**agents/react_chain.py**: Agent configuration (commented out, replaced by inline implementation)
- Originally designed for cached agent, now created per-session in chatbot.py
- See AGENTIC_SYSTEM_DESIGN.md for future multi-agent architecture

**tools/**: LangChain tools for agent (✅ Implemented)
- `sementic_search.py`: Pinecone semantic search with Upstage embeddings
- `ddgs_search.py`: DuckDuckGo web search for latest information
- `expert_search.py`: Domain expert advice generation
- `graph_search.py`: FalkorDB graph search (keyword-based relationship analysis)

**middleware/middleware.py**: LangGraph middleware (✅ Implemented)
- `dynamic_system_prompt`: Context-aware system prompt injection
- `SummarizationMiddleware`: Conversation summarization (4000 token threshold)
- `ToolCallLimitMiddleware`: Tool usage limits (websearch: 5 per thread, 3 per run)
- `ToolRetryMiddleware`: Automatic tool retry with exponential backoff
- `LoggingMiddleware`: Tool call and response logging

**schemas/**: Pydantic data models
- `UserProfile`: Career info, tech stack, work style with `to_context_string()`
- `UserConcern`: Category, title, description, urgency
- `CommonCompetencies`: Career-level-specific competencies and coaching approaches
- Tool-specific schemas: `ToolDdgsResult`, `ToolAnalyzeProfileOutput`

**utils/data_loader.py**: CSV processing utilities
- Functions: `load_csv_data()`, `extract_category()`, `combine_text_for_embedding()`, `prepare_documents_for_vectorstore()`
- Handles Korean text columns: `글 제목`, `출처`, `핵심 키워드`, `문제점 요약`, `글 내용 요약`

**utils/graph_db.py**: FalkorDB connection and schema management
- Functions: `get_falkordb_client()`, `get_graph()`, `create_graph_schema()`, `get_graph_stats()`
- Graph schema: Document, Keyword, Category nodes with HAS_KEYWORD, BELONGS_TO, CO_OCCURS_WITH relationships

**utils/graph_queries.py**: Graph database query functions
- Functions: `search_documents_by_keywords()`, `get_related_keywords()`, `get_keyword_network()`
- Cypher query utilities for keyword-based document search and relationship exploration

**prompts/**: System prompts (✅ Implemented)
- Career-level-specific prompts (junior, mid-level, senior)
- Common competencies and coaching approaches

### Key Patterns

**Streamlit Caching + LangGraph Integration** (⚠️ Critical Pattern):
```python
# ✅ Correct: Function returns cached instance
@st.cache_resource(show_spinner="🔄 Gemini 로드 중...", ttl=3600)
def get_gemini():
    """Cache: Gemini LLM Loader"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.7,
        max_tokens=3000,
        max_retries=3,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

# Usage in agent creation
llm = get_gemini()  # Call function to get cached instance
agent = create_agent(model=llm, tools=tools, ...)

# ❌ Wrong: Pre-instantiated at module level
get_gemini = _get_gemini()  # Causes bind_tools error
```

**Streaming with Progress Visualization** (✅ Implemented):
```python
# Use stream_mode="updates" for intermediate step tracking
for update in agent.stream(
    {"messages": st.session_state.chat_messages},
    {"configurable": {"thread_id": "1"}},
    context=profile,
    stream_mode="updates",  # Track each node execution
):
    for node_name, node_output in update.items():
        # Agent node: tool call decision
        if node_name == "agent":
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        # Display tool execution status
                        status_placeholder = st.status(
                            f"🔧 {tool_name} 실행 중...",
                            expanded=True,
                            state="running"
                        )

        # Tools node: tool execution result
        elif node_name == "tools":
            # Update tool status to complete
            status_placeholder.update(
                label=f"✅ {tool_name} 완료",
                state="complete"
            )

# Final response with typing effect
words = full_response.split()
for word in words:
    displayed_text += word + " "
    response_placeholder.markdown(displayed_text + "▌")
    time.sleep(0.02)
```

**Agent + Middleware Pattern**:
```python
# Dynamic system prompt with user context
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    profile: UserProfile = request.runtime.context
    return f"User profile: {profile.to_context_string()}"

# Agent creation with middleware stack
agent = create_agent(
    model=llm,
    tools=[ddgs_search, sementic_search, expert_search],
    middleware=[
        dynamic_system_prompt,  # First: inject context
        SummarizationMiddleware(model=llm, max_tokens_before_summary=4000),
        ToolCallLimitMiddleware(tool_name="websearch", thread_limit=5),
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0),
        LoggingMiddleware(),
    ],
    context_schema=UserProfile,
)
```

**Pinecone Namespace**:
- All vector operations use namespace `"20251029_crawling"`
- Must be specified in queries: `index.query(namespace="20251029_crawling", ...)`

**Graph Database Pattern**:
```python
# Get graph instance
from utils.graph_db import get_graph
graph = get_graph("mid_level_helper")

# Query documents by keywords
from utils.graph_queries import search_documents_by_keywords
docs = search_documents_by_keywords(["성장통", "재택근무"], limit=5)

# Get related keywords (co-occurrence based)
from utils.graph_queries import get_related_keywords
related = get_related_keywords("성장통", limit=10)

# Cypher query example
query = """
MATCH (d:Document)-[:HAS_KEYWORD]->(k:Keyword {name: $keyword})
RETURN d.title, d.category
LIMIT 10
"""
result = graph.query(query, {"keyword": "성장통"})
```

**Import Path Handling**:
- Scripts add project root to `sys.path` before imports
- Pattern: `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))`

**LangChain v1 + LangGraph Patterns**:
- Use `langchain.agents.create_agent()` for ReAct agents
- Use `langchain-google-genai.ChatGoogleGenerativeAI` (not legacy classes)
- Tools defined with `@tool` decorator
- Middleware for cross-cutting concerns (logging, retry, summarization)

**Korean Language Processing**:
- All UI text, data, and responses are in Korean
- Upstage Solar embeddings optimized for Korean semantic search
- Category extraction uses Korean and English patterns
- Dynamic prompts adapt to career level (주니어, 중니어, 시니어)

### Current vs Future Architecture

**Current Implementation** (✅ Working):
- ReAct Agent with 5 tools (sementic_search, graph_keyword_search, graph_related_keywords, ddgs_search, expert_search)
- FalkorDB graph database for keyword-based relationship analysis
- Progress streaming with `stream_mode="updates"` for intermediate steps
- Tool execution visualization with `st.status()` (running → complete)
- Typing effect for final response
- Middleware stack for logging, retry, summarization, and tool limits
- Dynamic system prompt based on user profile context
- Streamlit chatbot UI with real-time progress tracking

**Future Architecture** (AGENTIC_SYSTEM_DESIGN.md):
The project has detailed plans for expanding to a multi-agent system:
- **Supervisor Agent**: Query analysis, agent routing, result synthesis
- **RAG Agent**: Enhanced Pinecone semantic search
- **Web Search Agent**: Expanded DuckDuckGo integration
- **Profile Analyzer**: Deeper personalized analysis
- **Domain Expert**: Specialized advice (backend, frontend, career, management)

See [AGENTIC_SYSTEM_DESIGN.md](AGENTIC_SYSTEM_DESIGN.md) for complete multi-agent architecture design.

## Troubleshooting

### Common Issues

#### 1. `AttributeError: 'CachedFunc' object has no attribute 'bind_tools'`

**Cause**: Passing cached function wrapper to `create_agent()` instead of LLM instance.

**Solution**:
```python
# ✅ Correct
llm = get_gemini()  # Call function to get instance
agent = create_agent(model=llm, ...)

# ❌ Wrong
agent = create_agent(model=get_gemini, ...)  # Passing function reference
```

See [claudedocs/bind_tools_error_fix.md](claudedocs/bind_tools_error_fix.md) for detailed analysis.

#### 2. Streaming not working (showing complete response at once)

**Cause**: Using default `stream_mode="values"` instead of `stream_mode="messages"`.

**Solution**:
```python
# ✅ Correct: Token-by-token streaming
for chunk in agent.stream(..., stream_mode="messages"):
    msg, metadata = chunk
    # Extract tokens from AIMessageChunk

# ❌ Wrong: State updates only
for event in agent.stream(...):  # Default mode
    # Only gets completed messages
```

See [claudedocs/streaming_implementation_analysis.md](claudedocs/streaming_implementation_analysis.md) for implementation guide.

#### 3. Tool execution errors or timeouts

**Check**:
- API keys in `.env`: `UPSTAGE_API_KEY`, `PINECONE_API_KEY`, `GOOGLE_API_KEY`
- Pinecone index exists: `mid-level-helper` with namespace `20251029_crawling`
- Network connectivity for DuckDuckGo search
- Tool retry middleware is enabled (configured in [middleware/middleware.py](middleware/middleware.py))

**Debug**:
```python
# Enable verbose logging in middleware/middleware.py
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime):
        print(f"Tool calls: {len(state['messages'])} messages")
        return None
```

#### 4. Session state errors in Streamlit

**Cause**: Accessing session state before initialization.

**Solution**:
```python
# ✅ Initialize before use
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

# Then access
profile = st.session_state.user_profile
```

#### 5. Import errors for local modules

**Cause**: Python path not including project root.

**Solution** (for scripts):
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now can import project modules
from schemas import UserProfile
from main import get_pinecone
```

## Documentation

- **Project Overview**: [README.md](README.md)
- **Architecture Design**: [AGENTIC_SYSTEM_DESIGN.md](AGENTIC_SYSTEM_DESIGN.md)
- **Technical Docs**: [claudedocs/](claudedocs/) directory
  - `bind_tools_error_fix.md`: Streamlit caching + LangGraph integration
  - `streaming_implementation_analysis.md`: Real-time token streaming guide

## References

- [LangChain Agents](https://python.langchain.com/docs/how_to/custom_agent/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Streaming Guide](https://docs.langchain.com/oss/python/langchain/streaming)
- [Streamlit Caching](https://docs.streamlit.io/develop/concepts/architecture/caching)
