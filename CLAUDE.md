# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A LangChain v1.0 chatbot that analyzes mid-level (중니어) developer concerns, built with Streamlit and Google GenAI. The project uses 3000+ developer reflections and retrospectives to provide consultation on career growth challenges.

**중니어 (Mid-level)**: Korean term for developers between junior and senior levels who face unique growth challenges.

## Technology Stack

- **LangChain v1.0+**: Core framework (v1.0.2+) - use LangChain v1 patterns, not legacy chains
- **Google GenAI**: LLM provider via `langchain-google-genai` (v3.0.0+)
- **Streamlit**: UI framework (v1.50.0+) with multi-page app structure
- **Pinecone**: Vector database for semantic search
- **Upstage Solar**: Korean-optimized embeddings (4096 dimensions)
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

### Running the Application
```bash
# Run Streamlit app
streamlit run main.py

# The app has multiple pages:
# - main.py: Profile/concern registration
# - pages/chatbot.py: AI consultation (WIP)
# - pages/search.py: Semantic search (WIP)
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

## Architecture

### Core Components

**main.py**: Entry point with cached resource initialization
- `@st.cache_resource` for Pinecone index, Upstage client, Gemini LLM
- Session state management: `user_profile`, `user_concerns`, `chat_history`, `search_results`
- Profile/concern registration forms with validation

**chains/**: RAG implementation (partially implemented)
- `retriever.py`: Pinecone semantic search with Upstage embeddings
- `chain.py`: RAG chain (placeholder)

**schemas/**: Pydantic data models
- `UserProfile`: Career info, tech stack, work style
- `UserConcern`: Category, title, description, urgency
- Both have helper methods: `to_search_query()`, `to_context_string()`

**utils/data_loader.py**: CSV processing utilities
- Functions: `load_csv_data()`, `extract_category()`, `combine_text_for_embedding()`, `prepare_documents_for_vectorstore()`
- Handles Korean text columns: `글 제목`, `출처`, `핵심 키워드`, `문제점 요약`, `글 내용 요약`

### Key Patterns

**Resource Caching**:
```python
@st.cache_resource(show_spinner="...", ttl=3600)
def get_pinecone():
    # Expensive initialization cached for 1 hour
    # Used in main.py and chains/retriever.py
```

**Pinecone Namespace**:
- All vector operations use namespace `"20251029_crawling"`
- Must be specified in queries: `index.query(namespace="20251029_crawling", ...)`

**Import Path Handling**:
- Scripts add project root to `sys.path` before imports
- Pattern: `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))`

**LangChain v1 Patterns**:
- Use LCEL (LangChain Expression Language): `prompt | llm | parser`
- Use `langchain-google-genai.ChatGoogleGenerativeAI`, not legacy classes
- Prefer `RunnableSequence` and `RunnableParallel` over deprecated Chain classes

**Korean Language Processing**:
- All UI text, data, and responses are in Korean
- Upstage Solar embeddings optimized for Korean semantic search
- Category extraction uses Korean and English patterns

### Future Architecture (AGENTIC_SYSTEM_DESIGN.md)

The project has detailed plans for multi-agent system using LangGraph:
- **Supervisor Agent**: Query analysis, agent routing, result synthesis
- **RAG Agent**: Pinecone semantic search
- **Web Search Agent**: DuckDuckGo for latest information
- **Profile Analyzer**: Personalized analysis based on user profile
- **Domain Expert**: Specialized advice (backend, frontend, career, management)

See [AGENTIC_SYSTEM_DESIGN.md](AGENTIC_SYSTEM_DESIGN.md) for complete multi-agent architecture design.
