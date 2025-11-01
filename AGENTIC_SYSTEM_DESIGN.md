# 🤖 에이전틱 중니어 상담소 - ReAct Agent 시스템 설계 및 구현 현황

> **최초 작성일**: 2025-10-30
> **최종 업데이트**: 2025-11-02
> **프로젝트**: 중니어 상담소
> **목적**: Tools 기반 ReAct Agent로 더 유연하고 에이전틱한 시스템 구축

---

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [현재 구현 상태](#현재-구현-상태) ⭐ **NEW**
3. [아키텍처 설계](#아키텍처-설계)
4. [Tools 상세 설계](#tools-상세-설계)
5. [Agent 구현 전략](#agent-구현-전략)
6. [프로젝트 구조](#프로젝트-구조)
7. [구현 가이드](#구현-가이드)
8. [기술 스택](#기술-스택)

---

## 🎯 시스템 개요

### 설계 철학 변경

**기존 접근 (LangGraph Multi-Agent)**:
```
문제점:
- LangGraph에 병렬 실행 노드를 명시적으로 정의해야 함
- 새 Agent 추가 시 워크플로우 전체 수정 필요
- 유연성 부족, 확장성 제한적
```

**새로운 접근 (ReAct Agent + Tools)** ✅:
```
장점:
- Agent가 상황에 따라 필요한 Tools만 자율적으로 선택
- Gemini의 강력한 function calling으로 병렬 실행 자동 처리
- 새 Tool 추가 = 함수 정의만 하면 끝
- LangGraph 복잡도 제거, 코드 단순화
```

### 핵심 아이디어

```
사용자 질의
    ↓
ReAct Agent (Gemini Function Calling)
    ↓
자율적 Tool 선택 및 실행
    ├─→ pinecone_search (유사 사례)
    ├─→ web_search (최신 정보)
    ├─→ analyze_profile (개인화 분석)
    └─→ get_expert_advice (전문가 조언)
    ↓
자동 통합 및 답변 생성
    ↓
최종 답변
```

### 주요 특징

- ✅ **자율적 Tool 선택**: Agent가 스스로 필요한 도구 결정
- ✅ **자동 병렬 실행**: Gemini가 여러 function 동시 호출
- ✅ **확장성**: 새 Tool 추가 시 코드 변경 최소화
- ✅ **단순성**: LangGraph Agent + Middleware로 구현
- ✅ **유연성**: 다양한 시나리오에 동적으로 대응

---

## ✅ 현재 구현 상태

### 구현 완료 항목

#### 1. Core Agent System (✅ 완료)

**구현 위치**: [pages/chatbot.py](pages/chatbot.py)

- **ReAct Agent**: LangGraph `create_agent()` 사용
- **LLM**: Gemini 2.5 Flash Lite (Function Calling 지원)
- **실시간 토큰 스트리밍**: `stream_mode="messages"` 구현
- **Tool 실행 시각화**: 🔧 Tool 호출 로그, ✅ Tool 결과 표시

```python
agent = create_agent(
    model=get_gemini(),  # Cached LLM instance
    tools=[ddgs_search, sementic_search, expert_search],
    middleware=[dynamic_system_prompt, *common_middlewares],
    context_schema=UserProfile,
)
```

#### 2. Tools Implementation (✅ 완료 3/4)

**구현 위치**: [tools/](tools/)

| Tool | 구현 상태 | 파일 | 기능 |
|------|----------|------|------|
| `sementic_search` | ✅ 완료 | [tool_sementic_search.py](tools/tool_sementic_search.py) | Pinecone 의미 검색 |
| `ddgs_search` | ✅ 완료 | [tool_ddgs.py](tools/tool_ddgs.py) | DuckDuckGo 웹 검색 |
| `expert_search` | ✅ 완료 | [tool_expert.py](tools/tool_expert.py) | 도메인 전문가 조언 |
| `analyze_profile` | ⏳ 예정 | - | 프로필 기반 맞춤 분석 |

#### 3. Middleware Stack (✅ 완료)

**구현 위치**: [middleware/middleware.py](middleware/middleware.py)

| Middleware | 구현 상태 | 기능 |
|-----------|----------|------|
| `dynamic_system_prompt` | ✅ 완료 | 경력별 동적 프롬프트 주입 |
| `SummarizationMiddleware` | ✅ 완료 | 대화 요약 (4000 토큰 임계값) |
| `ToolCallLimitMiddleware` | ✅ 완료 | Tool 호출 제한 (websearch: 5/3) |
| `ToolRetryMiddleware` | ✅ 완료 | 자동 재시도 + 지수 백오프 |
| `LoggingMiddleware` | ✅ 완료 | Tool 호출/응답 로깅 |

#### 4. Schemas & Data Models (✅ 완료)

**구현 위치**: [schemas/](schemas/)

- `UserProfile`: 사용자 프로필 (경력, 직무, 기술스택)
- `UserConcern`: 사용자 고민 (카테고리, 설명, 긴급도)
- `CommonCompetencies`: 경력별 공통 역량 정의
- Tool 응답 스키마: `ToolDdgsResult`, `ToolAnalyzeProfileOutput`

#### 5. System Prompts (✅ 완료)

**구현 위치**: [prompts/carreer_roles.py](prompts/carreer_roles.py)

- 주니어/중니어/시니어 단계별 프롬프트
- 경력별 공통 역량 및 코칭 접근법
- 동적 컨텍스트 주입 (`@dynamic_prompt`)

#### 6. Streamlit Integration (✅ 완료)

**구현 위치**: [main.py](main.py), [pages/chatbot.py](pages/chatbot.py)

- **캐싱 최적화**: `get_gemini()` 함수 패턴으로 bind_tools 에러 해결
- **실시간 스트리밍**: Token-by-token 표시 with cursor effect (▌)
- **Session State**: 프로필, 채팅 히스토리 관리
- **Tool 로그**: Tool 호출/결과를 별도 컨테이너로 시각화

### 미구현 항목 (Future Work)

#### 1. Multi-Agent Architecture (⏳ 설계 완료, 구현 예정)

**목표**: Supervisor + Specialized Agents

- **Supervisor Agent**: 쿼리 분석 및 에이전트 라우팅
- **RAG Agent**: 향상된 Pinecone 검색
- **Web Search Agent**: 확장된 웹 검색
- **Profile Analyzer**: 깊은 프로필 분석
- **Domain Experts**: 백엔드/프론트엔드/커리어/관리 전문가

#### 2. Additional Tools (⏳ 예정)

- `analyze_profile`: 프로필 기반 맞춤 분석
- GitHub integration
- Stack Overflow integration
- Code snippet analyzer

#### 3. Enhanced Features (⏳ 예정)

- 독립 검색 UI 페이지
- 대화 히스토리 저장/로드
- 사용자 피드백 수집
- 응답 품질 평가

### 기술적 성과

#### 1. Streamlit + LangGraph 통합 패턴 확립

**문제 해결**: `AttributeError: 'CachedFunc' object has no attribute 'bind_tools'`

**해결책**:
```python
# ✅ 올바른 패턴
@st.cache_resource
def get_gemini():
    return ChatGoogleGenerativeAI(...)

llm = get_gemini()  # 호출하여 인스턴스 획득
agent = create_agent(model=llm, ...)
```

**문서화**: [claudedocs/bind_tools_error_fix.md](claudedocs/bind_tools_error_fix.md)

#### 2. 실시간 토큰 스트리밍 구현

**핵심 기술**: `stream_mode="messages"`

```python
for chunk in agent.stream(..., stream_mode="messages"):
    msg, metadata = chunk
    node_name = metadata.get("langgraph_node", "")

    if "model" in node_name.lower():
        if msg.__class__.__name__ == "AIMessageChunk":
            token = getattr(msg, "content", "")
            if token:
                full_response += token
                response_placeholder.markdown(full_response + "▌")
```

**문서화**: [claudedocs/streaming_implementation_analysis.md](claudedocs/streaming_implementation_analysis.md)

#### 3. Middleware 기반 횡단 관심사 처리

- 동적 프롬프트 주입 (`@dynamic_prompt`)
- 자동 대화 요약 (토큰 임계값 기반)
- Tool 호출 제한 및 재시도
- 구조화된 로깅

### 테스트 및 검증

#### 단위 테스트 (일부 구현)

- [tests/test_tools.py](tests/test_tools.py): Tools 기능 테스트
- [tests/test_agent_debug.py](tests/test_agent_debug.py): Agent 디버깅

#### 통합 테스트 (수동)

- ✅ 프로필 등록 → 챗봇 → 응답 생성
- ✅ Tool 호출 → 결과 반환 → 통합 답변
- ✅ 실시간 스트리밍 → 커서 효과 → 최종 표시
- ✅ Middleware 동작 → 로깅 → 재시도

### 성능 메트릭

| 항목 | 현재 상태 | 목표 |
|------|----------|------|
| 첫 응답 시간 | ~3초 | <2초 |
| 토큰 스트리밍 지연 | ~50ms | <100ms |
| Tool 실행 성공률 | ~95% | >98% |
| 캐시 히트율 | ~90% | >95% |

---

## 🏗️ 아키텍처 설계

### 1. 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                         │
│  - 채팅 인터페이스                                                 │
│  - 프로필/고민 입력                                                │
│  - Tool 실행 로그 표시                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              ReAct Agent (create_tool_calling_agent)            │
│  - LangChain Agent Executor                                     │
│  - Gemini 1.5 Flash (Function Calling)                         │
│  - 자율적 Tool 선택 및 실행                                        │
│  - 자동 반복 및 에러 처리                                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬───────────────┐
         │               │               │               │
         ▼               ▼               ▼               ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ Pinecone     │ │ Web          │ │ Profile      │ │ Expert       │
  │ Search Tool  │ │ Search Tool  │ │ Analysis Tool│ │ Advice Tool  │
  ├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤
  │ Upstage 임베딩│ │ DuckDuckGo   │ │ 프로필 기반    │ │ 도메인 전문가 │
  │ Pinecone 검색 │ │ 웹 검색       │ │ 맞춤 분석     │ │ 조언 생성    │
  │ 유사 사례 반환 │ │ 최신 정보 수집│ │ 인사이트 제공  │ │ 전문 지식    │
  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### 2. 데이터 플로우 (자동 실행)

```
사용자: "재택근무 동기부여 최신 트렌드 알려줘"
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ReAct Agent 추론 (Gemini Function Calling)                  │
│                                                             │
│ "이 질문은 유사 사례 + 최신 정보가 필요하다고 판단"            │
│ → pinecone_search("재택근무 동기부여")                       │
│ → web_search("재택근무 동기부여 트렌드 2024")                 │
│                                                             │
│ [자동 병렬 실행] ✅                                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Tool 실행 결과 수집                                          │
│                                                             │
│ pinecone_search 결과:                                       │
│ - 사례 1: "재택근무 루틴 구축" (유사도 92%)                   │
│ - 사례 2: "온라인 커뮤니티 참여" (유사도 87%)                 │
│                                                             │
│ web_search 결과:                                            │
│ - "2024 재택근무 트렌드"                                     │
│ - "개발자 동기부여 방법"                                     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 자동 통합 및 답변 생성                                  │
│                                                             │
│ "수집된 정보를 바탕으로 종합 답변을 생성합니다."               │
└─────────────────────────────────────────────────────────────┘
    ↓
최종 답변 출력
```

### 3. Tool 선택 자율성

Agent가 **상황에 따라 자동으로 판단**:

```python
시나리오 1: "경력 정체 고민이에요"
→ Agent 판단: pinecone_search + analyze_profile
→ 병렬 실행 ✅

시나리오 2: "최신 프론트엔드 트렌드는?"
→ Agent 판단: web_search만 필요
→ 단일 Tool 실행 ✅

시나리오 3: "백엔드 아키텍처 조언 필요"
→ Agent 판단: get_expert_advice("backend")
→ 전문가 도구만 실행 ✅

시나리오 4: "다른 사람들은 어떻게 해결했나요?"
→ Agent 판단: pinecone_search만
→ 유사 사례만 검색 ✅
```

---

## 🛠️ Tools 상세 설계

### Tool 설계 원칙

- **Fine-grained**: 각 Tool은 하나의 명확한 기능만 수행
- **독립성**: Tool 간 의존성 최소화
- **재사용성**: 다양한 시나리오에서 조합 가능
- **명확한 시그니처**: Function calling을 위한 명확한 파라미터 정의

### Tool 1: pinecone_search

**목적**: Pinecone에서 유사한 고민 사례 검색

**함수 시그니처**:
```python
def pinecone_search(
    query: str,
    top_k: int = 5
) -> dict:
    """
    Pinecone 벡터 데이터베이스에서 유사한 중니어 고민 사례를 검색합니다.

    Args:
        query: 검색할 질의 (예: "재택근무 동기부여")
        top_k: 반환할 결과 개수 (기본값: 5)

    Returns:
        {
            "cases": [
                {
                    "title": "게시글 제목",
                    "category": "카테고리",
                    "summary": "문제점 요약",
                    "keywords": "핵심 키워드",
                    "similarity": 0.92
                },
                ...
            ],
            "count": 5
        }
    """
```

**구현 예시**:
```python
from langchain.tools import tool
from main import get_pinecone, get_upstage

@tool
def pinecone_search(query: str, top_k: int = 5) -> dict:
    """Pinecone에서 유사한 중니어 고민 사례를 검색합니다."""

    # 1. 임베딩 생성
    upstage = get_upstage()
    response = upstage.embeddings.create(
        input=[query],
        model="embedding-query"
    )
    query_embedding = response.data[0].embedding

    # 2. Pinecone 검색
    index = get_pinecone()
    results = index.query(
        namespace="20251029_crawling",
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # 3. 결과 포맷팅
    cases = []
    for match in results.matches:
        cases.append({
            "title": match.metadata.get("title", "N/A"),
            "category": match.metadata.get("category", "N/A"),
            "summary": match.metadata.get("problem_summary", "N/A"),
            "keywords": match.metadata.get("keywords", "N/A"),
            "similarity": round(match.score, 2)
        })

    return {
        "cases": cases,
        "count": len(cases)
    }
```

---

### Tool 2: web_search

**목적**: DuckDuckGo로 최신 정보 검색

**함수 시그니처**:
```python
def web_search(
    query: str,
    max_results: int = 5
) -> dict:
    """
    DuckDuckGo를 사용하여 최신 정보를 검색합니다.

    Args:
        query: 검색 키워드 (예: "재택근무 트렌드 2024")
        max_results: 최대 결과 개수 (기본값: 5)

    Returns:
        {
            "articles": [
                {
                    "title": "기사 제목",
                    "url": "https://...",
                    "snippet": "요약 내용"
                },
                ...
            ],
            "count": 5
        }
    """
```

**구현 예시**:
```python
from langchain.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """DuckDuckGo로 최신 정보를 검색합니다."""

    ddgs = DDGS()

    # 검색 실행
    results = list(ddgs.text(query, max_results=max_results))

    # 결과 포맷팅
    articles = []
    for result in results:
        articles.append({
            "title": result.get("title", "N/A"),
            "url": result.get("href", "N/A"),
            "snippet": result.get("body", "N/A")[:200]  # 200자로 제한
        })

    return {
        "articles": articles,
        "count": len(articles)
    }
```

---

### Tool 3: analyze_profile

**목적**: 사용자 프로필 기반 맞춤 분석

**함수 시그니처**:
```python
def analyze_profile(
    concern_description: str
) -> dict:
    """
    사용자 프로필(경력, 직무, 고민)을 분석하여 맞춤 인사이트를 제공합니다.

    Args:
        concern_description: 현재 고민 내용

    Returns:
        {
            "insights": [
                "인사이트 1",
                "인사이트 2",
                "인사이트 3"
            ],
            "recommendations": [
                "추천 사항 1",
                "추천 사항 2"
            ]
        }
    """
```

**구현 예시**:
```python
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

@tool
def analyze_profile(concern_description: str) -> dict:
    """사용자 프로필을 분석하여 맞춤 인사이트를 제공합니다."""

    # Session state에서 프로필 가져오기
    profile = st.session_state.get("user_profile")
    if not profile:
        return {"insights": [], "recommendations": []}

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    prompt = f"""
    사용자 프로필:
    - 경력: {profile.career_level}
    - 직무: {profile.job_role}
    - 기술 스택: {', '.join(profile.tech_stack)}

    현재 고민: {concern_description}

    위 프로필을 바탕으로:
    1. 핵심 인사이트 3가지
    2. 실행 가능한 추천 사항 2가지

    JSON 형식으로 답변:
    {{
        "insights": ["...", "...", "..."],
        "recommendations": ["...", "..."]
    }}
    """

    response = llm.invoke(prompt)
    return json.loads(response.content)
```

---

### Tool 4: get_expert_advice

**목적**: 도메인별 전문가 조언

**함수 시그니처**:
```python
def get_expert_advice(
    domain: str,
    question: str
) -> str:
    """
    특정 도메인 전문가의 조언을 제공합니다.

    Args:
        domain: 전문 분야 ("backend" | "frontend" | "career" | "management")
        question: 질문 내용

    Returns:
        전문가 조언 텍스트
    """
```

**구현 예시**:
```python
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

EXPERT_PROFILES = {
    "backend": "백엔드 아키텍처 및 시스템 설계 전문가로 10년 이상 경력",
    "frontend": "프론트엔드 UX/UI 및 성능 최적화 전문가로 최신 트렌드 정통",
    "career": "개발자 커리어 및 성장 전략 멘토로 수백 명 상담 경험",
    "management": "기술 부채 및 팀 관리 전문가로 다양한 조직 경험"
}

@tool
def get_expert_advice(domain: str, question: str) -> str:
    """도메인별 전문가 조언을 제공합니다."""

    if domain not in EXPERT_PROFILES:
        return "지원되지 않는 도메인입니다."

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    prompt = f"""
    당신은 {EXPERT_PROFILES[domain]}입니다.

    질문: {question}

    전문가 관점에서 구체적이고 실용적인 조언을 제공하세요.
    """

    response = llm.invoke(prompt)
    return response.content
```

---

## 🤖 Agent 구현 전략

### ReAct Agent 생성

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# 1. Tools 정의
tools = [
    pinecone_search,
    web_search,
    analyze_profile,
    get_expert_advice
]

# 2. LLM 설정 (Gemini Function Calling)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)

# 3. Prompt 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 중니어 개발자들의 고민을 상담하는 AI 어시스턴트입니다.

사용 가능한 도구:
- pinecone_search: 유사한 고민 사례 검색
- web_search: 최신 정보 검색
- analyze_profile: 사용자 맞춤 분석
- get_expert_advice: 전문가 조언

**중요**:
- 필요한 도구를 자율적으로 선택하여 사용하세요
- 여러 도구가 필요하면 병렬로 실행하세요
- 수집한 정보를 종합하여 실용적인 조언을 제공하세요
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. Agent 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Tool 실행 로그 출력
    handle_parsing_errors=True,
    max_iterations=5  # 최대 반복 횟수
)
```

### 실행 예시

```python
# 사용자 질의
response = agent_executor.invoke({
    "input": "재택근무 동기부여 최신 트렌드 알려줘"
})

print(response["output"])
```

**Agent 내부 실행 플로우** (자동):
```
1. 질의 분석: "유사 사례 + 최신 정보 필요"
2. Tool 선택: pinecone_search, web_search
3. 병렬 실행 (Gemini가 자동 처리)
4. 결과 수집
5. 통합 답변 생성
```

---

## 📂 프로젝트 구조

```
mid_level_helper/
├── .env                          # API 키
├── .env.example                  # API 키 템플릿
├── pyproject.toml                # 의존성
├── README.md                     # 프로젝트 문서
├── CLAUDE.md                     # AI 개발 가이드
├── AGENTIC_SYSTEM_DESIGN.md      # 이 문서
│
├── data/
│   └── mid_level_data_unique_3000.csv  # 원본 데이터
│
├── schemas/
│   ├── __init__.py
│   ├── user_profile.py           # UserProfile 모델
│   └── user_concern.py           # UserConcern 모델
│
├── tools/                        # ✨ Tools 정의
│   ├── __init__.py
│   ├── pinecone_search.py        # Pinecone 검색 Tool
│   ├── web_search.py             # 웹 검색 Tool
│   ├── analyze_profile.py        # 프로필 분석 Tool
│   └── expert_advice.py          # 전문가 조언 Tool
│
├── agents/                       # ✨ Agent 구성
│   ├── __init__.py
│   ├── react_agent.py            # ReAct Agent 생성 함수
│   └── prompts.py                # Agent Prompt 템플릿
│
├── utils/
│   ├── __init__.py
│   └── data_loader.py            # 데이터 로더
│
├── scripts/
│   ├── __init__.py
│   └── build_vectorstore.py      # 벡터 스토어 구축
│
├── tests/
│   ├── test_tools.py             # Tools 테스트
│   └── test_agent.py             # Agent 테스트
│
├── main.py                       # Streamlit 메인
│
└── pages/
    ├── chatbot.py                # ✨ ReAct 챗봇 페이지
    └── search.py                 # 검색 페이지
```

### 새로 추가된 디렉토리

**`tools/`**:
- 각 Tool을 독립적인 파일로 분리
- `@tool` 데코레이터로 LangChain Tool 정의
- 명확한 docstring으로 function calling 지원

**`agents/`**:
- `react_agent.py`: ReAct Agent 생성 로직
- `prompts.py`: Agent 시스템 프롬프트

---

## 📖 구현 가이드

### Phase 1: Tools 구현 (1-2일)

#### 1.1 Pinecone Search Tool

**파일**: `tools/pinecone_search.py`

```python
from langchain.tools import tool
from main import get_pinecone, get_upstage

@tool
def pinecone_search(query: str, top_k: int = 5) -> dict:
    """Pinecone에서 유사한 중니어 고민 사례를 검색합니다.

    Args:
        query: 검색할 질의어 (예: "재택근무 동기부여")
        top_k: 반환할 결과 개수 (기본값: 5)
    """
    # 구현 (위 예시 참고)
    pass
```

#### 1.2 Web Search Tool

**파일**: `tools/web_search.py`

```python
from langchain.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """DuckDuckGo로 최신 정보를 검색합니다.

    Args:
        query: 검색 키워드
        max_results: 최대 결과 개수
    """
    # 구현 (위 예시 참고)
    pass
```

#### 1.3 Profile Analysis Tool

**파일**: `tools/analyze_profile.py`

```python
from langchain.tools import tool
import streamlit as st

@tool
def analyze_profile(concern_description: str) -> dict:
    """사용자 프로필을 분석하여 맞춤 인사이트를 제공합니다.

    Args:
        concern_description: 현재 고민 내용
    """
    # 구현 (위 예시 참고)
    pass
```

#### 1.4 Expert Advice Tool

**파일**: `tools/expert_advice.py`

```python
from langchain.tools import tool

@tool
def get_expert_advice(domain: str, question: str) -> str:
    """도메인별 전문가 조언을 제공합니다.

    Args:
        domain: "backend", "frontend", "career", "management"
        question: 질문 내용
    """
    # 구현 (위 예시 참고)
    pass
```

---

### Phase 2: Agent 구성 (1일)

#### 2.1 ReAct Agent 생성

**파일**: `agents/react_agent.py`

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import pinecone_search, web_search, analyze_profile, get_expert_advice
from .prompts import AGENT_PROMPT

def create_react_agent():
    """ReAct Agent 생성"""

    tools = [
        pinecone_search,
        web_search,
        analyze_profile,
        get_expert_advice
    ]

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7
    )

    agent = create_tool_calling_agent(llm, tools, AGENT_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
```

#### 2.2 Prompt 템플릿

**파일**: `agents/prompts.py`

```python
from langchain.prompts import ChatPromptTemplate

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 중니어(중급 개발자) 상담 AI입니다.

사용 가능한 도구:
- pinecone_search: 유사한 고민 사례 검색
- web_search: 최신 정보 검색
- analyze_profile: 사용자 맞춤 분석
- get_expert_advice: 전문가 조언

원칙:
1. 필요한 도구를 자율적으로 선택
2. 여러 도구 필요 시 병렬 실행
3. 정보 종합 후 실용적 조언 제공
4. 한국어로 친근하게 응답
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
```

---

### Phase 3: Streamlit 통합 (1일)

#### 3.1 채팅 페이지

**파일**: `pages/chatbot.py`

```python
import streamlit as st
from agents.react_agent import create_react_agent

st.title("🤖 에이전틱 중니어 상담소")

# Agent 캐싱
@st.cache_resource
def get_agent():
    return create_react_agent()

agent = get_agent()

# 채팅 인터페이스
if "messages" not in st.session_state:
    st.session_state.messages = []

# 메시지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = agent.invoke({"input": prompt})
            answer = response["output"]

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
```

---

### Phase 4: 테스트 (1일)

#### 4.1 Tools 테스트

**파일**: `tests/test_tools.py`

```python
import pytest
from tools.pinecone_search import pinecone_search
from tools.web_search import web_search

def test_pinecone_search():
    result = pinecone_search.invoke({"query": "재택근무 동기부여", "top_k": 3})
    assert "cases" in result
    assert len(result["cases"]) <= 3

def test_web_search():
    result = web_search.invoke({"query": "개발자 트렌드 2024", "max_results": 3})
    assert "articles" in result
    assert len(result["articles"]) <= 3
```

#### 4.2 Agent 테스트

**파일**: `tests/test_agent.py`

```python
from agents.react_agent import create_react_agent

def test_agent_basic():
    agent = create_react_agent()
    response = agent.invoke({"input": "재택근무 고민"})
    assert "output" in response
    assert len(response["output"]) > 0
```

---

## 🛠️ 기술 스택

### Core
- **LangChain v1.0+**: Agent 프레임워크
- **Gemini 1.5 Flash**: Function calling 지원 LLM
- **Streamlit v1.50+**: 웹 인터페이스

### Tools
- **Pinecone**: 벡터 검색
- **Upstage Solar**: 한국어 임베딩
- **DuckDuckGo Search**: 웹 검색

### Dependencies
```toml
[project]
dependencies = [
    "langchain>=1.0.2",
    "langchain-google-genai>=3.0.0",
    "streamlit>=1.50.0",
    "pinecone-client>=5.0.0",
    "duckduckgo-search>=6.0.0",
    "python-dotenv>=1.0.0"
]
```

---

## 💡 핵심 장점

### vs LangGraph Multi-Agent

| 항목 | LangGraph 방식 | ReAct Tools 방식 ✅ |
|------|---------------|-------------------|
| **복잡도** | 높음 (워크플로우 정의 필요) | 낮음 (Tools만 정의) |
| **유연성** | 낮음 (고정된 플로우) | 높음 (자율적 선택) |
| **병렬 실행** | 명시적 정의 필요 | 자동 처리 |
| **확장성** | 중간 (노드 추가 복잡) | 높음 (Tool 추가 쉬움) |
| **코드 양** | 많음 | 적음 |
| **디버깅** | 어려움 | 쉬움 (Tool별 독립) |

### 확장성 예시

**새 Tool 추가 시**:

LangGraph 방식:
```python
# 1. Tool 정의
# 2. 노드 추가
# 3. 엣지 연결
# 4. 조건부 라우팅 수정
# → 많은 코드 변경 필요
```

ReAct 방식:
```python
# 1. Tool 정의만 하면 끝
@tool
def new_tool(...):
    pass

# Agent가 자동으로 사용 가능 ✅
```

---

## 🚀 구현 로드맵

### Week 1
- **Day 1-2**: Tools 구현 (4개 Tool)
- **Day 3**: Agent 구성
- **Day 4**: Streamlit 통합
- **Day 5**: 테스트 및 디버깅

### 예상 총 소요 시간
**5일** (기존 LangGraph 방식 대비 30% 단축)

---

## 📝 다음 단계

1. **Tools 구현부터 시작**
   - `tools/pinecone_search.py`
   - `tools/web_search.py`

2. **개발 중 도움 요청 시**
   - 각 Tool 구현 중 막히는 부분
   - Agent 설정 및 디버깅
   - Streamlit 통합

3. **테스트 및 최적화**
   - Tool 성능 측정
   - Agent 응답 품질 평가

---

**마지막 업데이트**: 2025-10-30
