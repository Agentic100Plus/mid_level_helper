# 🤖 에이전틱 중니어 상담소 - Multi-Agent 시스템 설계

> **작성일**: 2025-10-29
> **프로젝트**: 중니어 상담소
> **목적**: 단순 RAG 기반 챗봇을 Multi-Agent 시스템으로 확장

---

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [아키텍처](#아키텍처)
3. [Agent 상세 설계](#agent-상세-설계)
4. [LangGraph 워크플로우](#langgraph-워크플로우)
5. [Streamlit 통합](#streamlit-통합)
6. [구현 로드맵](#구현-로드맵)
7. [기술 스택](#기술-스택)

---

## 🎯 시스템 개요

### 핵심 아이디어

**기존 (Simple RAG)**:
```
사용자 질의 → RAG 검색 → LLM 답변
```

**제안 (Agentic Multi-Agent)**:
```
사용자 질의
    ↓
Supervisor Agent (분석 & 계획)
    ↓
├─→ RAG Agent (유사 사례)
├─→ Web Search Agent (최신 정보)
├─→ Profile Analyzer (개인화 분석)
└─→ Domain Expert (분야별 전문가)
    ↓
Supervisor Agent (통합 & 조언)
    ↓
최종 답변
```

### 주요 특징

- ✅ **Multi-Agent**: 여러 전문 에이전트가 협력
- ✅ **Dynamic Routing**: 쿼리에 따라 필요한 에이전트만 실행
- ✅ **Parallel Execution**: 독립적인 에이전트는 병렬 실행
- ✅ **Web Search**: DuckDuckGo로 최신 정보 수집
- ✅ **Personalization**: 사용자 프로필 기반 맞춤 분석
- ✅ **Extensible**: 새로운 에이전트 추가 용이

---

## 🏗️ 아키텍처

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Supervisor Agent (Coordinator)                 │
│  - 사용자 쿼리 분석                                             │
│  - 서브태스크 분해.                                             │
│  - 에이전트 선택 & 실행 순서 결정.                                 │
│  - 결과 통합 & 최종 답변 생성                                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┐
    │              │              │              │
    ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   RAG    │  │   Web    │  │ Profile  │  │ Domain   │
│  Agent   │  │  Search  │  │ Analyzer │  │ Expert   │
│          │  │  Agent   │  │  Agent   │  │  Agent   │
├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤
│Pinecone  │  │DuckDuck  │  │User      │  │Gemini    │
│검색       │  │Go API    │  │Profile   │  │Specialist│
│유사 사례   │  │최신 정보   │   │맞춤 분석   │  │전문 조언   │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

### 데이터 플로우

```
사용자 입력
    ↓
Supervisor: 쿼리 분석
    - "재택근무 동기부여 + 최신 트렌드"
    - 필요 에이전트: RAG, Web Search, Profile
    ↓
병렬 실행
├─→ RAG Agent: Pinecone 유사 사례 검색
├─→ Web Search: DuckDuckGo 최신 정보
└─→ Profile Analyzer: 사용자 맞춤 분석
    ↓
Supervisor: 결과 통합
    - 유사 사례 3개
    - 최신 트렌드 5개
    - 개인화 인사이트
    ↓
최종 답변 생성
```

---

## 🤖 Agent 상세 설계

### 1. Supervisor Agent (코디네이터)

**역할**: 전체 프로세스 조율 및 통합

**핵심 기능**:
- 쿼리 분석 및 필요 에이전트 결정
- 실행 계획 수립 (병렬/순차)
- 서브 에이전트 결과 통합
- 최종 답변 생성

**구현 예시**:
```python
class SupervisorAgent:
    def analyze_query(self, query: str, profile: UserProfile) -> Plan:
        """쿼리 분석 및 실행 계획 생성"""
        prompt = f"""
        사용자 질의: {query}
        사용자 프로필: {profile.to_context_string()}

        어떤 정보가 필요한가?

        선택 가능한 에이전트:
        1. rag_agent - 유사 사례
        2. web_search_agent - 최신 정보
        3. profile_analyzer - 맞춤 분석
        4. domain_expert - 전문 조언

        JSON 형식으로 답변:
        {{
            "agents": ["rag_agent", "web_search_agent"],
            "execution": "parallel",
            "reasoning": "이유"
        }}
        """

        plan = llm.invoke(prompt)
        return parse_plan(plan)

    def execute_plan(self, plan: Plan, query: str) -> AgentResults:
        """계획에 따라 서브 에이전트 실행"""
        if plan.execution == "parallel":
            # 병렬 실행
            with ThreadPoolExecutor() as executor:
                futures = {
                    name: executor.submit(self.agents[name].run, query)
                    for name in plan.agents
                }
                return {name: f.result() for name, f in futures.items()}
        else:
            # 순차 실행
            return {
                name: self.agents[name].run(query)
                for name in plan.agents
            }

    def synthesize(self, query: str, results: AgentResults) -> str:
        """결과 통합 및 최종 답변"""
        context = self._format_results(results)

        prompt = f"""
        사용자 질의: {query}
        수집된 정보: {context}

        위 정보를 종합하여 실용적인 조언을 제공하세요.
        """

        return llm.invoke(prompt)
```

---

### 2. RAG Agent (유사 사례 검색)

**역할**: Pinecone에서 유사한 고민 사례 검색

**핵심 기능**:
- 쿼리 강화 (프로필 정보 추가)
- Upstage 임베딩 생성
- Pinecone 유사도 검색
- 결과 포맷팅

**구현 예시**:
```python
class RAGAgent:
    def __init__(self):
        self.index = get_pinecone_index()
        self.embeddings = get_upstage_client()

    def run(self, query: str, profile: UserProfile = None, top_k: int = 5) -> dict:
        # 1. 쿼리 강화
        if profile:
            enhanced_query = f"""
            {query}
            사용자: {profile.career_level}, {profile.job_role}
            """
        else:
            enhanced_query = query

        # 2. 임베딩 생성
        query_embedding = self.create_embedding(enhanced_query)

        # 3. Pinecone 검색
        results = self.index.query(
            namespace="20251029_crawling",
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # 4. 결과 반환
        return {
            "agent": "rag_agent",
            "task": "유사 사례 검색",
            "results": [
                {
                    "title": match.metadata['title'],
                    "category": match.metadata['category'],
                    "summary": match.metadata['problem_summary'],
                    "similarity": match.score
                }
                for match in results.matches
            ],
            "summary": f"{len(results.matches)}개 사례 발견"
        }
```

---

### 3. Web Search Agent (최신 정보)

**역할**: DuckDuckGo로 최신 정보 검색

**핵심 기능**:
- 검색어 최적화
- DuckDuckGo 검색 실행
- 결과 요약 (LLM)
- 관련성 필터링

**구현 예시**:
```python
class WebSearchAgent:
    def __init__(self):
        from duckduckgo_search import DDGS
        self.ddgs = DDGS()

    def run(self, query: str, max_results: int = 5) -> dict:
        # 1. 검색어 최적화
        search_query = self._optimize_query(query)

        # 2. DuckDuckGo 검색
        results = list(self.ddgs.text(
            search_query,
            max_results=max_results
        ))

        # 3. 결과 요약
        summaries = []
        for result in results:
            summary = self._summarize_content(result['body'])
            summaries.append({
                "title": result['title'],
                "url": result['href'],
                "summary": summary
            })

        return {
            "agent": "web_search_agent",
            "task": "최신 정보 검색",
            "results": summaries,
            "summary": f"{len(summaries)}개 자료 발견"
        }

    def _optimize_query(self, query: str) -> str:
        """검색어 최적화"""
        prompt = f"""
        다음 질의를 웹 검색에 적합한 키워드로:
        {query}

        한국어 + 영어 조합으로 만들어주세요.
        """
        return llm.invoke(prompt)
```

---

### 4. Profile Analyzer Agent (개인화 분석)

**역할**: 사용자 프로필 기반 맞춤 분석

**핵심 기능**:
- 프로필 정보 분석
- 경력 단계별 고려사항 도출
- 직무 특성 반영
- 맞춤 조언 포인트 제공

**구현 예시**:
```python
class ProfileAnalyzerAgent:
    def run(self, query: str, profile: UserProfile, concern: UserConcern) -> dict:
        prompt = f"""
        사용자 프로필:
        {profile.to_context_string()}

        현재 고민:
        [{concern.category}] {concern.title}
        {concern.description}

        사용자 질의: {query}

        위 정보를 바탕으로:
        1. 가장 중요한 고려사항 3가지
        2. {profile.career_level}에 맞는 접근법
        3. {profile.job_role} 특성 반영한 조언

        JSON 형식으로 답변.
        """

        analysis = llm.invoke(prompt)

        return {
            "agent": "profile_analyzer",
            "task": "개인화 분석",
            "insights": parse_analysis(analysis),
            "summary": "맞춤 분석 완료"
        }
```

---

### 5. Domain Expert Agent (전문 분야)

**역할**: 분야별 전문가 조언

**전문 분야**:
- `backend`: 백엔드 아키텍처 및 시스템 설계
- `frontend`: 프론트엔드 UX/UI 및 성능 최적화
- `career`: 개발자 커리어 및 성장 전략
- `management`: 기술 부채 및 팀 관리

**구현 예시**:
```python
class DomainExpertAgent:
    EXPERT_PROFILES = {
        "backend": "백엔드 아키텍처 및 시스템 설계 전문가",
        "frontend": "프론트엔드 UX/UI 및 성능 최적화 전문가",
        "career": "개발자 커리어 및 성장 전략 멘토",
        "management": "기술 부채 및 팀 관리 전문가",
    }

    def run(self, query: str, domain: str, context: dict) -> dict:
        prompt = f"""
        당신은 {self.EXPERT_PROFILES[domain]}입니다.

        사용자 질의: {query}
        참고 정보: {format_context(context)}

        전문가 관점에서 구체적 조언을 제공하세요.
        """

        advice = llm.invoke(prompt)

        return {
            "agent": "domain_expert",
            "domain": domain,
            "advice": advice,
            "summary": f"{domain} 전문가 조언"
        }
```

---

## 🔄 LangGraph 워크플로우

### StateGraph 구조

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    """에이전트 상태"""
    query: str
    profile: UserProfile
    concern: UserConcern

    # 실행 계획
    plan: dict

    # 서브 에이전트 결과
    rag_results: dict
    web_results: dict
    profile_insights: dict
    expert_advice: dict

    # 최종 결과
    final_answer: str
    agent_logs: Annotated[list, operator.add]


def create_agent_workflow():
    """Multi-Agent 워크플로우 생성"""

    workflow = StateGraph(AgentState)

    # 노드 정의
    workflow.add_node("supervisor_plan", supervisor_plan_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("profile_analyzer", profile_analyzer_node)
    workflow.add_node("supervisor_synthesize", supervisor_synthesize_node)

    # 엣지 정의
    workflow.set_entry_point("supervisor_plan")

    # 조건부 병렬 실행
    workflow.add_conditional_edges(
        "supervisor_plan",
        lambda state: "parallel_execution",
        {
            "parallel_execution": ["rag_agent", "web_search", "profile_analyzer"]
        }
    )

    # 모든 에이전트 → Supervisor 통합
    workflow.add_edge("rag_agent", "supervisor_synthesize")
    workflow.add_edge("web_search", "supervisor_synthesize")
    workflow.add_edge("profile_analyzer", "supervisor_synthesize")

    # 종료
    workflow.add_edge("supervisor_synthesize", END)

    return workflow.compile()
```

### 워크플로우 다이어그램

```
START
  ↓
[Supervisor: Plan]
  ├─ 쿼리 분석
  ├─ 에이전트 선택
  └─ 실행 계획 수립
  ↓
[Parallel Execution]
  ├─→ [RAG Agent]
  │    └─ Pinecone 검색
  ├─→ [Web Search]
  │    └─ DuckDuckGo 검색
  └─→ [Profile Analyzer]
       └─ 맞춤 분석
  ↓
[Supervisor: Synthesize]
  ├─ 결과 수집
  ├─ 컨텍스트 통합
  └─ 최종 답변 생성
  ↓
END
```

---

## 🎨 Streamlit 통합

### 에이전틱 챗봇 UI

```python
import streamlit as st
from agent_system import create_agent_workflow

st.title("🤖 에이전틱 중니어 상담소")

# 워크플로우 캐싱
@st.cache_resource
def get_agent_workflow():
    return create_agent_workflow()

workflow = get_agent_workflow()

# 채팅 인터페이스
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 (에이전트 실행)
    with st.chat_message("assistant"):
        # 진행 상황 표시
        status_container = st.empty()
        logs_container = st.expander("🔍 에이전트 활동 로그", expanded=True)

        # 초기 상태
        initial_state = {
            "query": prompt,
            "profile": st.session_state.user_profile,
            "concern": st.session_state.user_concerns[0],
            "agent_logs": []
        }

        # 워크플로우 스트리밍 실행
        for state in workflow.stream(initial_state):
            # 로그 업데이트
            if "agent_logs" in state:
                with logs_container:
                    for log in state["agent_logs"]:
                        st.caption(log)

            # 진행 상황
            if "plan" in state:
                status_container.info(
                    f"📋 실행 계획: {', '.join(state['plan']['agents'])}"
                )

        # 최종 답변
        final_state = state
        st.markdown(final_state["final_answer"])

        # 참고 자료
        with st.expander("📚 참고 자료"):
            if final_state.get("rag_results"):
                st.markdown("### 유사 사례")
                for case in final_state["rag_results"]["results"][:3]:
                    st.markdown(f"- **{case['title']}** ({case['similarity']:.1%})")

            if final_state.get("web_results"):
                st.markdown("### 최신 정보")
                for article in final_state["web_results"]["results"][:3]:
                    st.markdown(f"- [{article['title']}]({article['url']})")
```

### UI 구조

```
┌──────────────────────────────────────────┐
│  🤖 에이전틱 중니어 상담소                 │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  [대화 내역]                              │
│                                          │
│  👤 재택근무 동기부여 + 최신 트렌드       │
│                                          │
│  🤖 [에이전트 활동 로그 ▼]                │
│     📋 실행 계획: rag, web_search, profile│
│     🔍 RAG: 3개 유사 사례 발견            │
│     🌐 Web: 5개 최신 자료 발견            │
│     👤 Profile: 맞춤 분석 완료            │
│     ✅ 최종 답변 생성 완료                │
│                                          │
│  김개발님의 상황을 고려하여...            │
│                                          │
│  [참고 자료 ▼]                           │
│  - 유사 사례 3개                         │
│  - 최신 정보 5개                         │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  💬 [메시지 입력...]           [전송]     │
└──────────────────────────────────────────┘
```

---

## 📊 실행 예시

### 사용자 쿼리
```
"재택근무하면서 동기부여가 떨어져요. 최신 트렌드도 알고 싶어요."
```

### 시스템 실행 플로우

```
[1] Supervisor Agent 분석
📋 실행 계획:
  - rag_agent (유사 사례 검색)
  - web_search_agent (최신 재택근무 트렌드)
  - profile_analyzer (개인 맞춤 분석)
  실행 방식: 병렬

[2] 병렬 실행
┌─────────────────┬─────────────────┬─────────────────┐
│  RAG Agent      │  Web Search     │  Profile        │
├─────────────────┼─────────────────┼─────────────────┤
│ Pinecone 검색   │ DuckDuckGo 검색 │ 프로필 분석     │
│ - 사례 #1       │ - 2024 재택     │ - 3년차 중니어  │
│ - 사례 #2       │   트렌드        │ - 백엔드 개발   │
│ - 사례 #3       │ - 동기부여      │ - 맞춤 조언     │
│                 │   방법          │   포인트        │
└─────────────────┴─────────────────┴─────────────────┘

[3] Supervisor 통합
✅ 3개 에이전트 결과 수집
✅ 컨텍스트 통합
✅ 최종 답변 생성

[4] 최종 답변
"김개발님(3년차 중니어, 백엔드)의 상황을 고려하여...

[유사 사례 참고]
비슷한 상황의 개발자들은 다음과 같은 방법으로 극복했습니다:
- 사례 1: 재택근무 루틴 구축 (유사도 92%)
- 사례 2: 온라인 커뮤니티 참여 (유사도 87%)

[최신 트렌드]
2024년 재택근무 동기부여 방법:
- Virtual Office 활용
- Pomodoro + 원격 협업
...

[맞춤 조언]
백엔드 개발자로서 고려할 점:
1. ...
2. ...
"
```

---

## 🚀 구현 로드맵

### Phase 1: 기본 Multi-Agent (1-2일)
- [ ] Supervisor Agent 구현
- [ ] RAG Agent 구현
- [ ] 단순 순차 실행
- [ ] Streamlit 기본 통합

### Phase 2: 웹 검색 추가 (1일)
- [ ] DuckDuckGo Agent 구현
- [ ] 병렬 실행 구현
- [ ] 에러 핸들링

### Phase 3: 고도화 (2-3일)
- [ ] Profile Analyzer 추가
- [ ] Domain Expert 추가
- [ ] 조건부 라우팅 구현
- [ ] LangGraph 통합

### Phase 4: 최적화 (1-2일)
- [ ] 성능 튜닝
- [ ] 캐싱 전략
- [ ] 로깅 강화
- [ ] UI/UX 개선

**총 예상 기간**: 5-8일

---

## 🛠️ 기술 스택

### Core
- **LangChain v1.0+**: RAG 체인 구축
- **LangGraph v1.0+**: Multi-Agent 워크플로우
- **Streamlit v1.50+**: 웹 인터페이스

### LLM & Embeddings
- **Google Gemini 1.5 Flash**: 메인 LLM
- **Upstage Solar Embeddings**: 한국어 임베딩 (4096차원)

### Vector Store & Search
- **Pinecone**: 벡터 데이터베이스
- **DuckDuckGo Search**: 웹 검색 API

### Additional
- **python-dotenv**: 환경 변수 관리
- **duckduckgo-search**: 웹 검색 라이브러리
- **pandas**: 데이터 처리

---

## 💡 핵심 이점

### vs 단순 RAG

| 항목 | 단순 RAG | Multi-Agent |
|------|----------|-------------|
| **정보 소스** | Pinecone만 | Pinecone + Web + 분석 |
| **개인화** | 프롬프트만 | 전용 Analyzer Agent |
| **최신성** | 없음 | 웹 검색으로 최신 정보 |
| **확장성** | 낮음 | 높음 (에이전트 추가 쉬움) |
| **복잡도** | 낮음 | 중간 |
| **응답 품질** | 보통 | 높음 (다각도 분석) |

### 확장 가능성

새로운 에이전트 추가 예시:
- **Code Review Agent**: 코드 리뷰 관련 조언
- **Interview Prep Agent**: 면접 준비 도움
- **Salary Negotiation Agent**: 연봉 협상 가이드
- **Learning Path Agent**: 학습 로드맵 제시

---

## 📝 참고 문서

- [LangGraph 공식 문서](https://python.langchain.com/docs/langgraph)
- [Multi-Agent Systems](https://python.langchain.com/docs/use_cases/agent_simulations/)
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)
- [Pinecone Vector Database](https://docs.pinecone.io/)

---

## 🔗 관련 파일

- `models.py` - 데이터 모델
- `agents/supervisor.py` - Supervisor Agent
- `agents/rag_agent.py` - RAG Agent
- `agents/web_search.py` - Web Search Agent
- `agents/profile_analyzer.py` - Profile Analyzer
- `workflow.py` - LangGraph 워크플로우
- `pages/agentic_chatbot.py` - Streamlit UI

---

**마지막 업데이트**: 2025-10-29
