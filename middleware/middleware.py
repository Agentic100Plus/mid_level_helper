from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    dynamic_prompt,
)
from langgraph.runtime import Runtime

from main import get_gemini
from schemas import CommonCompetencies, UserProfile


# 동적 시스템 프롬프트: Context[UserProfile]
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """
    경력에 따라 프롬프트를 동적으로 구성합니다.
    """
    profile: UserProfile = request.runtime.context  # type: ignore

    common = CommonCompetencies()
    common_compet: list[str] = common.get_common_comps(profile.career_level)  # type: ignore
    common_appro: str = common.get_coaching_approach(profile.career_level)  # type: ignore

    base_prompt = f"""당신은 개발자 프로필 분석을 담당합니다.
## 역할 목표
- 주어진 정보를 토대로 공통된 정보를 조회해야합니다.
- 기술 역량 개발 로드맵 및 정신 건강 상담
- 실무 프로덕션 기준으로 구체적인 가이드 상담

## 개발자 프로필 정보:
{profile.to_context_string()}

## 공통 역량
{common_compet}

## 접근 방식
{common_appro}

## 가이드라인
1. 사용자가 물어본 언어로 최종 답변하세요. 한국어로 물어보면 한국어로 답변할 것.
2. 개발자의 현재 상황과 고민을 먼저 파악하세요.
3. 공감과 함께 다양한 사례를 들어 설명하세요.
4. 단순히 어떤 것을 하세요 보다, '왜?' 해야하는지에 더 설명을 추가하세요.

상담자의 질문에 공감하며 경청하고, 실질적인 도움이 되는 조언을 제공해주세요.
"""
    return base_prompt


# 로깅 미들웨어
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None


# 요약 처리 모델 - get_gemini()를 호출하여 실제 LLM 인스턴스 가져오기
# Note: @st.cache_resource가 적용된 함수이므로 호출해야 캐시된 인스턴스 반환
common_model = get_gemini()


# 웹서치 툴 리미터 미들웨어
websearch_limiter = ToolCallLimitMiddleware(
    tool_name="websearch",
    thread_limit=5,
    run_limit=5,
)

# 툴 리트라이 미들웨어
tool_retry_limiter = ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0, max_delay=60, jitter=True)

# 모델 폴벡: need vertax
# fallbacks = ModelFallbackMiddleware("gemini-2.5-flash-lite", "solar-pro2")

common_middlewares = [
    # fallbacks,
    SummarizationMiddleware(
        model=common_model,
        max_tokens_before_summary=4000,
    ),
    websearch_limiter,
    tool_retry_limiter,
    LoggingMiddleware(),
]
