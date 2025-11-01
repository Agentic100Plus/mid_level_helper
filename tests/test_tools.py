import os
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from schemas import UserProfile


class WebSearchSchemas(BaseModel):
    title: str = Field(description="제목")
    body: str = Field(description="본문 요약")
    href: str = Field(description="출처")


class WebSearchToolResponseSchemas(BaseModel):
    count: int = Field(description="결과 수")
    articles: list[WebSearchSchemas] = Field(description="결과 데이터")


# def test_ddgs_search():
#     """
#     ddgs 검색
#     """
#     results: list[dict[str, str]] = DDGS().text(
#         query="중니어 고민",
#         region="kr-kr",
#         max_results=10,
#         page=1,
#         backend="auto",
#     )
#     context = [WebSearchSchemas(**data) for data in results]

#     data = WebSearchToolResponseSchemas(count=len(context), articles=context)
#     assert data.count == 10
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from tools.web_search import ddgs_search


class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    base_prompt = "사용자 프로필 정보를 토대로 맞춤 인사이트를 제공합니다."
    profile = request.runtime.context
    career_level = profile.career_level
    #     profile_text = f"""
    # 경력: {profile.career_level} ({profile.years_of_experience}년차)
    # 직무: {profile.job_role}
    # 기술스택: {", ".join(profile.tech_stack)}
    # 회사규모: {profile.company_size or "미지정"}
    # 근무형태: {profile.work_style or "미지정"}
    #     """.strip()
    profile_text = profile.to_context_string()

    base_prompt += f"\n\n사용자프로필 정보: {profile_text}"

    if career_level == "주니어":
        base_prompt += "\n\n주니어 입장에서 어떤 인사이트를 줄지 고민해야해."
    elif career_level == "중니어":
        base_prompt += (
            "\n\n중니어 입장에서 조금 더 포괄적으로 접근해야해. 포괄적 접근을 위해 'ddgs_search' 도구를 반드시 사용해야해."
        )
        base_prompt += " 도구 'ddgs_search' 를 사용해서 나온 정보도 마지막에 반드시 추가해. "
    else:
        base_prompt += "\n\n프로필 정보를 기반으로만 접근해야해."

    #     base_prompt += """JSON 형식으로 답변:
    # {{
    #     "insights": ["...", "...", "..."],
    #     "recommendations": ["...", "..."]
    # }}"""
    return base_prompt


def test_analyze_proflile_prompt():
    """
    사용자 프로필 정보 기반 프롬프트 생성
    """
    # profile_text = UserProfile(
    #     name="김개발",
    #     career_level="주니어",
    #     years_of_experience=3,
    #     job_role="백엔드",
    #     tech_stack=["python", "django", "fastAPI", "AWS"],
    #     company_size="스타트업",
    #     work_style="재택",
    #     created_at=datetime.now(),
    # ).to_context_string()

    # query = "년차에 비해 할 줄 아는게 너무 없다고 생각해"

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,  # 0.1 → 0.7 변경 (Tool 호출 증가)
        max_tokens=1000,  # 500 → 1000 변경 (더 풍부한 답변)
        timeout=30,
        max_retries=3,
    )

    agent = create_agent(
        model=model,
        tools=[ddgs_search],
        middleware=[
            # SummarizationMiddleware(
            #     model=model,
            #     max_tokens_before_summary=3000,
            #     messages_to_keep=2,
            # ),
            # LoggingMiddleware(),
            user_role_prompt,
        ],
        context_schema=UserProfile,
    )
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "파이썬 백엔드 중니어로써 뭘해야할지 모르겠어?"},
            ]
        },
        context=UserProfile(
            name="김개발",
            career_level="중니어",
            years_of_experience=4,
            job_role="백엔드",
            tech_stack=["python", "django", "fastAPI", "AWS"],
            company_size="스타트업",
            work_style="재택",
            created_at=datetime.now(),
        ),
    )

    # Tool 호출 확인
    print("\n" + "=" * 80)
    print("📋 Result Messages:")
    print("=" * 80)

    for msg in result["messages"]:
        msg_type = msg.__class__.__name__
        print(f"\n🔹 Message Type: {msg_type}")

        # Tool 호출 확인
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("  ✅ Tool Calls Found!")
            for tool_call in msg.tool_calls:
                print(f"     🔧 Tool: {tool_call['name']}")
                print(f"     📝 Args: {tool_call['args']}")

        # Tool 결과 확인
        if msg_type == "ToolMessage":
            print(f"  📦 Tool Result: {msg.content[:200]}...")

        # AI 최종 응답
        if msg_type == "AIMessage" and hasattr(msg, "content") and msg.content:
            print(f"  💬 AI Response: {msg.content[:200]}...")

    assert 1 == 2
