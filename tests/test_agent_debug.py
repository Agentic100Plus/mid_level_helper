"""Agent Tool 사용 디버깅 스크립트"""
import os
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
import langchain

# ✨ LangChain 디버그 모드 활성화
langchain.debug = True

from schemas import UserProfile
from tools.web_search import ddgs_search


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    base_prompt = "사용자 프로필 정보를 토대로 맞춤 인사이트를 제공합니다."
    profile = request.runtime.context
    career_level = profile.career_level
    profile_text = f"""
경력: {profile.career_level} ({profile.years_of_experience}년차)
직무: {profile.job_role}
기술스택: {", ".join(profile.tech_stack)}
회사규모: {profile.company_size or "미지정"}
근무형태: {profile.work_style or "미지정"}
    """.strip()

    base_prompt += f"\n\n사용자프로필 정보: {profile_text}"

    if career_level == "중니어":
        base_prompt += (
            "\n\n중니어 입장에서 조금 더 포괄적으로 접근해야해. 포괄적 접근을 위해 'ddgs_search' 도구를 반드시 사용해야해."
        )
        base_prompt += " 도구 'ddgs_search' 를 사용해서 나온 정보도 마지막에 반드시 추가해. "

    return base_prompt


def main():
    print("=" * 80)
    print("Agent Tool 사용 테스트")
    print("=" * 80)

    # Model 설정
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,  # 0.1에서 0.7로 증가 (더 창의적)
        max_tokens=1000,  # 500에서 1000으로 증가
        timeout=30,
        max_retries=3,
    )

    # Agent 생성
    agent = create_agent(
        model=model,
        tools=[ddgs_search],
        middleware=[user_role_prompt],
        context_schema=UserProfile,
    )

    # 실행
    print("\n질문: 파이썬 백엔드 중니어로써 뭘해야할지 모르겠어?")
    print("\n" + "-" * 80)

    # Stream 방식으로 이벤트 확인
    tool_calls_found = []
    tool_results_found = []

    for event in agent.stream(
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
    ):
        # 이벤트 타입 출력
        for node_name, node_data in event.items():
            print(f"\n📍 Node: {node_name}")

            if "messages" in node_data:
                for msg in node_data["messages"]:
                    msg_type = msg.__class__.__name__
                    print(f"  └─ Message Type: {msg_type}")

                    # Tool 호출 감지
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            print(f"     🔧 Tool Call: {tool_call['name']}")
                            print(f"        Args: {tool_call['args']}")
                            tool_calls_found.append(tool_call)

                    # Tool 결과 감지
                    if msg_type == "ToolMessage":
                        print(f"     ✅ Tool Result: {msg.content[:100]}...")
                        tool_results_found.append(msg.content)

                    # AI 응답
                    if msg_type == "AIMessage" and hasattr(msg, "content"):
                        print(f"     💬 AI: {msg.content[:100]}...")

    # 마지막 결과
    result = {"messages": [], "tool_calls": tool_calls_found, "tool_results": tool_results_found}

    print("\n" + "=" * 80)
    print("Tool 사용 요약:")
    print("=" * 80)

    # Tool 호출 출력
    if tool_calls_found:
        print(f"\n✅ Tool이 {len(tool_calls_found)}번 호출되었습니다!")
        for i, tool_call in enumerate(tool_calls_found, 1):
            print(f"\n[Tool Call #{i}]")
            print(f"  Tool: {tool_call['name']}")
            print(f"  Args: {tool_call['args']}")
    else:
        print("\n❌ Tool이 사용되지 않았습니다.")
        print("\n가능한 원인:")
        print("1. System prompt가 명확하지 않음")
        print("2. Tool description이 부족함")
        print("3. Model이 Tool 필요성을 판단하지 못함")

    # Tool 결과 출력
    if tool_results_found:
        print(f"\n📦 Tool 결과: {len(tool_results_found)}개")
        for i, result_content in enumerate(tool_results_found, 1):
            print(f"\n[Result #{i}]")
            print(result_content[:300] + ("..." if len(result_content) > 300 else ""))


if __name__ == "__main__":
    main()
