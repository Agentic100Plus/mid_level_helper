"""
Streamlit 채팅 UI

스트리밍 기반 ReAct Agent 챗봇 인터페이스
- 실시간 토큰 단위 스트리밍 (LangChain stream_mode="messages")
- 중간 과정 상태 표시 (stream_mode="updates")
- 도구 호출 및 결과 시각화
- 채팅 히스토리 관리
"""

import streamlit as st
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from main import get_gemini
from middleware.middleware import common_middlewares, dynamic_system_prompt
from schemas import UserProfile
from tools import ddgs_search, expert_search, sementic_search
from tools.graph_search import graph_keyword_search, graph_related_keywords

# 툴 등록
tools = [
    sementic_search,
    graph_keyword_search,
    graph_related_keywords,
    ddgs_search,
    expert_search,
]
# ========================================
# Page Configuration
# ========================================

st.set_page_config(
    page_title="중니어 고민 상담 챗봇",
    page_icon="💬",
    layout="wide",
)

st.title("💬 중니어 고민 상담 챗봇")
st.caption("AI가 당신의 고민을 함께 해결합니다")

# ========================================
# Session State 초기화
# ========================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "user_profile" not in st.session_state:
    st.error("❌ 사용자 프로필이 없습니다. 먼저 메인 페이지에서 프로필을 등록하세요.")
    st.stop()

# ========================================
# Sidebar: 프로필 정보
# ========================================

with st.sidebar:
    st.header("👤 사용자 프로필")

    profile: UserProfile = st.session_state.user_profile

    st.write(f"**이름**: {profile.name}")
    st.write(f"**경력**: {profile.career_level.value} ({profile.years_of_experience}년차)")
    st.write(f"**직무**: {profile.job_role.value}")
    st.write(f"**기술스택**: {', '.join(profile.tech_stack)}")

    if profile.company_size:
        st.write(f"**회사규모**: {profile.company_size}")
    if profile.work_style:
        st.write(f"**근무형태**: {profile.work_style}")

    st.divider()

    # 채팅 히스토리 초기화 버튼
    if st.button("🗑️ 대화 내역 삭제", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

# ========================================
# 채팅 히스토리 표시
# ========================================

for msg in st.session_state.chat_messages:
    role = msg["role"]
    content = msg["content"]
    msg_type = msg.get("type", "normal")

    if role == "user":
        with st.chat_message("user"):
            st.write(content)

    elif role == "assistant":
        with st.chat_message("assistant"):
            st.write(content)

    elif role == "tool":
        # Tool 호출 로그
        tool_name = msg.get("tool_name", "Tool")
        tool_args = msg.get("tool_args", {})

        with st.status(f"✅ {tool_name} 완료", expanded=False, state="complete"):
            st.write(f"**도구**: {tool_name}")
            if tool_args:
                st.json(tool_args, expanded=False)

    elif role == "tool_result":
        # Tool 결과는 위 status에 포함되므로 별도 표시 안함
        pass

# ========================================
# 채팅 입력 처리
# ========================================

if prompt := st.chat_input("고민을 입력하세요..."):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(prompt)

    # 히스토리에 추가
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    # Agent 스트리밍 실행
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_container = st.container()

        full_response = ""
        tool_statuses = {}  # 도구별 상태 추적: {tool_name: status_placeholder}

        try:
            # Agent 가져오기 (Streamlit 캐싱)
            llm = get_gemini() if callable(get_gemini) else get_gemini

            agent = create_agent(
                model=llm,
                tools=tools,
                middleware=[
                    dynamic_system_prompt,  # type:ignore
                    *common_middlewares,
                ],
                checkpointer=InMemorySaver(),
                context_schema=UserProfile,
            )

            # stream_mode="updates"로 변경하여 중간 과정 추적
            # updates 모드: 각 노드의 실행 결과를 받음
            for update in agent.stream(
                {"messages": st.session_state.chat_messages},
                {"configurable": {"thread_id": "1"}},
                context=profile,
                stream_mode="updates",
            ):
                # update는 {node_name: node_output} 형태의 딕셔너리
                for node_name, node_output in update.items():

                    # Agent 노드: 도구 호출 결정
                    if node_name == "agent":
                        if "messages" in node_output:
                            messages = node_output["messages"]
                            for msg in messages:
                                msg_class = msg.__class__.__name__

                                # AIMessage with tool_calls: 도구 호출 시작
                                if msg_class == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        tool_name = tool_call.get("name", "Unknown")
                                        tool_args = tool_call.get("args", {})

                                        # 도구 호출 상태 표시
                                        with status_container:
                                            status_placeholder = st.status(
                                                f"🔧 {tool_name} 실행 중...",
                                                expanded=True,
                                                state="running"
                                            )
                                            with status_placeholder:
                                                st.write(f"**도구**: {tool_name}")
                                                if tool_args:
                                                    st.json(tool_args, expanded=False)

                                        # 상태 추적
                                        tool_statuses[tool_name] = status_placeholder

                                        # 히스토리 저장
                                        st.session_state.chat_messages.append({
                                            "role": "tool",
                                            "type": "call",
                                            "content": f"🔧 {tool_name} 호출",
                                            "tool_name": tool_name,
                                            "tool_args": tool_args
                                        })

                    # Tools 노드: 도구 실행 결과
                    elif node_name == "tools":
                        if "messages" in node_output:
                            messages = node_output["messages"]
                            for msg in messages:
                                msg_class = msg.__class__.__name__

                                # ToolMessage: 도구 실행 완료
                                if msg_class == "ToolMessage":
                                    tool_name = getattr(msg, "name", "Unknown")
                                    tool_result = getattr(msg, "content", "")

                                    # 도구 상태 업데이트
                                    if tool_name in tool_statuses:
                                        status_placeholder = tool_statuses[tool_name]
                                        status_placeholder.update(
                                            label=f"✅ {tool_name} 완료",
                                            state="complete",
                                            expanded=False
                                        )
                                        with status_placeholder:
                                            st.write(f"**도구**: {tool_name}")
                                            st.write(f"**결과**:")
                                            result_preview = str(tool_result)[:1000]
                                            if len(str(tool_result)) > 1000:
                                                result_preview += "..."
                                            st.text(result_preview)

                                    # 히스토리 저장
                                    st.session_state.chat_messages.append({
                                        "role": "tool_result",
                                        "content": tool_result,
                                        "tool_name": tool_name
                                    })

            # 최종 응답 추출 및 타이핑 효과
            import time

            final_state = agent.get_state({"configurable": {"thread_id": "1"}})
            if final_state and "messages" in final_state.values:
                messages = final_state.values["messages"]
                # 마지막 AIMessage 찾기
                for msg in reversed(messages):
                    if msg.__class__.__name__ == "AIMessage":
                        content = getattr(msg, "content", "")
                        # Tool calls가 없는 최종 응답만
                        has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
                        if content and not has_tool_calls:
                            full_response = content
                            break

            # 응답 표시 (타이핑 효과)
            if full_response:
                # 타이핑 효과: 단어 단위로 표시
                words = full_response.split()
                displayed_text = ""

                for i, word in enumerate(words):
                    displayed_text += word + " "
                    response_placeholder.markdown(displayed_text + "▌")
                    time.sleep(0.02)  # 단어당 20ms 지연

                # 최종 표시 (커서 제거)
                response_placeholder.markdown(full_response)
                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
            else:
                error_msg = "⚠️ 응답을 생성하지 못했습니다."
                response_placeholder.error(error_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

        except Exception as e:
            error_msg = f"❌ 오류 발생: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

            # 디버깅용 상세 에러
            with st.expander("🐛 상세 에러 정보"):
                st.exception(e)

# ========================================
# Footer
# ========================================

st.divider()
st.caption("💡 Tip: 구체적으로 고민을 설명하면 더 정확한 조언을 받을 수 있습니다.")
