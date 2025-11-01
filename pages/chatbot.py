"""
Streamlit 채팅 UI

스트리밍 기반 ReAct Agent 챗봇 인터페이스
- 실시간 토큰 단위 스트리밍 (LangChain stream_mode="messages")
- 중간 과정 로그 표시 (Tool calls, Results)
- 채팅 히스토리 관리
"""

import streamlit as st
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from main import get_gemini
from middleware.middleware import common_middlewares, dynamic_system_prompt
from schemas import UserProfile
from tools import ddgs_search, expert_search, sementic_search

# 툴 등록
tools = [
    ddgs_search,
    sementic_search,
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
        with st.chat_message("assistant", avatar="🔧"):
            st.caption(content)

    elif role == "tool_result":
        # Tool 결과 로그
        with st.chat_message("assistant", avatar="✅"):
            with st.expander(f"📦 {msg.get('tool_name', 'Tool')} 결과", expanded=False):
                st.text(content[:500] + "..." if len(content) > 500 else content)

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
        tool_log_container = st.container()

        full_response = ""

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

            chat_messages = [
                {"role": "user" if msg["role"] == "user" else "ai", "content": msg["content"]}
                for msg in st.session_state.chat_messages
            ]
            # 토큰 단위 스트리밍: stream_mode="messages"
            # 참고: https://docs.langchain.com/oss/python/langchain/streaming
            for chunk in agent.stream(
                {"messages": st.session_state.chat_messages},
                {"configurable": {"thread_id": "1"}},
                context=profile,
                stream_mode="messages",
            ):
                # chunk는 (message, metadata) 튜플 형태
                # 튜플 언패킹 확인
                if not isinstance(chunk, tuple) or len(chunk) != 2:
                    continue

                msg, metadata = chunk

                # 메타데이터가 dict인지 확인
                if not isinstance(metadata, dict):
                    continue

                # 메타데이터에서 현재 노드 확인
                node_name = metadata.get("langgraph_node", "")

                # 메시지 클래스 확인
                msg_class = msg.__class__.__name__
                print(msg)

                # Tool 노드에서의 메시지 처리
                if "tools" in node_name.lower():
                    # Tool 호출 감지 (AIMessageChunk with tool_calls)
                    if msg_class == "AIMessageChunk" and hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("name", "Unknown")
                                if tool_name and tool_name != "Unknown":
                                    tool_log = f"🔧 **{tool_name}** 호출 중..."

                                    with tool_log_container:
                                        with st.chat_message("assistant", avatar="🔧"):
                                            st.caption(tool_log)

                                    st.session_state.chat_messages.append(
                                        {"role": "tool", "content": tool_log, "tool_name": tool_name}
                                    )

                    # Tool 결과 감지
                    elif msg_class == "ToolMessage":
                        tool_name = getattr(msg, "name", "Unknown")
                        tool_result = getattr(msg, "content", "")
                        tool_call_id = getattr(msg, "tool_call_id", "")

                        with tool_log_container:
                            with st.chat_message("assistant", avatar="✅"):
                                with st.expander(f"📦 {tool_name} 결과", expanded=False):
                                    result_preview = str(tool_result)[:500]
                                    if len(str(tool_result)) > 500:
                                        result_preview += "..."
                                    st.text(result_preview)

                        st.session_state.chat_messages.append(
                            {"role": "tool", "content": tool_result, "tool_name": tool_name, "tool_call_id": tool_call_id}
                        )

                # LLM 노드에서의 토큰 스트리밍
                elif "model" in node_name.lower() or "agent" in node_name.lower():
                    # AIMessageChunk에서 토큰 추출
                    if msg_class == "AIMessageChunk" and hasattr(msg, "content"):
                        token = getattr(msg, "content", "")
                        if token:
                            # Tool calls가 없는 경우만 응답 토큰으로 간주
                            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
                            if not has_tool_calls:
                                full_response += token
                                # 실시간 토큰 표시 (커서 효과)
                                response_placeholder.markdown(full_response + "▌")

            # 최종 응답 표시 (커서 제거)
            if full_response:
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
