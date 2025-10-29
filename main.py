import os
from datetime import datetime

import streamlit as st
from openai import OpenAI

from schemas import UserConcern, UserProfile

# ====================================
# For Streamlit Cache Resource
# 1. Pinecone Index
# 2. Upstage Client
# 3. Google Gemini LLM
# 4. RAG Chain
# ====================================


@st.cache_resource(show_spinner="🔄 Pinecone 인덱스 로드 중...", ttl=3600)
def get_pinecone():
    """Cache: Pinecone Index"""
    from pinecone import Pinecone, ServerlessSpec

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "mid-level-helper")
        if index_name not in pc.list_indexes().names():
            print(f"📦 인덱스 생성 중: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=4096,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("✅ 인덱스 생성 완료")
        else:
            print(f"✅ 인덱스 존재 확인: {index_name}")
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"☠️ Pinecone 초기화 실패: {e}")
        st.stop()


@st.cache_resource(show_spinner="🔄 Upstage 로드 중...", ttl=3600)
def get_upstage():
    """Cache: For embedding client - OpenAI wrapper"""
    try:
        return OpenAI(api_key=os.getenv("UPSTAGE_API_KEY"), base_url="https://api.upstage.ai/v1/solar")
    except Exception as e:
        st.error(f"☠️ Upstage 초기화 실패: {e}")
        st.stop()


@st.cache_resource(show_spinner="🔄 Gemini 로드 중...", ttl=3600)
def get_gemini():
    """Cache: Gemini Loader"""
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            max_token=5000,
            max_retries=3,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    except Exception as e:
        st.error(f"☠️ Gemini 초기화 실패: {e}")
        st.stop()


# ====================================
# Main Pages: 소개 -> 프로필 -> 고민 등록
# ====================================

# Initialize session state


if "user_profile" not in st.session_state:
    st.session_state.user_profile: UserProfile | None = None

if "profile_completed" not in st.session_state:
    st.session_state.profile_completed: bool = False
    st.session_state.current_page: str = "main"

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "user_concerns" not in st.session_state:
    st.session_state.user_concerns: list[UserConcern] = []

if "search_results" not in st.session_state:
    st.session_state.search_results: list[dict] = []

st.set_page_config(
    page_title="중니어 상담소",
    page_icon="🐒",
)

with st.expander("📖 서비스 소개", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("학습된 사례", "3,001개", "+100 (월간)")
    with col2:
        st.metric("지원 카테고리", "8개", "성장통, 경력 등")
    with col3:
        st.metric("검색 정확도", "95%", "Upstage 임베딩")

    st.info("""
    **중니어 상담소**는 3,000개 이상의 중니어 개발자 고민 사례를 학습한 AI 멘토입니다.

    - 💬 **개인화된 상담**: 당신의 프로필과 고민을 기반으로 맞춤 조언
    - 🔍 **유사 사례 검색**: 비슷한 고민을 겪은 개발자들의 경험 찾기
    - 📊 **카테고리별 분석**: 성장통, 경력 정체, 기술 부채 등 체계적 분류
    """)

st.markdown("---")
st.subheader("👤 프로필 등록")

if st.session_state.user_profile:
    st.success("✅ 프로필이 등록되었습니다!")

    with st.expander("등록된 프로필 보기"):
        profile = st.session_state.user_profile
        st.write(f"**이름**: {profile.name}")
        st.write(f"**경력**: {profile.career_level} ({profile.years_of_experience}년차)")
        st.write(f"**직무**: {profile.job_role}")
        st.write(f"**기술스택**: {', '.join(profile.tech_stack)}")

        if st.button("프로필 수정"):
            st.session_state.user_profile = None
            st.rerun()
else:
    with st.form("profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("이름 (별명 가능)", placeholder="김개발")
            career_level = st.selectbox("경력 단계", ["주니어 (0-2년)", "중니어 (3-5년)", "시니어 (6년+)"])
            years = st.number_input("연차", min_value=0, max_value=30, value=3)

        with col2:
            job_role = st.selectbox("직무", ["백엔드", "프론트엔드", "풀스택", "데이터", "DevOps", "모바일", "기타"])
            tech_stack_input = st.text_input("기술스택 (쉼표로 구분)", placeholder="Python, Django, PostgreSQL")
            company_size = st.selectbox(
                "회사 규모 (선택)", ["선택 안함", "스타트업 (1-50명)", "중견 (50-300명)", "대기업 (300명+)"]
            )

        work_style = st.radio("근무 형태 (선택)", ["선택 안함", "재택", "출근", "하이브리드"], horizontal=True)

        submitted = st.form_submit_button("프로필 저장", use_container_width=True)

        if submitted:
            if not name or not tech_stack_input:
                st.error("이름과 기술스택은 필수입니다!")
            else:
                # 프로필 생성
                tech_stack = [t.strip() for t in tech_stack_input.split(",")]

                profile = UserProfile(
                    name=name,
                    career_level=career_level.split(" ")[0],
                    years_of_experience=years,
                    job_role=job_role,
                    tech_stack=tech_stack,
                    company_size=None if company_size == "선택 안함" else company_size,
                    work_style=None if work_style == "선택 안함" else work_style,
                    created_at=datetime.now(),
                )

                st.session_state.user_profile = profile
                st.session_state.profile_completed = True
                st.success("✅ 프로필이 저장되었습니다!")
                st.rerun()

# ============================================
# 고민 등록
# ============================================
st.markdown("---")
st.subheader("💭 현재 고민 등록")

with st.form("concern_form"):
    col1, col2 = st.columns([2, 1])

    with col1:
        concern_category = st.selectbox(
            "카테고리", ["성장통", "성장 슬럼프", "경력 정체", "기술 부채", "커리어", "팀워크", "번아웃", "기타"]
        )
        concern_title = st.text_input("제목", placeholder="예: 재택근무 동기부여 문제")

    with col2:
        concern_urgency = st.radio("우선순위", ["긴급", "중요", "보통"], horizontal=False)

    concern_description = st.text_area("상세 설명", placeholder="현재 겪고 있는 고민을 자세히 적어주세요...", height=100)

    add_concern = st.form_submit_button("고민 추가", use_container_width=True)

    if add_concern:
        if not concern_title or not concern_description:
            st.error("제목과 설명을 모두 입력해주세요!")
        else:
            concern = UserConcern(
                category=concern_category,
                title=concern_title,
                description=concern_description,
                urgency=concern_urgency,
                created_at=datetime.now(),
            )

            st.session_state.user_concerns.append(concern)
            st.success(f"✅ '{concern_title}' 고민이 추가되었습니다!")
            st.rerun()


# 등록된 고민 목록
if st.session_state.user_concerns:
    st.markdown("#### 등록된 고민")

    for i, concern in enumerate(st.session_state.user_concerns):
        urgency_emoji = {"긴급": "🔴", "중요": "🟡", "보통": "🟢"}

        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f"{urgency_emoji[concern.urgency]} **[{concern.category}] {concern.title}**  \n_{concern.description[:50]}..._"
            )
        with col2:
            if st.button("삭제", key=f"delete_{i}"):
                st.session_state.user_concerns.pop(i)
                st.rerun()
else:
    st.info("아직 등록된 고민이 없습니다. 위에서 고민을 추가해보세요!")


# ============================================
# 다음 단계
# ============================================
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("💬 챗봇 상담 시작", use_container_width=True, type="primary"):
        if not st.session_state.user_profile:
            st.warning("먼저 프로필을 등록해주세요!")
        else:
            if not st.session_state.user_concerns:
                st.warning("고민도 등록해주세요!")
            else:
                st.switch_page("pages/chatbot.py")

with col2:
    if st.button("🔍 사례 검색", use_container_width=True):
        st.switch_page("pages/search.py")
