"""
전문가 어드바이스 툴: 직무별 맞춤형 프롬프트 제공

MCP 스타일 동적 프롬프트 로딩 시스템
"""

from pathlib import Path

import streamlit as st
import yaml
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from schemas import JobRole


class PromptMetadata(BaseModel):
    """프롬프트 메타데이터 (YAML frontmatter)"""

    name: str = Field(description="프롬프트 이름")
    description: str = Field(description="프롬프트 설명")
    category: str = Field(description="카테고리 (engineering, analysis, etc.)")


class ExpertPrompt(BaseModel):
    """전문가 프롬프트 전체 구조"""

    metadata: PromptMetadata
    content: str = Field(description="프롬프트 본문 (Markdown)")
    file_path: str = Field(description="원본 파일 경로")


class PromptLoader:
    """
    MCP 스타일 프롬프트 동적 로더

    Features:
    - 자동 발견: prompts/ 디렉토리 스캔
    - Lazy Loading: 필요한 프롬프트만 로드
    - 캐싱: 내부 딕셔너리로 중복 로드 방지
    - Fallback: 매칭 실패시 기본 프롬프트
    - Streamlit 통합: @st.cache_resource로 앱 전체 공유
    """

    # JobRole → Prompt 파일명 매핑
    ROLE_TO_PROMPT_MAP = {
        JobRole.BACKEND: "backend-architect",
        JobRole.FRONTEND: "frontend-architect",
        JobRole.DEVOPS: "devops-architect",
        JobRole.DATA_ENGINEER: "python-expert",
        JobRole.ML_ENGINEER: "python-expert",
        JobRole.FULLSTACK: "system-architect",  # 풀스택은 시스템 아키텍트
        JobRole.IOS: "frontend-architect",  # iOS도 프론트엔드 범주
        JobRole.ANDROID: "frontend-architect",  # AOS도 프론트엔드 범주
        JobRole.ETC: "learning-guide",  # 기타는 학습 가이드
    }

    def __init__(self, prompts_dir: str | None = None):
        """
        Args:
            prompts_dir: 프롬프트 디렉토리 경로 (기본: 프로젝트 루트/prompts)
        """
        if prompts_dir is None:
            # 현재 파일 기준으로 프로젝트 루트 찾기
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            prompts_dir = project_root / "prompts"  # type: ignore

        self.prompts_dir = Path(prompts_dir)  # type: ignore
        self._cache: dict[str, ExpertPrompt] = {}  # 내부 캐시 딕셔너리

        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

    def load_prompt(self, prompt_name: str) -> ExpertPrompt | None:
        """
        단일 프롬프트 로드 (캐싱됨)

        Args:
            prompt_name: 프롬프트 파일명 (확장자 제외)

        Returns:
            ExpertPrompt 또는 None (파일 없음)
        """
        # 캐시 확인
        if prompt_name in self._cache:
            return self._cache[prompt_name]

        file_path = self.prompts_dir / f"{prompt_name}.md"

        if not file_path.exists():
            return None

        # 파일 읽기
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # YAML frontmatter + Markdown 분리
        if content.startswith("---"):
            # frontmatter 파싱
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1].strip()
                markdown_content = parts[2].strip()

                # YAML 파싱
                try:
                    frontmatter_data = yaml.safe_load(frontmatter_str)
                    metadata = PromptMetadata(**frontmatter_data)
                except (yaml.YAMLError, ValueError) as e:
                    print(f"⚠️ YAML 파싱 실패: {file_path.name} - {e}")
                    return None

                prompt = ExpertPrompt(
                    metadata=metadata,
                    content=markdown_content,
                    file_path=str(file_path),
                )

                # 캐시에 저장
                self._cache[prompt_name] = prompt
                return prompt

        # frontmatter 없는 경우
        print(f"⚠️ Frontmatter 없음: {file_path.name}")
        return None

    def scan_all_prompts(self) -> dict[str, ExpertPrompt]:
        """
        모든 프롬프트 스캔 및 로드

        Returns:
            {prompt_name: ExpertPrompt} 딕셔너리
        """
        prompts = {}

        for md_file in self.prompts_dir.glob("*.md"):
            prompt_name = md_file.stem
            prompt = self.load_prompt(prompt_name)

            if prompt:
                prompts[prompt_name] = prompt

        return prompts

    def get_by_role(self, job_role: JobRole) -> ExpertPrompt | None:
        """
        JobRole에 맞는 프롬프트 반환

        Args:
            job_role: 직무 역할 (Enum)

        Returns:
            해당 역할의 ExpertPrompt 또는 None
        """
        # 매핑 테이블에서 프롬프트명 찾기
        prompt_name = self.ROLE_TO_PROMPT_MAP.get(job_role)

        if not prompt_name:
            print(f"⚠️ JobRole 매핑 없음: {job_role}")
            return None

        # 프롬프트 로드 (캐싱됨)
        return self.load_prompt(prompt_name)

    def get_fallback_prompt(self) -> str:
        """
        매칭 실패시 기본 프롬프트

        Returns:
            범용 어드바이스 프롬프트
        """
        return """
## 중니어 개발자를 위한 커리어 조언

당신은 중급 개발자(중니어)의 커리어 성장을 돕는 전문 컨설턴트입니다.

### 주요 접근 방식:
1. **기술 역량 평가**: 현재 기술 스택과 경험 연차 분석
2. **성장 방향 제시**: 개인 상황에 맞는 구체적이고 실행 가능한 성장 로드맵
3. **실무 인사이트**: 실제 현장 경험과 사례 기반 조언
4. **장기 관점**: 단기 해결책이 아닌 지속 가능한 커리어 발전 방향

### 조언 원칙:
- 구체적이고 실행 가능한 액션 아이템 제시
- 개인의 상황과 관심사 고려
- 업계 트렌드와 실무 경험 반영
- 긍정적이면서도 현실적인 피드백
"""


@st.cache_resource(show_spinner="🔄 전문가 프롬프트 로더 초기화 중...", ttl=3600)
def get_prompt_loader() -> PromptLoader:
    """
    PromptLoader 인스턴스 반환 (Streamlit 캐싱)

    Streamlit cache_resource로 앱 전체에서 하나의 인스턴스만 생성
    TTL: 1시간 (3600초)

    Returns:
        PromptLoader 인스턴스
    """
    return PromptLoader()


# ========================================
# LangChain Tool 정의
# ========================================


class ExpertSearchInput(BaseModel):
    """전문가 검색 입력 스키마"""

    job_role: str = Field(description="직무 (백엔드, 프론트엔드, DevOps, iOS, AOS, 풀스택, 데이터 엔지니어, ML 엔지니어, 기타)")


@tool("expert", args_schema=ExpertSearchInput)
def expert_search(job_role: str, runtime: ToolRuntime | None = None) -> str:
    """
    직무에 맞는 전문가 어드바이스 프롬프트 검색 도구

    사용자의 직무에 따라 특화된 커리어 조언과 성장 가이드를 제공합니다.

    Args:
        job_role: 사용자의 직무 (예: "백엔드", "프론트엔드", "DevOps")
        runtime: LangGraph 런타임 컨텍스트 (optional)

    Returns:
        해당 직무에 맞는 전문가 프롬프트 텍스트
    """
    # Stream writer 초기화
    writer = runtime.stream_writer if runtime else None
    if writer:
        writer(f"🔍 전문가 어드바이스 검색 중: {job_role}")

    # PromptLoader 인스턴스
    loader = get_prompt_loader()

    # job_role 문자열 → JobRole Enum 변환
    try:
        # "백엔드" → JobRole.BACKEND
        role_enum = None
        for role in JobRole:
            if role.value == job_role:
                role_enum = role
                break

        if not role_enum:
            if writer:
                writer(f"⚠️ 지원하지 않는 직무: {job_role}")
            return loader.get_fallback_prompt()

    except ValueError:
        if writer:
            writer(f"⚠️ 직무 파싱 실패: {job_role}")
        return loader.get_fallback_prompt()

    # JobRole에 맞는 프롬프트 로드
    expert_prompt = loader.get_by_role(role_enum)

    if not expert_prompt:
        if writer:
            writer("⚠️ 프롬프트 로드 실패, Fallback 사용")
        return loader.get_fallback_prompt()

    # 성공
    if writer:
        writer(f"✅ 프롬프트 로드 성공: {expert_prompt.metadata.name}")

    # 메타데이터 + 본문 결합
    result = f"""# {expert_prompt.metadata.name}

**설명**: {expert_prompt.metadata.description}
**카테고리**: {expert_prompt.metadata.category}

---

{expert_prompt.content}
"""

    return result


# ========================================
# CLI 테스트용 (선택사항)
# ========================================

if __name__ == "__main__":
    """
    테스트 실행:
    python -m tools.expert_advice
    """
    print("=" * 80)
    print("🔧 Expert Advice Prompt Loader Test")
    print("=" * 80)

    loader = get_prompt_loader()

    # 1. 모든 프롬프트 스캔
    print("\n📂 Scanning all prompts...")
    all_prompts = loader.scan_all_prompts()
    print(f"✅ Found {len(all_prompts)} prompts:")
    for name, prompt in all_prompts.items():
        print(f"  - {name}: {prompt.metadata.description}")

    # 2. JobRole별 테스트
    print("\n🎯 Testing JobRole mappings...")
    test_roles = [JobRole.BACKEND, JobRole.FRONTEND, JobRole.DEVOPS, JobRole.DATA_ENGINEER]

    for role in test_roles:
        prompt = loader.get_by_role(role)
        if prompt:
            print(f"  ✅ {role.value}: {prompt.metadata.name}")
        else:
            print(f"  ❌ {role.value}: Not found")

    # 3. Tool 실행 테스트
    print("\n🔨 Testing LangChain tool...")
    result = expert_search.invoke({"job_role": "백엔드"})
    print(f"✅ Tool returned {len(result)} characters")
    print(f"\nFirst 300 chars:\n{result[:300]}...")
