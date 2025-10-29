from dataclasses import dataclass
from datetime import datetime


@dataclass
class UserProfile:
    """사용자 프로필"""

    name: str
    career_level: str  # 주니어, 중니어, 시니어
    years_of_experience: int
    job_role: str  # 백엔드, 프론트엔드, 풀스택, 데이터, devops, 안드로이드, ios
    tech_stack: list[str]  # [python, react, aws]

    # 선택 정보
    company_size: str | None = None  # "스타트업", "중견", "대기업"
    work_style: str | None = None  # "재택", "출근", "하이브리드"

    # 메타 정보
    created_at: datetime = None

    def to_context_string(self) -> str:
        """프로필을 컨텍스트 문자열로 변환"""
        return f"""
경력: {self.career_level} ({self.years_of_experience}년차)
직무: {self.job_role}
기술스택: {", ".join(self.tech_stack)}
회사규모: {self.company_size or "미지정"}
근무형태: {self.work_style or "미지정"}
        """.strip()
