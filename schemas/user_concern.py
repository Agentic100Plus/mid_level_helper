from dataclasses import dataclass
from datetime import datetime


@dataclass
class UserConcern:
    """사용자 고민 데이터 모델"""

    # 고민 내용
    category: str  # "성장통", "경력 정체", "기술 부채", "커리어", "기타"
    title: str
    description: str

    # 우선순위
    urgency: str  # "긴급", "중요", "보통"

    # 메타 정보
    created_at: datetime | None = None

    def to_search_query(self) -> str:
        """검색 쿼리로 변환"""
        return f"{self.title}. {self.description}"
