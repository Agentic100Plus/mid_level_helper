from schemas import CareerLevel


class CommonCompetencies:
    COMMON_COMPETENCIES = {
        CareerLevel.JUNIOR: [
            "기본적인 CS 지식 (자료구조, 알고리즘)",
            "Git 버전 관리",
            "코드 리뷰 참여 및 피드백 수용",
            "기술 문서 작성 및 읽기",
            "문제 해결을 위한 검색 능력",
            "팀 커뮤니케이션",
        ],
        CareerLevel.MID: [
            "설계 패턴 및 아키텍처 이해",
            "코드 리뷰 주도",
            "주니어 멘토링",
            "프로젝트 일정 관리",
            "성능 최적화",
            "크로스팀 협업",
            "기술 의사결정 참여",
        ],
        CareerLevel.SENIOR: [
            "시스템 아키텍처 설계",
            "기술 리더십",
            "비즈니스 이해 및 기술 전환",
            "기술 로드맵 수립",
            "기술 부채 관리",
            "조직 표준 수립",
            "채용 및 평가",
        ],
    }

    APPROACHES = {
        CareerLevel.JUNIOR: """
- 기초를 탄탄히 다지는 것에 집중
- 실무 경험을 쌓을 수 있는 프로젝트 추천
- 학습 방법과 효율적인 성장 경로 제시
- 좋은 코딩 습관과 협업 방법 강조
- 멘토 찾기 및 커뮤니티 참여 독려
            """,
        CareerLevel.MID: """
- 전문성 깊이를 더하는 방향 제시
- 리더십과 멘토링 역량 개발 지원
- 아키텍처 및 설계 능력 향상 가이드
- 프로젝트 주도 경험 축적 방법 제안
- 기술 의사결정 능력 강화
            """,
        CareerLevel.SENIOR: """
- 기술 리더십과 조직 영향력 확대
- 시스템적 사고와 전략적 의사결정
- 팀/조직 성장을 위한 방법론 제시
- 기술 비전 수립 및 로드맵 작성 지원
- 후배 양성 및 문화 구축 가이드
            """,
    }

    def __init__(self) -> None:
        pass

    def _get_level(self, _level: str) -> CareerLevel:
        if _level == "주니어":
            return CareerLevel.JUNIOR
        elif _level == "중니어":
            return CareerLevel.MID
        else:
            return CareerLevel.SENIOR

    # 공통 역량 조회
    def get_common_comps(self, _level: str) -> list[str]:
        level = self._get_level(_level)

        return self.COMMON_COMPETENCIES.get(level, [])

    def get_coaching_approach(self, _level: str) -> str:
        level = self._get_level(_level)

        return self.APPROACHES.get(level, "")
