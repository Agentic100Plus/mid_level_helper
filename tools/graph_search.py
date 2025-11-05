"""그래프 데이터베이스 검색 도구 (LangChain Tool)."""

from langchain.tools import tool

from utils.graph_queries import (
    get_related_keywords,
    search_documents_by_keywords,
)


@tool
def graph_keyword_search(keywords: str) -> str:
    """키워드 기반 그래프 검색으로 관련 개발자 사례를 찾습니다.

    이 도구는 FalkorDB 그래프 데이터베이스를 사용하여 키워드와 연결된 문서를 검색합니다.
    키워드 간의 관계(공동 출현)를 활용하여 더 정확한 검색 결과를 제공합니다.

    Args:
        keywords: 검색할 키워드들 (쉼표로 구분, 예: "성장통, 재택근무, 동기부여")

    Returns:
        검색된 문서 정보 (제목, 카테고리, 문제 요약, 매칭 키워드)
    """
    # 키워드 파싱
    keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]

    if not keyword_list:
        return "검색할 키워드를 입력해주세요."

    try:
        # 그래프 검색 실행
        documents = search_documents_by_keywords(keyword_list, limit=5)

        if not documents:
            return f"키워드 '{keywords}'와 관련된 문서를 찾을 수 없습니다."

        # 결과 포매팅
        result_lines = [f"🔍 키워드 '{keywords}' 검색 결과: {len(documents)}개 문서 발견\n"]

        for i, doc in enumerate(documents, 1):
            matched_kw = ", ".join(doc["matched_keywords"])
            result_lines.append(f"[{i}] {doc['title']}")
            result_lines.append(f"    카테고리: {doc['category']}")
            result_lines.append(f"    매칭 키워드: {matched_kw}")
            result_lines.append(f"    관련도: {doc['relevance_score']}점")
            result_lines.append(f"    문제: {doc['problem_summary'][:100]}...")
            result_lines.append("")

        return "\n".join(result_lines)

    except Exception as e:
        return f"❌ 그래프 검색 중 오류 발생: {str(e)}"


@tool
def graph_related_keywords(keyword: str) -> str:
    """특정 키워드와 관련된 다른 키워드들을 찾습니다.

    이 도구는 그래프 데이터베이스에서 키워드 간의 공동 출현 관계를 분석하여
    연관된 키워드를 추천합니다. 사용자의 고민을 확장하거나 구체화하는 데 유용합니다.

    Args:
        keyword: 기준 키워드 (예: "성장통")

    Returns:
        관련 키워드 목록 (공동 출현 빈도 순)
    """
    if not keyword:
        return "검색할 키워드를 입력해주세요."

    try:
        # 관련 키워드 검색
        related = get_related_keywords(keyword, limit=10)

        if not related:
            return f"키워드 '{keyword}'와 관련된 키워드를 찾을 수 없습니다."

        # 결과 포매팅
        result_lines = [f"🔗 '{keyword}'와 관련된 키워드:\n"]

        for i, kw in enumerate(related, 1):
            result_lines.append(f"{i}. {kw['name']}")
            result_lines.append(f"   - 공동 출현: {kw['weight']}회")
            result_lines.append(f"   - 관련 문서: {kw['documents_count']}개")

        return "\n".join(result_lines)

    except Exception as e:
        return f"❌ 관련 키워드 검색 중 오류 발생: {str(e)}"


# 도구 목록 (export)
graph_tools = [graph_keyword_search, graph_related_keywords]


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 그래프 검색 도구 테스트\n")

    # 1. 키워드 검색
    print("=" * 60)
    print("테스트 1: 키워드 검색")
    print("=" * 60)
    result1 = graph_keyword_search.invoke({"keywords": "성장통, 재택근무"})
    print(result1)

    # 2. 관련 키워드
    print("\n" + "=" * 60)
    print("테스트 2: 관련 키워드")
    print("=" * 60)
    result2 = graph_related_keywords.invoke({"keyword": "성장통"})
    print(result2)
