"""FalkorDB 그래프 데이터베이스 유틸리티."""

import os
from typing import Any

from dotenv import load_dotenv
from falkordb import FalkorDB

load_dotenv()


def get_falkordb_client() -> FalkorDB:
    """FalkorDB 클라이언트 생성.

    Returns:
        FalkorDB 클라이언트 인스턴스
    """
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))

    try:
        client = FalkorDB(host=host, port=port)
        return client
    except Exception as e:
        raise ConnectionError(f"FalkorDB 연결 실패: {e}")


def get_graph(graph_name: str = "mid_level_helper"):
    """그래프 인스턴스 가져오기.

    Args:
        graph_name: 그래프 이름

    Returns:
        Graph 인스턴스
    """
    client = get_falkordb_client()
    return client.select_graph(graph_name)


def create_graph_schema(graph_name: str = "mid_level_helper") -> None:
    """그래프 스키마 생성 (인덱스 및 제약조건).

    그래프 스키마:
        노드:
            - Document: 문서 (id, title, source, problem_summary, category)
            - Keyword: 키워드 (name)
            - Category: 카테고리 (name)

        관계:
            - (Document)-[HAS_KEYWORD]->(Keyword)
            - (Document)-[BELONGS_TO]->(Category)
            - (Keyword)-[CO_OCCURS_WITH {weight}]->(Keyword)

    Args:
        graph_name: 그래프 이름
    """
    graph = get_graph(graph_name)

    # 인덱스 생성 (성능 최적화)
    index_queries = [
        "CREATE INDEX FOR (d:Document) ON (d.id)",
        "CREATE INDEX FOR (k:Keyword) ON (k.name)",
        "CREATE INDEX FOR (c:Category) ON (c.name)",
    ]

    for query in index_queries:
        try:
            graph.query(query)
            print(f"✅ 인덱스 생성: {query}")
        except Exception as e:
            # 이미 존재하는 인덱스는 무시
            if "already exists" not in str(e).lower():
                print(f"⚠️ 인덱스 생성 실패: {query} - {e}")


def clear_graph(graph_name: str = "mid_level_helper") -> None:
    """그래프의 모든 노드와 관계 삭제.

    Args:
        graph_name: 그래프 이름
    """
    graph = get_graph(graph_name)

    try:
        result = graph.query("MATCH (n) DETACH DELETE n")
        print(f"✅ 그래프 초기화 완료")
    except Exception as e:
        print(f"❌ 그래프 초기화 실패: {e}")
        raise


def get_graph_stats(graph_name: str = "mid_level_helper") -> dict[str, Any]:
    """그래프 통계 조회.

    Args:
        graph_name: 그래프 이름

    Returns:
        통계 정보 딕셔너리
    """
    graph = get_graph(graph_name)

    stats = {}

    # 노드 수 조회
    node_queries = {
        "documents": "MATCH (d:Document) RETURN count(d) as count",
        "keywords": "MATCH (k:Keyword) RETURN count(k) as count",
        "categories": "MATCH (c:Category) RETURN count(c) as count",
    }

    for name, query in node_queries.items():
        try:
            result = graph.query(query)
            stats[name] = result.result_set[0][0] if result.result_set else 0
        except Exception as e:
            print(f"⚠️ {name} 조회 실패: {e}")
            stats[name] = 0

    # 관계 수 조회
    relationship_queries = {
        "has_keyword": "MATCH ()-[r:HAS_KEYWORD]->() RETURN count(r) as count",
        "belongs_to": "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as count",
        "co_occurs_with": "MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) as count",
    }

    for name, query in relationship_queries.items():
        try:
            result = graph.query(query)
            stats[name] = result.result_set[0][0] if result.result_set else 0
        except Exception as e:
            print(f"⚠️ {name} 조회 실패: {e}")
            stats[name] = 0

    return stats


def print_graph_stats(graph_name: str = "mid_level_helper") -> None:
    """그래프 통계 출력.

    Args:
        graph_name: 그래프 이름
    """
    stats = get_graph_stats(graph_name)

    print("\n" + "=" * 60)
    print("📊 그래프 통계")
    print("=" * 60)
    print(f"그래프 이름: {graph_name}")
    print("\n노드:")
    print(f"  - Document: {stats.get('documents', 0):,}개")
    print(f"  - Keyword: {stats.get('keywords', 0):,}개")
    print(f"  - Category: {stats.get('categories', 0):,}개")
    print("\n관계:")
    print(f"  - HAS_KEYWORD: {stats.get('has_keyword', 0):,}개")
    print(f"  - BELONGS_TO: {stats.get('belongs_to', 0):,}개")
    print(f"  - CO_OCCURS_WITH: {stats.get('co_occurs_with', 0):,}개")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 테스트 실행
    print("🔧 FalkorDB 연결 테스트...")

    try:
        client = get_falkordb_client()
        print("✅ FalkorDB 연결 성공")

        # 그래프 스키마 생성
        print("\n🔨 그래프 스키마 생성...")
        create_graph_schema()

        # 통계 출력
        print_graph_stats()

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
