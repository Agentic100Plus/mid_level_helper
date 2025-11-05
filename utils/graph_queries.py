"""FalkorDB 그래프 쿼리 함수."""

from typing import Any

from utils.graph_db import get_graph


def search_documents_by_keywords(
    keywords: list[str], graph_name: str = "mid_level_helper", limit: int = 10
) -> list[dict[str, Any]]:
    """키워드로 관련 문서 검색.

    Args:
        keywords: 검색할 키워드 리스트
        graph_name: 그래프 이름
        limit: 반환할 최대 문서 수

    Returns:
        문서 정보 리스트 (id, title, category, problem_summary, matched_keywords, relevance_score)
    """
    if not keywords:
        return []

    graph = get_graph(graph_name)

    # Cypher 쿼리: 키워드와 매칭되는 문서 찾기
    query = """
    UNWIND $keywords AS keyword
    MATCH (d:Document)-[:HAS_KEYWORD]->(k:Keyword)
    WHERE k.name = keyword
    WITH d, collect(DISTINCT k.name) AS matched_keywords, count(DISTINCT k) AS relevance_score
    RETURN d.id AS id,
           d.title AS title,
           d.category AS category,
           d.problem_summary AS problem_summary,
           d.source AS source,
           matched_keywords,
           relevance_score
    ORDER BY relevance_score DESC
    LIMIT $limit
    """

    try:
        result = graph.query(query, {"keywords": keywords, "limit": limit})

        documents = []
        for row in result.result_set:
            documents.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "category": row[2],
                    "problem_summary": row[3],
                    "source": row[4],
                    "matched_keywords": row[5],
                    "relevance_score": row[6],
                }
            )

        return documents
    except Exception as e:
        print(f"❌ 쿼리 실패: {e}")
        return []


def get_related_keywords(
    keyword: str, graph_name: str = "mid_level_helper", limit: int = 10
) -> list[dict[str, Any]]:
    """특정 키워드와 관련된 키워드 찾기 (공동 출현 기반).

    Args:
        keyword: 기준 키워드
        graph_name: 그래프 이름
        limit: 반환할 최대 키워드 수

    Returns:
        관련 키워드 리스트 (name, weight, documents_count)
    """
    graph = get_graph(graph_name)

    query = """
    MATCH (k1:Keyword {name: $keyword})-[r:CO_OCCURS_WITH]-(k2:Keyword)
    OPTIONAL MATCH (k2)<-[:HAS_KEYWORD]-(d:Document)
    WITH k2, r.weight AS weight, count(DISTINCT d) AS documents_count
    RETURN k2.name AS name, weight, documents_count
    ORDER BY weight DESC
    LIMIT $limit
    """

    try:
        result = graph.query(query, {"keyword": keyword, "limit": limit})

        related_keywords = []
        for row in result.result_set:
            related_keywords.append({"name": row[0], "weight": row[1], "documents_count": row[2]})

        return related_keywords
    except Exception as e:
        print(f"❌ 쿼리 실패: {e}")
        return []


def get_documents_by_category(
    category: str, graph_name: str = "mid_level_helper", limit: int = 10
) -> list[dict[str, Any]]:
    """카테고리별 문서 검색.

    Args:
        category: 카테고리 이름
        graph_name: 그래프 이름
        limit: 반환할 최대 문서 수

    Returns:
        문서 정보 리스트
    """
    graph = get_graph(graph_name)

    query = """
    MATCH (d:Document)-[:BELONGS_TO]->(c:Category {name: $category})
    OPTIONAL MATCH (d)-[:HAS_KEYWORD]->(k:Keyword)
    WITH d, collect(DISTINCT k.name) AS keywords
    RETURN d.id AS id,
           d.title AS title,
           d.category AS category,
           d.problem_summary AS problem_summary,
           d.source AS source,
           keywords
    LIMIT $limit
    """

    try:
        result = graph.query(query, {"category": category, "limit": limit})

        documents = []
        for row in result.result_set:
            documents.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "category": row[2],
                    "problem_summary": row[3],
                    "source": row[4],
                    "keywords": row[5],
                }
            )

        return documents
    except Exception as e:
        print(f"❌ 쿼리 실패: {e}")
        return []


def get_keyword_network(
    keyword: str, graph_name: str = "mid_level_helper", depth: int = 2
) -> dict[str, Any]:
    """키워드 네트워크 탐색 (N-hop 이웃).

    Args:
        keyword: 중심 키워드
        graph_name: 그래프 이름
        depth: 탐색 깊이 (1 = 직접 연결, 2 = 2-hop 이웃)

    Returns:
        네트워크 정보 (nodes, edges)
    """
    graph = get_graph(graph_name)

    # 가변 깊이 경로 쿼리
    query = f"""
    MATCH path = (k1:Keyword {{name: $keyword}})-[:CO_OCCURS_WITH*1..{depth}]-(k2:Keyword)
    WITH k1, k2, relationships(path) AS rels
    UNWIND rels AS r
    WITH DISTINCT startNode(r) AS start_node, endNode(r) AS end_node, r.weight AS weight
    RETURN start_node.name AS source, end_node.name AS target, weight
    """

    try:
        result = graph.query(query, {"keyword": keyword})

        nodes = set()
        edges = []

        for row in result.result_set:
            source, target, weight = row[0], row[1], row[2]
            nodes.add(source)
            nodes.add(target)
            edges.append({"source": source, "target": target, "weight": weight})

        return {"nodes": list(nodes), "edges": edges}
    except Exception as e:
        print(f"❌ 쿼리 실패: {e}")
        return {"nodes": [], "edges": []}


def get_top_keywords_by_category(
    category: str, graph_name: str = "mid_level_helper", limit: int = 10
) -> list[dict[str, Any]]:
    """카테고리별 상위 키워드 조회.

    Args:
        category: 카테고리 이름
        graph_name: 그래프 이름
        limit: 반환할 최대 키워드 수

    Returns:
        키워드 리스트 (name, count)
    """
    graph = get_graph(graph_name)

    query = """
    MATCH (d:Document)-[:BELONGS_TO]->(c:Category {name: $category})
    MATCH (d)-[:HAS_KEYWORD]->(k:Keyword)
    WITH k, count(d) AS count
    RETURN k.name AS name, count
    ORDER BY count DESC
    LIMIT $limit
    """

    try:
        result = graph.query(query, {"category": category, "limit": limit})

        keywords = []
        for row in result.result_set:
            keywords.append({"name": row[0], "count": row[1]})

        return keywords
    except Exception as e:
        print(f"❌ 쿼리 실패: {e}")
        return []


def get_similar_documents_by_keywords(
    doc_id: str, graph_name: str = "mid_level_helper", limit: int = 5
) -> list[dict[str, Any]]:
    """특정 문서와 유사한 문서 찾기 (공통 키워드 기반).

    Args:
        doc_id: 기준 문서 ID
        graph_name: 그래프 이름
        limit: 반환할 최대 문서 수

    Returns:
        유사 문서 리스트 (id, title, category, common_keywords, similarity_score)
    """
    graph = get_graph(graph_name)

    query = """
    MATCH (d1:Document {id: $doc_id})-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(d2:Document)
    WHERE d1 <> d2
    WITH d2, collect(DISTINCT k.name) AS common_keywords, count(DISTINCT k) AS similarity_score
    RETURN d2.id AS id,
           d2.title AS title,
           d2.category AS category,
           d2.problem_summary AS problem_summary,
           common_keywords,
           similarity_score
    ORDER BY similarity_score DESC
    LIMIT $limit
    """

    try:
        result = graph.query(query, {"doc_id": doc_id, "limit": limit})

        documents = []
        for row in result.result_set:
            documents.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "category": row[2],
                    "problem_summary": row[3],
                    "common_keywords": row[4],
                    "similarity_score": row[5],
                }
            )

        return documents
    except Exception as e:
        print(f"❌ 쿼리 실패: {e}")
        return []


def get_all_categories(graph_name: str = "mid_level_helper") -> list[str]:
    """모든 카테고리 목록 조회.

    Args:
        graph_name: 그래프 이름

    Returns:
        카테고리 이름 리스트
    """
    graph = get_graph(graph_name)

    query = """
    MATCH (c:Category)
    RETURN c.name AS name
    ORDER BY name
    """

    try:
        result = graph.query(query)
        return [row[0] for row in result.result_set]
    except Exception as e:
        print(f"❌ 쿼리 실패: {e}")
        return []


if __name__ == "__main__":
    # 테스트 실행
    print("🔍 그래프 쿼리 테스트\n")

    # 1. 카테고리 목록
    print("=" * 60)
    print("📂 카테고리 목록:")
    categories = get_all_categories()
    for cat in categories:
        print(f"   - {cat}")

    # 2. 키워드로 문서 검색
    print("\n" + "=" * 60)
    print("🔎 키워드 검색 테스트: ['성장통', '재택근무']")
    docs = search_documents_by_keywords(["성장통", "재택근무"], limit=3)
    for doc in docs:
        print(f"\n   [{doc['relevance_score']}점] {doc['title']}")
        print(f"   카테고리: {doc['category']}")
        print(f"   매칭 키워드: {', '.join(doc['matched_keywords'])}")

    # 3. 관련 키워드
    print("\n" + "=" * 60)
    print("🔗 관련 키워드: '성장통'")
    related = get_related_keywords("성장통", limit=5)
    for kw in related:
        print(f"   - {kw['name']} (공동출현: {kw['weight']}회, 문서: {kw['documents_count']}개)")

    # 4. 카테고리별 상위 키워드
    if categories:
        print("\n" + "=" * 60)
        print(f"📊 카테고리 '{categories[0]}' 상위 키워드:")
        top_kw = get_top_keywords_by_category(categories[0], limit=5)
        for kw in top_kw:
            print(f"   - {kw['name']}: {kw['count']}개 문서")

    print("\n" + "=" * 60)
