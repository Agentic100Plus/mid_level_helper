"""Pinecone 벡터 데이터를 FalkorDB 그래프로 마이그레이션."""

import os
import sys
from collections import Counter, defaultdict
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import extract_keywords_list
from utils.graph_db import (
    clear_graph,
    create_graph_schema,
    get_graph,
    print_graph_stats,
)

load_dotenv()

# 설정
NAMESPACE = "20251029_crawling"
GRAPH_NAME = "mid_level_helper"
BATCH_SIZE = 100

print("\n" + "=" * 60)
print("🚀 FalkorDB 그래프 구축 시작")
print("=" * 60)

# ============================================
# 1. Pinecone 연결
# ============================================
print("\n📦 Pinecone 연결 중...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "mid-level-helper")
index = pc.Index(index_name)

# 인덱스 통계 확인
stats = index.describe_index_stats()
print(f"✅ Pinecone 인덱스: {index_name}")
print(f"   - 총 벡터 수: {stats.total_vector_count:,}")
print(f"   - 네임스페이스: {NAMESPACE}")

# ============================================
# 2. FalkorDB 초기화
# ============================================
print("\n🔨 FalkorDB 초기화 중...")
graph = get_graph(GRAPH_NAME)

# 기존 데이터 삭제 (선택적)
print("⚠️  기존 그래프 데이터를 삭제하시겠습니까? (y/N): ", end="")
response = input().strip().lower()
if response == "y":
    clear_graph(GRAPH_NAME)
    print("✅ 기존 데이터 삭제 완료")
else:
    print("⏭️  기존 데이터 유지")

# 스키마 생성
create_graph_schema(GRAPH_NAME)

# ============================================
# 3. Pinecone에서 데이터 가져오기
# ============================================
print("\n📥 Pinecone 데이터 가져오기...")


def fetch_all_vectors_from_pinecone(
    index: Any, namespace: str, batch_size: int = 100
) -> list[dict[str, Any]]:
    """Pinecone에서 모든 벡터 메타데이터 가져오기.

    Args:
        index: Pinecone 인덱스
        namespace: 네임스페이스
        batch_size: 배치 크기

    Returns:
        벡터 메타데이터 리스트
    """
    all_vectors = []

    # Pinecone의 list 메서드로 모든 ID 가져오기
    try:
        # Query 방식으로 샘플링 (Pinecone의 제한으로 인해)
        # 더미 벡터로 쿼리하여 모든 데이터 접근
        print("   벡터 데이터 샘플링 중...")

        # stats에서 네임스페이스별 벡터 수 확인
        namespace_stats = stats.namespaces.get(namespace, {})
        total_count = namespace_stats.vector_count if hasattr(namespace_stats, "vector_count") else 0

        print(f"   - 대상 벡터 수: {total_count:,}개")

        # 방법 1: 더미 쿼리로 top_k 방식 (제한적)
        # 방법 2: list_paginated를 사용한 ID 가져오기
        results = index.list_paginated(namespace=namespace, limit=10000)

        vector_ids = [v.id for v in results.vectors]
        print(f"   - 가져온 ID 수: {len(vector_ids):,}개")

        # fetch로 메타데이터 가져오기
        for i in tqdm(range(0, len(vector_ids), batch_size), desc="메타데이터 가져오기"):
            batch_ids = vector_ids[i : i + batch_size]
            fetch_result = index.fetch(ids=batch_ids, namespace=namespace)

            for vec_id, vector_data in fetch_result.vectors.items():
                metadata = vector_data.metadata
                metadata["id"] = vec_id
                all_vectors.append(metadata)

    except Exception as e:
        print(f"❌ 데이터 가져오기 실패: {e}")
        print("⚠️  대안: CSV 데이터에서 직접 로드")
        return []

    return all_vectors


vectors = fetch_all_vectors_from_pinecone(index, NAMESPACE, BATCH_SIZE)
print(f"✅ 벡터 메타데이터 가져오기 완료: {len(vectors):,}개")

# CSV 대체 방법이 필요한 경우
if len(vectors) == 0:
    print("\n⚠️  Pinecone에서 데이터를 가져올 수 없습니다.")
    print("   CSV 파일에서 직접 로드합니다...")

    from utils.data_loader import load_csv_data, prepare_documents_for_vectorstore

    df = load_csv_data()
    _, metadatas = prepare_documents_for_vectorstore(df)
    vectors = metadatas
    print(f"✅ CSV에서 로드 완료: {len(vectors):,}개")

# ============================================
# 4. 그래프 구축
# ============================================
print("\n🔨 그래프 구축 중...")

# 카테고리 노드 생성
categories = set()
for vec in vectors:
    category = vec.get("category", "기타")
    if category:
        categories.add(category)

print(f"\n📂 카테고리 노드 생성: {len(categories)}개")
for category in tqdm(categories, desc="카테고리"):
    query = f"""
    MERGE (c:Category {{name: $name}})
    """
    graph.query(query, {"name": category})

# 키워드 노드 및 문서 노드 생성
print(f"\n📄 문서 및 키워드 노드 생성: {len(vectors)}개")

keyword_counter = Counter()
keyword_cooccurrence: dict[str, Counter] = defaultdict(Counter)

for vec in tqdm(vectors, desc="문서 처리"):
    doc_id = vec.get("id", "")
    title = vec.get("title", "")
    source = vec.get("source", "")
    problem_summary = vec.get("problem_summary", "")
    category = vec.get("category", "기타")
    keywords_str = vec.get("keywords", "")

    # 문서 노드 생성
    doc_query = """
    MERGE (d:Document {id: $id})
    SET d.title = $title,
        d.source = $source,
        d.problem_summary = $problem_summary,
        d.category = $category
    """
    graph.query(
        doc_query,
        {
            "id": doc_id,
            "title": title,
            "source": source,
            "problem_summary": problem_summary,
            "category": category,
        },
    )

    # 카테고리 관계 생성
    category_rel_query = """
    MATCH (d:Document {id: $doc_id})
    MATCH (c:Category {name: $category})
    MERGE (d)-[:BELONGS_TO]->(c)
    """
    graph.query(category_rel_query, {"doc_id": doc_id, "category": category})

    # 키워드 처리
    keywords = extract_keywords_list(keywords_str)

    for keyword in keywords:
        if not keyword:
            continue

        keyword_counter[keyword] += 1

        # 키워드 노드 생성
        keyword_query = """
        MERGE (k:Keyword {name: $name})
        """
        graph.query(keyword_query, {"name": keyword})

        # 문서-키워드 관계 생성
        doc_keyword_query = """
        MATCH (d:Document {id: $doc_id})
        MATCH (k:Keyword {name: $keyword})
        MERGE (d)-[:HAS_KEYWORD]->(k)
        """
        graph.query(doc_keyword_query, {"doc_id": doc_id, "keyword": keyword})

    # 키워드 공동 출현 추적
    for i, kw1 in enumerate(keywords):
        for kw2 in keywords[i + 1 :]:
            if kw1 and kw2 and kw1 != kw2:
                keyword_cooccurrence[kw1][kw2] += 1
                keyword_cooccurrence[kw2][kw1] += 1

# 키워드 공동 출현 관계 생성
print(f"\n🔗 키워드 공동 출현 관계 생성...")
total_cooccurrences = sum(len(v) for v in keyword_cooccurrence.values()) // 2

for kw1, cooccurs in tqdm(
    keyword_cooccurrence.items(), desc="공동 출현", total=len(keyword_cooccurrence)
):
    for kw2, weight in cooccurs.items():
        if kw1 < kw2:  # 중복 방지 (양방향 중 한 번만)
            cooccur_query = """
            MATCH (k1:Keyword {name: $kw1})
            MATCH (k2:Keyword {name: $kw2})
            MERGE (k1)-[r:CO_OCCURS_WITH]-(k2)
            SET r.weight = $weight
            """
            graph.query(cooccur_query, {"kw1": kw1, "kw2": kw2, "weight": weight})

# ============================================
# 5. 결과 확인
# ============================================
print("\n" + "=" * 60)
print("✅ 그래프 구축 완료!")
print("=" * 60)

print_graph_stats(GRAPH_NAME)

# 상위 키워드 출력
print("\n📊 상위 10개 키워드:")
for keyword, count in keyword_counter.most_common(10):
    print(f"   {count:4d}회 - {keyword}")

print("\n" + "=" * 60)
print("🎉 FalkorDB 그래프 구축 완료!")
print("=" * 60)
