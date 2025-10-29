"""
Upstage 임베딩을 통한 Retriever 구현
테스트 코드: tests/test_retriever.py

main.py 에서 구현한 캐시 리소스 사용 권장
"""

import os

from main import get_pinecone as index
from main import get_upstage

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "mid-level-helper")
namespace = "20251029_crawling"


def _create_query_embedding(query_text: str) -> list[float]:
    """쿼리 텍스트를 Upstage 임베딩으로 변환"""
    response = get_upstage().embeddings.create(input=[query_text], model="embedding-query")
    return response.data[0].embedding


def sementic_search(query: str):
    """시멘틱 검색 텍스트"""
    query_embedding = _create_query_embedding(query)
    results = index.query(
        namespace=namespace,
        vector=query_embedding,  # ✅ 임베딩 벡터 사용
        top_k=5,  # 상위 5개
        include_metadata=True,  # 메타데이터 포함
    )

    for i, match in enumerate(results.matches, 1):
        print(f"\n[{i}] 유사도: {match.score:.4f}")
        print(f"ID: {match.id}")

        if match.metadata:
            print(f"제목: {match.metadata.get('title', 'N/A')}")
            print(f"카테고리: {match.metadata.get('category', 'N/A')}")
            print(f"키워드: {match.metadata.get('keywords', 'N/A')}")
            print(f"출처: {match.metadata.get('source', 'N/A')}")
    return results
