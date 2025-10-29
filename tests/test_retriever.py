"""
Pinecone 기반 retriever 구현 및 테스트
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# 환경 변수
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "mid-level-helper")
namespace = "20251029_crawling"

# Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)

# Upstage 임베딩 클라이언트 (OpenAI 호환)
upstage_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1/solar")


def create_query_embedding(query_text: str) -> list[float]:
    """쿼리 텍스트를 Upstage 임베딩으로 변환"""
    response = upstage_client.embeddings.create(input=[query_text], model="embedding-query")
    return response.data[0].embedding


class TestRetrieverPineConeClass:
    def test_index_load(self):
        """인덱스 로드 테스트"""
        if index_name not in pc.list_indexes().names():
            print("☠️ Pinecone 인덱스가 없습니다.")
            print(f"사용 가능한 인덱스: {pc.list_indexes().names()}")
            raise Exception("인덱스를 찾을 수 없습니다")

        # 인덱스 로드
        index = pc.Index(index_name)

        # 인덱스 통계 출력
        stats = index.describe_index_stats()
        print(f"✅ 인덱스 로드 성공: {index_name}")
        print(f"  - 총 벡터 수: {stats.total_vector_count}")
        assert 3000 == stats.total_vector_count
        print(f"  - 차원: {stats.dimension}")
        assert 4096 == stats.dimension
        print(f"  - 네임스페이스: {stats.namespaces}")

    def test_semantic_search(self):
        """시맨틱 검색 테스트"""
        query = "재택근무하면서 동기부여가 떨어져요"

        print(f"\n🔍 검색 쿼리: {query}")
        print(f"네임스페이스: {namespace}")

        # 1. 쿼리를 임베딩으로 변환
        print("⏳ 쿼리 임베딩 생성 중...")
        query_embedding = create_query_embedding(query)
        print(f"✅ 임베딩 생성 완료 (차원: {len(query_embedding)})")

        # 2. Pinecone에서 유사 벡터 검색
        index = pc.Index(index_name)

        results = index.query(
            namespace=namespace,
            vector=query_embedding,  # ✅ 임베딩 벡터 사용
            top_k=5,  # 상위 5개
            include_metadata=True,  # 메타데이터 포함
        )

        # 3. 결과 출력
        print(f"\n✅ 검색 완료: {len(results.matches)}개 결과")
        print("=" * 80)

        for i, match in enumerate(results.matches, 1):
            print(f"\n[{i}] 유사도: {match.score:.4f}")
            print(f"ID: {match.id}")

            if match.metadata:
                print(f"제목: {match.metadata.get('title', 'N/A')}")
                print(f"카테고리: {match.metadata.get('category', 'N/A')}")
                print(f"키워드: {match.metadata.get('keywords', 'N/A')}")
                print(f"출처: {match.metadata.get('source', 'N/A')}")

            print("-" * 80)

        assert results != []

    def test_filtered_search(self):
        """메타데이터 필터링 검색 테스트"""
        query = "기술 부채 관리"
        category_filter = "성장통"

        print("\n🔍 필터링 검색")
        print(f"쿼리: {query}")
        print(f"카테고리 필터: {category_filter}")

        # 쿼리 임베딩
        query_embedding = create_query_embedding(query)

        # 필터링 검색
        index = pc.Index(index_name)

        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={"category": {"$eq": category_filter}},  # 카테고리 필터
        )

        print(f"\n✅ 필터링 검색 완료: {len(results.matches)}개 결과")

        for i, match in enumerate(results.matches, 1):
            print(f"\n[{i}] 유사도: {match.score:.4f}")
            print(f"제목: {match.metadata.get('title', 'N/A')}")
            print(f"카테고리: {match.metadata.get('category', 'N/A')}")

        assert results != []


# 직접 실행 시 테스트
if __name__ == "__main__":
    tester = TestRetrieverPineConeClass()

    print("=" * 80)
    print("🧪 Pinecone Retriever 테스트 시작")
    print("=" * 80)

    # 테스트 1: 인덱스 로드
    print("\n[테스트 1] 인덱스 로드")
    tester.test_index_load()

    # 테스트 2: 시맨틱 검색
    print("\n[테스트 2] 시맨틱 검색")
    tester.test_semantic_search()

    # 테스트 3: 필터링 검색
    print("\n[테스트 3] 필터링 검색")
    tester.test_filtered_search()

    print("\n" + "=" * 80)
    print("✅ 모든 테스트 완료!")
    print("=" * 80)
