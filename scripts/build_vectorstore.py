import os
import sys

# ===== 프로젝트 루트를 경로에 추가 =====
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
# =====================================

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from utils.data_loader import load_csv_data, prepare_documents_for_vectorstore

load_dotenv()

# ============================================
# 1. Pinecone 초기화
# ============================================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME", "mid-level-helper")

if index_name not in pc.list_indexes().names():
    print(f"📦 인덱스 생성 중: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("✅ 인덱스 생성 완료")
else:
    print(f"✅ 인덱스 존재 확인: {index_name}")


# Pinecone 인덱스 로드
index = pc.Index(index_name)

# ============================================
# 2. Upstage 임베딩 클라이언트 (OpenAI Wrapper)
# ============================================
client = OpenAI(api_key=os.getenv("UPSTAGE_API_KEY"), base_url="https://api.upstage.ai/v1/solar")

print("\n" + "=" * 60)
print("📂 데이터 로드 중...")
print("=" * 60)

# 데이터 로드
df = load_csv_data()
texts, metadatas = prepare_documents_for_vectorstore(df)
print(f"✅ 데이터 준비 완료: {len(texts)}개 문서")

# Pinecone 배치 사이즈
BATCH_SIZE = 100


def create_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """텍스트 배치 -> 임베딩 변환"""
    try:
        res = client.embeddings.create(input=texts, model="embedding-query")
        return [emb.embedding for emb in res.data]
    except Exception as e:
        print(f"☠️ 임베딩 실패: {e}")
        raise


def pinecone_batch(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> list[dict]:
    """Pinecone 업로드 데이터 포맷"""
    return [
        {"id": id_, "values": embedding, "metadata": metadata} for id_, embedding, metadata in zip(ids, embeddings, metadatas)
    ]


print("\n" + "=" * 60)
print("🔄 임베딩 생성 및 업로드 중...")
print("=" * 60)

total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
uploaded_count = 0

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="배치 처리"):
    batch_texts = texts[i : i + BATCH_SIZE]
    batch_metadatas = metadatas[i : i + BATCH_SIZE]
    batch_ids = [metadata["id"] for metadata in batch_metadatas]

    # 임베딩 생성
    embeddings = create_embeddings_batch(batch_texts)

    # Pinecone 포맷 변환
    vectors = pinecone_batch(batch_ids, embeddings, batch_metadatas)

    # Pinecone에 업로드
    try:
        index.upsert(vectors=vectors, namespace="20251029_crawling")
        uploaded_count += len(vectors)
    except Exception as e:
        print(f"❌ 업로드 실패 (배치 {i // BATCH_SIZE + 1}): {e}")
        raise

print(f"\n✅ 업로드 완료: {uploaded_count}개 벡터")

# ============================================
# 5. 검증
# ============================================
print("\n" + "=" * 60)
print("🔍 검증 중...")
print("=" * 60)

stats = index.describe_index_stats()
print(f"총 벡터 수: {stats.total_vector_count}")
print(f"차원: {stats.dimension}")

# 샘플 검색 테스트
print("\n샘플 검색 테스트:")
test_query = "재택근무하면서 동기부여가 떨어져요"
test_embedding = create_embeddings_batch([test_query])[0]

results = index.query(vector=test_embedding, top_k=3, include_metadata=True, namespace="20251029_crawling")

for i, match in enumerate(results.matches, 1):
    print(f"\n[{i}] 유사도: {match.score:.4f}")
    print(f"제목: {match.metadata.get('title', 'N/A')}")
    print(f"카테고리: {match.metadata.get('category', 'N/A')}")

print("\n" + "=" * 60)
print("✅ 벡터 스토어 구축 완료!")
print("=" * 60)
