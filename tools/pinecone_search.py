"""
Upstage 임베딩을 통한 Retriever 구현
테스트 코드: tests/test_retriever.py

main.py 에서 구현한 캐시 리소스 사용 권장
"""

import os

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from main import get_pinecone, get_upstage

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "mid-level-helper")
namespace = "20251029_crawling"


def _get_index():
    """Pinecone Index 반환 (함수 호출)"""
    return get_pinecone


class PineconeSchemas(BaseModel):
    title: str = Field(description="제목")
    category: str = Field(description="카테고리")
    summary: str = Field(description="요약")
    keywords: str = Field(description="키워드")
    similarity: float = Field(description="유사도")
    source: str = Field(description="출처")


class RagToolResponseSchemas(BaseModel):
    count: int = Field(description="결과 수")
    cases: list[PineconeSchemas] = Field(description="결과 데이터")


class PineconeSearchInput(BaseModel):
    """Pinecone 검색 입력 스키마"""

    query: str = Field(description="검색할 고민이나 상황 (예: '성장 슬럼프', '이직 고민')")


def _create_query_embedding(query_text: str) -> list[float]:
    """쿼리 텍스트를 Upstage 임베딩으로 변환"""
    response = get_upstage.embeddings.create(input=[query_text], model="embedding-query")
    return response.data[0].embedding


@tool("pinecone_search", args_schema=PineconeSearchInput)
def sementic_search(query: str, runtime: ToolRuntime | None = None) -> RagToolResponseSchemas:
    """Search for similar cases on concerns, reflections, emotions, and more in the Vector Store.

    Args:
        query(str): search terms to look for

    Returns:
        class PineconeSchemas(BaseModel):
            title: str = Field(description="제목")
            category: str = Field(description="카테고리")
            summary: str = Field(description="요약")
            keywords: str = Field(description="키워드")
            similarity: float = Field(description="유사도")
            source: str = Field(description="출처")

        class RagToolResponseSchemas(BaseModel):
            count: int = Field(description="결과 수")
            cases: list[PineconeSchemas] = Field(description="결과 데이터")

        return objects: RagToolResponseSchemas
    """
    if runtime:
        writer = runtime.stream_writer
        writer(f"✨ Search Reference Datas: [{query}]")

    query_embedding = _create_query_embedding(query)
    index = _get_index()  # Pinecone Index 객체 가져오기

    results = index.query(
        namespace=namespace,
        vector=query_embedding,
        top_k=5,  # 상위 5개
        include_metadata=True,  # 메타데이터 포함
    )

    if results and runtime:
        writer(f"✨ Find Datas: {len(results.matches)}")  # type: ignore
    cases = []
    for match in results.matches:  # type: ignore
        cases.append(
            PineconeSchemas(
                title=match.metadata.get("title", "N/A"),
                category=match.metadata.get("category", "N/A"),
                summary=match.metadata.get("problem_summary", "N/A"),
                keywords=match.metadata.get("keywords", "N/A"),
                similarity=round(match.score, 2),
                source=match.metadata.get("source", "N/A"),
            )
        )
    return RagToolResponseSchemas(cases=cases, count=len(cases))
