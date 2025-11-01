from pydantic import BaseModel, Field


class WebSearchSchemas(BaseModel):
    title: str = Field(description="제목")
    body: str = Field(description="본문 요약")
    href: str = Field(description="출처")


class WebSearchToolResponseSchemas(BaseModel):
    count: int = Field(description="결과 수")
    articles: list[WebSearchSchemas] = Field(description="결과 데이터")


class DDGSSearchInput(BaseModel):
    """DuckDuckGo 검색 입력 스키마"""

    query: str = Field(description="사용자 질문에 대한 웹 검색 쿼리")
    page: int = Field(default=1, description="검색 페이지 번호 (기본값: 1, 페이지당 최대 10개)")
