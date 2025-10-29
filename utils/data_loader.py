"""데이터 로딩 및 전처리 유틸리티."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_csv_data(csv_path: str = "data/mid_level_data_unique_3000.csv") -> pd.DataFrame:
    """CSV 파일 로드.

    Args:
        csv_path: CSV 파일 경로

    Returns:
        DataFrame with columns: 글 제목, 출처, 핵심 키워드, 문제점 요약, 글 내용 요약
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ 데이터 로드 완료: {len(df)}개 레코드")
    return df


def extract_category(problem_summary: str) -> str:
    """문제점 요약에서 카테고리 추출.

    Examples:
        "재택근무로 인해... (성장통 이슈 사례 1)" -> "성장통"
        "... (growth challenge 이슈 사례 802)" -> "growth challenge"

    Args:
        problem_summary: 문제점 요약 텍스트

    Returns:
        추출된 카테고리 또는 "기타"
    """
    # 한국어 카테고리 패턴
    korean_patterns = [
        r'\((\S+)\s+이슈\s+사례',  # (성장통 이슈 사례 1)
        r'\((\S+)\s+관련\s+상황',  # (성장통 관련 상황)
    ]

    for pattern in korean_patterns:
        match = re.search(pattern, problem_summary)
        if match:
            return match.group(1)

    # 영어 카테고리 패턴
    english_pattern = r'\((\w+(?:\s+\w+)?)\s+이슈\s+사례'
    match = re.search(english_pattern, problem_summary)
    if match:
        return match.group(1)

    return "기타"


def extract_keywords_list(keywords: str) -> List[str]:
    """키워드 문자열을 리스트로 변환.

    Args:
        keywords: "키워드1, 키워드2, 키워드3" 형식

    Returns:
        ["키워드1", "키워드2", "키워드3"]
    """
    if pd.isna(keywords) or not keywords:
        return []
    return [k.strip() for k in keywords.split(",") if k.strip()]


def combine_text_for_embedding(row: pd.Series) -> str:
    """레코드의 모든 텍스트 필드를 결합하여 임베딩용 텍스트 생성.

    Args:
        row: DataFrame의 한 행

    Returns:
        결합된 텍스트 (제목 + 키워드 + 문제점 요약 + 내용 요약)
    """
    parts = []

    if pd.notna(row.get("글 제목")):
        parts.append(f"제목: {row['글 제목']}")

    if pd.notna(row.get("핵심 키워드")):
        parts.append(f"키워드: {row['핵심 키워드']}")

    if pd.notna(row.get("문제점 요약")):
        parts.append(f"문제: {row['문제점 요약']}")

    if pd.notna(row.get("글 내용 요약")):
        parts.append(f"내용: {row['글 내용 요약']}")

    return "\n".join(parts)


def create_metadata(row: pd.Series, index: int) -> Dict[str, str]:
    """레코드에서 메타데이터 딕셔너리 생성.

    Args:
        row: DataFrame의 한 행
        index: 레코드 인덱스

    Returns:
        메타데이터 딕셔너리
    """
    category = extract_category(row.get("문제점 요약", ""))

    return {
        "id": str(index),
        "title": str(row.get("글 제목", "")),
        "source": str(row.get("출처", "")),
        "keywords": str(row.get("핵심 키워드", "")),
        "problem_summary": str(row.get("문제점 요약", "")),
        "category": category,
    }


def prepare_documents_for_vectorstore(
    df: pd.DataFrame,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """벡터 스토어용 문서와 메타데이터 준비.

    Args:
        df: 원본 DataFrame

    Returns:
        (texts, metadatas) 튜플
        - texts: 임베딩할 텍스트 리스트
        - metadatas: 각 텍스트에 대응하는 메타데이터 리스트
    """
    texts = []
    metadatas = []

    for idx, row in df.iterrows():
        text = combine_text_for_embedding(row)
        metadata = create_metadata(row, idx)

        texts.append(text)
        metadatas.append(metadata)

    print(f"✅ 문서 준비 완료: {len(texts)}개")
    return texts, metadatas


def get_category_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """카테고리별 레코드 수 집계.

    Args:
        df: 원본 DataFrame

    Returns:
        {카테고리: 개수} 딕셔너리
    """
    categories = df["문제점 요약"].apply(extract_category)
    return categories.value_counts().to_dict()


def print_data_stats(df: pd.DataFrame) -> None:
    """데이터 통계 출력."""
    print("\n" + "=" * 60)
    print("📊 데이터 통계")
    print("=" * 60)
    print(f"전체 레코드 수: {len(df)}")
    print(f"컬럼: {', '.join(df.columns)}")

    print("\n카테고리 분포:")
    category_dist = get_category_distribution(df)
    for category, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {category}: {count}개")

    print("\n샘플 레코드:")
    sample = df.iloc[0]
    print(f"  제목: {sample['글 제목']}")
    print(f"  키워드: {sample['핵심 키워드']}")
    print(f"  카테고리: {extract_category(sample['문제점 요약'])}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 테스트 실행
    df = load_csv_data()
    print_data_stats(df)

    # 문서 준비 테스트
    texts, metadatas = prepare_documents_for_vectorstore(df)

    print("\n샘플 임베딩 텍스트:")
    print(texts[0][:300] + "...")

    print("\n샘플 메타데이터:")
    print(metadatas[0])
