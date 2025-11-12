"""
init_chroma_colab.py
----------------------------------------
Colab 환경에서도 작동하는 안전형 데이터 초기화 스크립트.
- description 컬럼이 없으면 자동 생성
- GPU/CPU 환경 자동 감지
- 배치 단위 업로드 + 로깅 + GPU 메모리 표시
"""

import os
import sys
import pandas as pd
from pathlib import Path
import torch

# backend 경로 인식
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.vector_client import get_vector_client, reset_vector_client
from utils.logger import logger


# ============================================================
# 1️⃣ description 자동 생성 함수
# ============================================================

def safe_get(row, key):
    val = row.get(key, "")
    if pd.isna(val) or str(val).strip() in ["", "nan", "None"]:
        return ""
    return str(val).strip()


def build_description(row):
    """시설 정보를 자연어 설명으로 변환"""
    parts = []
    name = safe_get(row, "Name")
    region = " ".join(filter(None, [safe_get(row, "CTPRVN_NM"), safe_get(row, "SIGNGU_NM")]))
    category = " ".join(filter(None, [safe_get(row, "Category1"), safe_get(row, "Category2"), safe_get(row, "Category3")]))
    if name:
        parts.append(f"{name}은(는)")
    if region:
        parts.append(f"{region}에 위치한")
    if category:
        parts.append(f"{category} 관련 시설입니다.")
    else:
        parts.append("가족 및 유아가 함께 즐길 수 있는 시설입니다.")
    inout = safe_get(row, "in_out")
    cost = safe_get(row, "Cost")
    age = safe_get(row, "Age")
    time = safe_get(row, "Time")
    day = safe_get(row, "Day")
    note = safe_get(row, "Note")
    address = safe_get(row, "Address")
    if inout:
        parts.append(f"이 시설은 {inout} 장소이며,")
    if age:
        parts.append(f"이용 연령대는 {age}입니다.")
    if cost:
        parts.append(f"이용 요금은 {cost}입니다.")
    if time or day:
        parts.append(f"운영 시간은 {day} {time}입니다.")
    if address:
        parts.append(f"주소는 {address}입니다.")
    if note:
        parts.append(note)
    return " ".join(parts).replace("  ", " ").strip()


# ============================================================
# 2️⃣ CSV → ChromaDB 업로드 함수
# ============================================================

def load_csv_to_chroma(csv_path: str, batch_size: int = 100):
    try:
        logger.info(f"📂 CSV 파일 로딩: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"✅ {len(df)}개 행 로드 완료")

        # description 자동 생성
        if "description" not in df.columns:
            logger.warning("⚠️ description 컬럼 없음 → 자동 생성 중...")
            df["description"] = df.apply(build_description, axis=1)

        # facility_name 컬럼 보정
        if "facility_name" not in df.columns and "Name" in df.columns:
            df["facility_name"] = df["Name"]

        # 정제
        df = df.dropna(subset=["description"])
        df = df[df["description"].str.strip() != ""]
        logger.info(f"🧹 정제 후 {len(df)}개 행 남음")

        # VectorClient 초기화
        reset_vector_client()
        client = get_vector_client()
        info = client.get_collection_info()
        logger.info(f"📊 현재 환경: {info['environment']} | 모델: {info['embedding_model']}")

        # 배치 업로드
        total_batches = (len(df) + batch_size - 1) // batch_size
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(f"⏳ 배치 {batch_num}/{total_batches} 처리 중...")
            documents = batch_df["description"].tolist()
            metadatas = batch_df.to_dict("records")
            ids = [f"facility_{idx}" for idx in batch_df.index]
            client.add_documents(documents, metadatas, ids)
            logger.info(f"✅ 배치 {batch_num} 완료 ({len(documents)}개 문서)")

        # 완료 로그
        info = client.get_collection_info()
        logger.info("🎉 데이터 로드 완료!")
        logger.info(f"📚 총 문서 수: {info['count']}개 | 환경: {info['environment']}")
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1024**3
            resv = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"💾 GPU 메모리 사용량: {alloc:.2f}GB / {resv:.2f}GB")

    except Exception as e:
        logger.error(f"❌ 데이터 로드 실패: {e}")
        import traceback; traceback.print_exc()


# ============================================================
# 3️⃣ 메인 실행
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="CSV 파일 경로")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    load_csv_to_chroma(args.csv_path, args.batch_size)