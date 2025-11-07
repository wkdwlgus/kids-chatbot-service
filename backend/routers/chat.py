"""
Chat Router

메인 챗봇 API - 프론트엔드 연동
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from models.chat_schema import ChatRequest, ChatResponse, MapData, MapMarker
from services.rag_service import get_rag_service
from services.llm_service import get_llm_service
from utils.logger import logger

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """
    메인 챗봇 엔드포인트 - 프론트엔드 Message 타입 호환
    
    TODO: LangGraph Agent 통합 예정
    - 멀티 에이전트 (RAG + Weather + Map)
    - 조건부 도구 호출
    - 멀티턴 대화 관리
    """
    try:
        logger.info(f"💬 챗봇 메시지: '{request.message}'")
        
        # TODO: LangGraph Agent 구현 예정
        # 현재는 기본 RAG만 사용
        
        # 1. RAG 검색
        rag_service = get_rag_service()
        search_results = rag_service.search_and_rerank(request.message)
        
        # 2. LLM 답변 생성
        llm_service = get_llm_service()
        answer = llm_service.generate_answer(request.message, search_results)
        
        # 3. 지도 데이터가 있는지 확인 (좌표 정보가 있으면)
        map_data = _create_map_data_if_needed(search_results)
        
        if map_data:
            # 지도 포함 응답
            return ChatResponse(
                role="ai",
                content=answer,
                type="map", 
                data=map_data
            )
        else:
            # 일반 텍스트 응답
            return ChatResponse(
                role="ai",
                content=answer,
                type="text"
            )
        
    except Exception as e:
        logger.error(f"챗봇 오류: {e}")
        return ChatResponse(
            role="ai",
            content="죄송합니다. 일시적인 오류가 발생했습니다.",
            type="text"
        )


def _create_map_data_if_needed(search_results: List[Dict[str, Any]]) -> Optional[MapData]:
    """검색 결과에서 지도 데이터 생성 (좌표가 있으면)"""
    if not search_results:
        return None
    
    # 좌표가 있는 결과들만 필터링
    locations = []
    for doc in search_results[:5]:  # 최대 5개
        meta = doc.get('metadata', {})
        lat = meta.get('latitude')
        lng = meta.get('longitude')
        name = meta.get('facility_name')
        
        if lat and lng and name:
            try:
                locations.append({
                    'name': str(name),
                    'lat': float(lat),
                    'lng': float(lng),
                    'desc': f"{meta.get('category1', '')} - {meta.get('category2', '')}"
                })
            except (ValueError, TypeError):
                continue
    
    if not locations:
        return None
    
    # 중심점 계산 (평균 좌표)
    center_lat = sum(loc['lat'] for loc in locations) / len(locations)
    center_lng = sum(loc['lng'] for loc in locations) / len(locations)
    
    # MapData 객체 생성
    markers = [MapMarker(**loc) for loc in locations]
    
    return MapData(
        center={"lat": center_lat, "lng": center_lng},
        markers=markers
    )


# TODO: LangGraph 통합 예정
# - Agent workflow 정의
# - State 관리
# - Tool 호출 순서 결정
# - 멀티턴 대화 처리