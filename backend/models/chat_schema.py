"""Pydantic schemas for chat - 프론트엔드 Message 타입 호환"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class MapMarker(BaseModel):
    """지도 마커 데이터"""
    name: str
    lat: float
    lng: float
    desc: Optional[str] = None


class MapData(BaseModel):
    """지도 데이터"""
    center: Dict[str, float]  # {"lat": float, "lng": float}
    markers: List[MapMarker]


class ChatRequest(BaseModel):
    """채팅 요청"""
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")


class ChatResponse(BaseModel):
    """채팅 응답 - 프론트엔드 Message 타입과 호환"""
    role: str = Field(default="ai", description="메시지 역할: user, ai")
    content: str = Field(..., description="메시지 내용")
    type: Optional[str] = Field(default="text", description="응답 타입: text, map")
    data: Optional[MapData] = Field(None, description="지도 데이터 (type=map일 때)")