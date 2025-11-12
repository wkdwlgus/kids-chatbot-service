# services/agent_service.py
"""
Kids Chatbot Agent Service

- LangGraph의 create_react_agent로 실제 에이전트 플로우를 만든다.
- 우리가 가진 weather / RAG / map 서비스를 LangChain Tool로 감싼다.
- 0개 RAG 결과일 때는 "조건에 맞는 시설을 찾지 못했다"라고만 말한다. (❗목데이터 금지)
- 위치는 에이전트 호출 전에 우리가 한번 정규화해서 힌트로 붙인다.
"""

import json
import re
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from utils.logger import logger
from services.llm_service import get_llm_service
from services.rag_service import get_rag_service
from services.weather_service import get_weather
from services.map_service import get_map_markers

# 위치 보정용 매핑들 (네가 만든 파일)
from data.location import (
    KMA_LOCATION_CODES,
    DONG_TO_CITY,
    LANDMARK_TO_CITY,
    UNIVERSITY_TO_CITY,
    LOCATION_MAP,
)


# ---------------------------------------------------------------------------
# 1. 위치 정규화 유틸
# ---------------------------------------------------------------------------

def normalize_location_from_text(text: str) -> Optional[str]:
    """
    사용자 입력에서 우리가 알고 있는 도시/동/명소를 최대한 찾아서
    대표 도시 이름(서울, 부산, 광주...)으로 리턴한다.
    LLM이 하게 하지 않고 여기서 1차 보정해주는 게 포인트.
    """
    raw = text.strip().replace(" ", "")

    # 1) 기상청 코드에 있는 도시 우선
    for city in KMA_LOCATION_CODES.keys():
        if city in raw:
            return city

    # 2) 동 → 도시
    for dong, city in DONG_TO_CITY.items():
        if dong.replace(" ", "") in raw:
            return city

    # 3) 랜드마크 → 도시
    for landmark, city in LANDMARK_TO_CITY.items():
        if landmark.replace(" ", "") in raw:
            return city

    # 4) 대학 → 도시
    for univ, city in UNIVERSITY_TO_CITY.items():
        if univ.replace(" ", "") in raw:
            return city

    # 5) 확장 맵
    for city, names in LOCATION_MAP.items():
        for name in names:
            if name.replace(" ", "") in raw:
                return city

    return None


# ---------------------------------------------------------------------------
# 2. Tool 정의
# ---------------------------------------------------------------------------

@tool
def weather_tool(location: str, date: str = "today") -> str:
    """
    날씨 정보를 조회합니다.
    - location: '서울', '부산 해운대' 같은 한국 지명
    - date: 'today', 'tomorrow', 'weekend' 같은 자연어를 LLM이 넣어줌
    과거 날짜거나 알 수 없는 날짜면 그 사실을 응답에 넣습니다.
    """
    logger.info(f"[weather_tool] location={location}, date={date}")

    # 아주 간단한 과거 날짜 방지 (YYYY-MM-DD가 들어오고 오늘보다 과거인 경우 등은
    # 여기서 더 빡세게 검사할 수 있음)
    # 지금은 서비스 단에서 과거 쿼리가 거의 안 온다는 전제로 간단 처리
    try:
        result = get_weather(location=location, target_date=date)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[weather_tool] error: {e}")
        return json.dumps(
            {
                "error": str(e),
                "message": "날씨 정보를 불러오지 못했어요. 나중에 다시 시도해 주세요.",
            },
            ensure_ascii=False,
        )


@tool
def rag_search_tool(query: str, location: str = "") -> str:
    """
    문화/체험/놀이 시설을 RAG로 검색합니다.
    - query: 사용자가 말한 원래 요청 (예: '광주 아이들 문화시설')
    - location: '광주', '서울'처럼 우리가 아는 대표 도시 이름 (없으면 빈 문자열)
    RAG 결과가 0개면 mock을 만들지 않고, 0개라는 정보를 그대로 돌려줍니다.
    """
    logger.info(f"[rag_search_tool] query={query}, location={location}")
    rag = get_rag_service()

    # location 있으면 메타 필터 만들어주기
    filters = None
    if location:
        # 우리 컬럼이 대문자였던 거 기억나지? (CTPRVN_NM, SIGNGU_NM ...)
        # 여기서는 시 단위까지만 거는 걸로
        filters = {"CTPRVN_NM": location}

    results = rag.search_and_rerank(
        query=query,
        top_k=5,
        filters=filters,
    )

    if not results:
        logger.warning("[rag_search_tool] 검색 결과 0개")
        return json.dumps(
            {
                "success": False,
                "message": f"'{location or '해당 지역'}'에서 조건에 맞는 시설을 찾지 못했어요. 다른 지역이나 키워드로 다시 말해줄래요?",
                "results": [],
            },
            ensure_ascii=False,
        )

    # 필요한 필드만 뽑아서 리턴
    simplified = []
    for doc in results:
        meta = doc.get("metadata", {})
        simplified.append(
            {
                "name": meta.get("Name") or meta.get("facility_name") or "이름 없음",
                "city": meta.get("CTPRVN_NM", ""),
                "district": meta.get("SIGNGU_NM", ""),
                "category1": meta.get("Category1", ""),
                "category2": meta.get("Category2", ""),
                "address": meta.get("Address", ""),
                "lat": meta.get("LAT", ""),
                "lon": meta.get("LON", ""),
            }
        )

    return json.dumps(
        {"success": True, "results": simplified},
        ensure_ascii=False,
    )


@tool
def map_tool(facilities_json: str) -> str:
    """
    RAG로 찾은 시설 리스트(JSON)를 받아서 지도 마커를 만든다.
    프론트에서 받아 그리기 쉽게 center, markers를 만들어준다.
    """
    logger.info("[map_tool] 호출")
    try:
        result = get_map_markers(facilities_json)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[map_tool] error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_tools():
    return [weather_tool, rag_search_tool, map_tool]


# ---------------------------------------------------------------------------
# 3. Agent 생성
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """너는 '아이랑 갈 곳 추천'을 도와주는 한국어 어시스턴트다.

규칙을 반드시 지켜라:

1. 사용자의 말에 지역이 없으면 먼저 "어느 지역을 생각하고 계세요?"라고 물어보고 대화를 멈춰라.
   - 단, 입력에 [LOCATION_HINT: XXX]가 있으면 그걸 지역으로 사용해라.
   - 한강 → 서울, 광안리 → 부산처럼 한국에서 흔한 명소는 도시로 간주해라.

2. 지역이 확인되면 먼저 weather_tool을 호출해서 그날 날씨를 확인해라.
   - 사용자가 날짜를 말하지 않으면 date="today"로 넣어라.
   - '내일', '모레', '이번 주말' 같은 표현은 date에 그대로 넣어도 된다.

3. 날씨를 확인한 뒤에는 rag_search_tool을 호출해서 시설을 찾아라.
   - rag_search_tool의 결과가 success=False 이면, 그 메시지를 사용자한테 그대로 전달하고
     "지역을 바꾸거나 키워드를 바꿔달라"고 제안해라.
   - 임의로 시설 이름을 만들어내지 마라.

4. 사용자가 '지도', '위치로 보여줘'라고 하면 방금 찾은 시설 결과를 map_tool에 넘겨라.

5. 감사/칭찬만 있는 입력에는 도구를 호출하지 말고 짧게 응답해라.

친절하고, 이모지를 가볍게 써라.
"""


def _load_llm_for_agent():
    """
    services.llm_service에 뭘 넣어놨는지 모르니 방어적으로 LLM을 꺼내는 함수.
    - OpenAI 있으면 그거
    - 아니면 llm_service 안에 있는 파이프라인
    - 그래도 없으면 아주 단순한 mock
    """
    llm_service = get_llm_service()

    # 1) llm_service가 chat 모델을 직접 주는 경우
    for attr in ["get_chat_model", "get_llm", "get_model"]:
        if hasattr(llm_service, attr):
            llm = getattr(llm_service, attr)()
            if llm is not None:
                return llm

    # 2) OpenAI 환경일 수도 있으니 한 번 더 시도
    try:
        import os
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.7)
    except Exception:
        pass

    # 3) 완전 mock
    class _Mock:
        def invoke(self, msgs):
            return AIMessage(content="Mock LLM 입니다. 실제 모델이 설정되지 않았어요.")
    return _Mock()


def create_agent():
    """LangGraph ReAct Agent를 만들어서 돌려줄 준비를 한다."""
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        logger.error("langgraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
        return None

    llm = _load_llm_for_agent()
    tools = get_tools()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        system_prompt=SUPERVISOR_PROMPT,
    )
    return agent


# ---------------------------------------------------------------------------
# 4. 히스토리 변환 & 실행
# ---------------------------------------------------------------------------

def _convert_history(history: List[Dict[str, str]]) -> List:
    """우리 서비스 포맷의 히스토리를 LangChain 메시지로 변환"""
    messages = []
    for h in history[-6:]:  # 최근 6개만
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
    return messages


def run_agent(
    user_query: str,
    conversation_id: str = "default",
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    실제로 콜랩/백엔드에서 부를 엔트리포인트.
    1) 위치 먼저 정규화해서 힌트로 붙이고
    2) 에이전트에 히스토리랑 같이 던지고
    3) 에이전트가 어떤 툴을 썼는지, 무슨 답을 했는지 되돌려준다.
    """
    conversation_history = conversation_history or []
    logger.info(f"🚀 Agent run: '{user_query}'")

    # 1) 위치 정규화 먼저
    normalized_location = normalize_location_from_text(user_query)
    if normalized_location:
        augmented_user_query = f"[LOCATION_HINT: {normalized_location}]\n{user_query}"
    else:
        augmented_user_query = user_query

    # 2) 에이전트 생성
    agent = create_agent()
    if agent is None:
        # agent 자체를 못 만들면 그냥 fallback
        fallback = "지금은 에이전트를 초기화할 수 없어요. 나중에 다시 시도해 주세요."
        return {
            "answer": fallback,
            "conversation_history": conversation_history + [
                {"role": "user", "content": user_query},
                {"role": "ai", "content": fallback},
            ],
            "tools_used": [],
        }

    # 3) 히스토리 변환
    lc_messages = _convert_history(conversation_history)
    lc_messages.append(HumanMessage(content=augmented_user_query))

    # 4) 실행
    try:
        result = agent.invoke({"messages": lc_messages})
    except Exception as e:
        logger.error(f"❌ Agent 실행 중 오류: {e}", exc_info=True)
        fallback = "요청을 처리하는 중에 오류가 발생했어요. 다른 표현으로 다시 말씀해 주실까요?"
        return {
            "answer": fallback,
            "conversation_history": conversation_history + [
                {"role": "user", "content": user_query},
                {"role": "ai", "content": fallback},
            ],
            "tools_used": [],
        }

    # 5) 결과에서 마지막 AI 메시지 찾기 + 어떤 툴을 썼는지 추출
    answer = "응답을 해석하지 못했어요."
    tools_used: List[str] = []
    messages = result.get("messages", [])

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            answer = msg.content
            break

    for msg in messages:
        # tool 호출은 AIMessage 안에 tool_calls로 들어오는 포맷이 많음
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name")
                if name:
                    tools_used.append(name)

    # 6) 히스토리 갱신
    new_history = conversation_history + [
        {"role": "user", "content": user_query},
        {"role": "ai", "content": answer},
    ]

    return {
        "answer": answer,
        "conversation_history": new_history,
        "tools_used": list(set(tools_used)),
    }