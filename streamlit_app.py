import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import pickle
import numpy as np
import time
from pathlib import Path

# --- 환경설정 및 API 클라이언트 생성 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 벡터 DB와 문서 로드 함수 ---
@st.cache_resource(show_spinner=False)
def load_vector_db():
    try:
        with open("vectors.pkl", "rb") as f:
            data = pickle.load(f)
        texts = data["texts"]
        vectors = data["vectors"]
        # faiss는 read_index가 아니라 pickle로 로드
        with open("faiss_index.pkl", "rb") as f_idx:
            index, _ = pickle.load(f_idx)
        return texts, vectors, index
    except Exception as e:
        st.error(f"벡터 DB 로드 실패: {e}")
        return None, None, None

texts, vectors, index = load_vector_db()

# --- 질문 답변 함수 ---
def ask_question(query: str, level: str = "초급", top_k: int = 3) -> str:
    if not index:
        return "벡터 DB가 로드되지 않았습니다. 관리자에게 문의하세요."

    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_vector = np.array(embedding_response.data[0].embedding).astype("float32")
    except Exception as e:
        return f"임베딩 생성 중 오류 발생: {e}"

    D, I = index.search(np.array([query_vector]), top_k)
    context = "\n\n".join([texts[i] for i in I[0]])

    prompt = f"""
당신은 친절하고 전문적인 증권 금융 교육 챗봇입니다.

아래는 관련 학습 자료입니다:
{context}

금융 지식 수준: {level}

사용자 질문: {query}

이 질문에 대해 {level} 수준에 맞게 쉽고 명확하게 설명해주세요.
필요하면 예시도 포함하세요.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 금융 교육 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        answer = f"답변 생성 중 오류가 발생했습니다: {e}"

    return answer

# --- Streamlit UI ---

st.set_page_config(
    page_title="💰 대학생 증권 금융 교육 챗봇",
    page_icon="💹",
    layout="wide"
)

# CSS 스타일 적용
st.markdown("""
<style>
    .title-font {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0B3D91;
        margin-bottom: 10px;
    }
    .sub-font {
        font-size: 16px !important;
        color: #555555;
        margin-bottom: 20px;
    }
    .chat-box {
        background-color: #f0f4f8;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    .chat-question {
        font-weight: 600;
        color: #1565c0;
        margin-bottom: 8px;
    }
    .chat-answer {
        white-space: pre-wrap;
        font-size: 15px;
        color: #333333;
    }
    hr {
        margin-top: 30px;
        margin-bottom: 30px;
        border: none;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# 사이드바: 사용자 프로필
with st.sidebar:
    st.header("👤 사용자 정보 입력")
    user_level = st.selectbox("금융 지식 수준 선택", ["초급", "중급", "고급"])
    user_grade = st.selectbox("학년 선택", ["1학년", "2학년", "3학년", "4학년", "기타"])
    st.markdown("---")
    st.markdown("본 챗봇은 증권 금융 교육 목적으로 제작되었습니다.")

# 메인 영역 제목 및 설명
st.markdown('<div class="title-font">대학생 증권 금융 교육 챗봇 (RAG 기반)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-font">궁금한 내용을 입력하면 금융 지식 수준에 맞춰 친절하고 명확한 답변을 제공합니다.</div>', unsafe_allow_html=True)

# 2컬럼 레이아웃: 질문 입력 / 대화 기록
col1, col2 = st.columns([3, 2])

with col1:
    query = st.text_area("❓ 질문을 입력하세요", placeholder="예: ETF가 뭐예요?", height=130)
    if st.button("질문 보내기"):
        if not query.strip():
            st.warning("먼저 질문을 입력해주세요.")
        elif not texts or not index:
            st.error("시스템 준비가 완료되지 않았습니다. 잠시 후 다시 시도해주세요.")
        else:
            with st.spinner("답변 생성 중... 잠시만 기다려주세요."):
                answer = ask_question(query, user_level)
                time.sleep(0.5)
                # 대화 기록 저장
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({"질문": query, "답변": answer})
                # 입력창 비우기 (UI 처리)
                st.experimental_rerun()

with col2:
    st.markdown('<div class="title-font">🗒️ 대화 기록</div>', unsafe_allow_html=True)
    if "history" in st.session_state and st.session_state.history:
        for i, chat in enumerate(reversed(st.session_state.history)):
            st.markdown(f'''
                <div class="chat-box">
                    <div class="chat-question">Q{i+1}. {chat['질문']}</div>
                    <div class="chat-answer">{chat['답변']}</div>
                </div>
            ''', unsafe_allow_html=True)
    else:
        st.info("아직 질문이 없습니다. 왼쪽에 질문을 입력해보세요!")

# 푸터
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="font-size:12px; color:#999;">ⓒ 2025 증권교육 AI 챗봇 프로젝트</p>', unsafe_allow_html=True)
