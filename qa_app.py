import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# 🔑 API 로딩
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 📦 벡터 DB 및 문서 로드
with open("faiss_index.pkl", "rb") as f:
    index, documents = pickle.load(f)

# 📚 질문 유형 감지 함수
def detect_question_type(query: str) -> str:
    if "차이" in query or "비교" in query:
        return "비교"
    elif "계산" in query or "구하는" in query:
        return "계산"
    elif "예시" in query:
        return "예시"
    else:
        return "정의"

# 🧠 GPT 응답 함수
def ask_question(query, level="중"):
    # 1. 질문 임베딩
    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_vector = np.array(embedding_response.data[0].embedding).astype("float32")

    # 2. FAISS 검색
    D, I = index.search(np.array([query_vector]), k=3)
    context = "\n\n".join([documents[i] for i in I[0]])

    # 3. 질문 유형 분류
    q_type = detect_question_type(query)

    # 4. 레벨별 말투 및 안내 문구
    style_guide = {
        "초": "아주 쉽게 풀어 설명하고, 일상적인 예시를 포함하세요.",
        "중": "쉬운 말로 설명하고, 필요 시 사례나 요약을 제공하세요.",
        "고": "전문 용어와 배경 지식을 포함해 자세히 설명하세요."
    }

    # 5. 프롬프트 구성
    prompt = f"""
당신은 대학생 대상의 증권 금융 교육 챗봇입니다.

📘 [사용자 정보]
- 지식 수준: {level}
- 질문 유형: {q_type}
- 질문: {query}

📚 [참고 학습 자료]
{context}

💬 [답변 가이드]
- {q_type} 유형의 질문에 맞춰 설명
- {style_guide.get(level, '쉬운 말로 설명하세요.')}
- 핵심 개념 → 예시/사례 → 핵심 요약 순으로 설명
- 너무 어렵게 설명하지 말고 친절하게 말해줘
"""

    # 6. GPT 호출
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 금융교육 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content.strip()

# 🚀 실행
if __name__ == "__main__":
    print("📘 증권 교육 챗봇입니다.")
    level = input("👤 금융 지식 수준을 입력해주세요 (초/중/고) >>> ").strip()
    while True:
        q = input("\n❓ 질문을 입력하세요 (종료하려면 'exit') >>> ").strip()
        if q.lower() == "exit":
            break
        answer = ask_question(q, level=level)
        print("\n💡 GPT 응답:\n", answer)
