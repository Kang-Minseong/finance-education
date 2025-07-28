import os
import pickle
from dotenv import load_dotenv
from pathlib import Path
from typing import List
import numpy as np
import faiss
from openai import OpenAI

# 🔑 API 키 로딩
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 📄 문서 불러오기
def load_documents(folder_path: str) -> List[str]:
    docs = []
    for file_path in Path(folder_path).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

# 🧠 임베딩 생성
def get_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

# ✅ 실행
if __name__ == "__main__":
    print("📄 문서 불러오는 중...")
    documents = load_documents("docs")
    print(f"총 {len(documents)}개의 문서 로드 완료.")

    print("🧠 임베딩 생성 중...")
    vectors = get_embeddings(documents)

    print("📦 FAISS 인덱스 생성 중...")
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    print("💾 인덱스 저장 중...")
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump((index, documents), f)  # ✅ 튜플 형태로 저장해야 나중에 index, documents로 분리됨

    print("✅ 저장 완료!")
