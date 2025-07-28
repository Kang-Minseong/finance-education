import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")

print("✅ API 키 로딩 성공:", openai_api_key[:10] + "...")