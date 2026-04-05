FROM python:3.11-slim

WORKDIR /app

# 先複製依賴清單（利用 Docker layer cache）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY main.py semantic_cache.py ./

# 預先下載 sentence-transformers 模型（避免首次請求冷啟動延遲）
# 此步驟約需 80MB 下載，但之後每次啟動容器無需重新下載
RUN python -c "from redisvl.utils.vectorize import HFTextVectorizer; HFTextVectorizer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
