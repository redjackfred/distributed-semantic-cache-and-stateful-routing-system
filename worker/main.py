"""
Python Worker FastAPI 入口。

Worker 在 Polyglot 架構中的角色：
    - 接收 Go Gateway 透過一致性雜湊路由過來的請求
    - 相同 system_prompt 的請求持續路由到同一台 Worker（一致性雜湊穩定性），
      使本地 Redis 快取效益最大化
    - 每台 Worker 以 WORKER_ID 環境變數識別（Docker Compose 注入）
"""
import logging
import os
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from semantic_cache import SemanticCacheManager

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)

WORKER_ID = os.getenv("WORKER_ID", socket.gethostname())
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

cache_manager: SemanticCacheManager = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化快取管理器（應用程式啟動時執行一次）。"""
    global cache_manager
    logger.info(f"Worker {WORKER_ID} starting, connecting to Redis: {REDIS_URL}")
    try:
        cache_manager = SemanticCacheManager(
            redis_url=REDIS_URL,
            distance_threshold=float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.2")),
            ttl=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
        )
        logger.info(f"Worker {WORKER_ID} ready")
    except Exception as e:
        logger.warning(f"Cache manager init failed (will retry on first request): {e}")
        cache_manager = None
    yield
    logger.info(f"Worker {WORKER_ID} shutting down")


app = FastAPI(
    title="Semantic Cache Worker",
    description="語義向量快取 Worker 節點",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """來自 Go Gateway 的查詢請求格式。"""
    system_prompt: str = ""
    query: str
    user_id: str = ""


class QueryResponse(BaseModel):
    """回傳給 Go Gateway 的查詢結果格式。"""
    response: str
    cache_hit: bool
    latency_ms: float
    worker_id: str


@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest) -> QueryResponse:
    """
    主要查詢端點。

    Go Gateway 透過 ReverseProxy 轉發請求到此端點，
    Worker 做向量搜尋後回傳結果，Gateway 直接將結果傳給使用者。
    """
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not available")
    try:
        result = cache_manager.query(
            system_prompt=req.system_prompt,
            user_query=req.query,
        )
        return QueryResponse(
            response=result.response,
            cache_hit=result.is_cache_hit,
            latency_ms=result.latency_ms,
            worker_id=WORKER_ID,
        )
    except Exception as e:
        logger.error(f"query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Worker internal error: {str(e)}")


@app.get("/health")
async def health_check():
    """健康檢查端點。Docker Compose healthcheck 與 Go Gateway 監控使用。"""
    return {
        "status": "ok",
        "worker_id": WORKER_ID,
        "redis_url": REDIS_URL,
    }
