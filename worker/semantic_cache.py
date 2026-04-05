"""
語義快取核心模組。

系統角色：
    Worker 接收 Go Gateway 轉發的請求後，進入此模組的查詢流程：
    1. 以 (system_prompt + query) 組合查詢鍵，生成 Embedding
    2. 在 Redis Stack 向量索引中做 ANN 近似最近鄰搜尋
    3. Cache Hit（距離 <= threshold）→ 直接回傳，省去 LLM API 呼叫費用
    4. Cache Miss → 呼叫 LLM → 結果寫入 RedisVL → 回傳

EvictionPolicy 擴充點：
    redisvl 目前以 TTL 做基本淘汰。
    未來透過 EvictionPolicy 介面實作 SphereLFU，
    使用核密度估計（KDE）評估查詢熱度分佈，
    淘汰低頻且邊緣的向量，保留高頻核心區域的快取條目。
"""
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

logger = logging.getLogger(__name__)


@dataclass
class CacheResult:
    """封裝單次查詢的完整結果，包含快取命中狀態與延遲指標。"""
    response: str
    is_cache_hit: bool
    latency_ms: float
    worker_id: str = ""
    metadata: dict = field(default_factory=dict)


class EvictionPolicy(ABC):
    """
    快取淘汰策略的抽象介面。

    子類需實作 should_evict() 和 record_access()。
    SemanticCacheManager 可呼叫此介面決定是否主動淘汰條目。
    """

    @abstractmethod
    def should_evict(self, key: str, access_count: int, last_access_ts: float) -> bool:
        """
        判斷指定快取條目是否應被淘汰。

        Args:
            key: 快取條目的向量鍵
            access_count: 該條目的歷史存取次數
            last_access_ts: 最後一次存取的 Unix timestamp

        Returns:
            True 表示應淘汰，False 表示應保留
        """
        ...

    @abstractmethod
    def record_access(self, key: str) -> None:
        """記錄一次存取，更新內部頻率統計。"""
        ...


class SphereLFUPolicy(EvictionPolicy):
    """
    SphereLFU（球形軟性頻率更新）淘汰策略。

    核心思想：
        傳統 LFU 使用精確計數，但快取中的向量並非孤立點，
        而是嵌入在高維空間中的語義叢集。
        SphereLFU 以「球形鄰域」為單位評估頻率：
        當查詢 q 命中快取條目 c 時，不只更新 c 的頻率，
        而是對所有與 c 的餘弦距離 <= radius 的鄰居也做衰減加權更新，
        使頻率估計更符合語義熱度分佈。

        淘汰時，使用核密度估計（KDE）計算每個條目在高維空間的
        局部密度，優先淘汰「密度低 + 頻率低」的邊緣條目。

    TODO（未來實作步驟）：
        1. 在 __init__ 中注入 redis client，用於讀取鄰居向量
        2. 實作 _gaussian_kernel(distance, bandwidth) 計算核函數權重
        3. 實作 _estimate_density(key) 用 redis VSIM 命令取得鄰居，
           加權累加頻率得到局部密度估計值
        4. should_evict() 中：若 density < low_density_threshold
           且 access_count < min_access_count，回傳 True
        5. record_access() 中：更新鄰域內所有向量的衰減頻率計數
    """

    def __init__(self, radius: float = 0.15, bandwidth: float = 0.05):
        self.radius = radius
        self.bandwidth = bandwidth

    def should_evict(self, key: str, access_count: int, last_access_ts: float) -> bool:
        """TODO: 實作基於核密度估計的淘汰決策。"""
        raise NotImplementedError(
            "SphereLFU.should_evict() not yet implemented. "
            "System falls back to redisvl TTL-based eviction."
        )

    def record_access(self, key: str) -> None:
        """TODO: 實作鄰域衰減頻率更新。"""
        raise NotImplementedError("SphereLFU.record_access() not yet implemented.")


class TTLOnlyPolicy(EvictionPolicy):
    """純 TTL 淘汰策略（預設使用）。完全委派給 redisvl TTL 機制。"""

    def should_evict(self, key: str, access_count: int, last_access_ts: float) -> bool:
        return False

    def record_access(self, key: str) -> None:
        pass


class SemanticCacheManager:
    """
    語義快取管理器：封裝 redisvl SemanticCache 的完整查詢流程。

    快取鍵設計：
        鍵 = f"{system_prompt}\\n\\n{user_query}"
        將 system_prompt 納入鍵，確保不同角色設定的查詢互相隔離。

    距離門檻說明：
        redisvl 使用餘弦距離（值域 [0, 2]），值越小越相似。
        distance_threshold=0.2 等同於餘弦相似度 >= 0.8。
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "llmcache",
        distance_threshold: float = 0.2,
        ttl: int = 3600,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        eviction_policy: Optional[EvictionPolicy] = None,
    ):
        self.eviction_policy = eviction_policy or TTLOnlyPolicy()

        vectorizer = HFTextVectorizer(model=model_name)

        self._cache = SemanticCache(
            name=index_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=distance_threshold,
            ttl=ttl,
        )

        logger.info(
            "SemanticCacheManager initialized",
            extra={
                "redis_url": redis_url,
                "distance_threshold": distance_threshold,
                "ttl": ttl,
                "model": model_name,
            },
        )

    def query(self, system_prompt: str, user_query: str) -> CacheResult:
        """
        執行語義快取查詢的主要入口。

        Args:
            system_prompt: LLM 的角色設定（用作路由鍵與快取命名空間）
            user_query: 使用者的自然語言查詢

        Returns:
            CacheResult 包含回應文字、是否命中快取、延遲毫秒數
        """
        start_ts = time.monotonic()

        # 組合快取查詢鍵：不同 system_prompt 的相同問題不互相污染快取
        cache_prompt = f"{system_prompt}\n\n{user_query}"

        hits = self._cache.check(prompt=cache_prompt, num_results=1)

        latency_ms = (time.monotonic() - start_ts) * 1000

        if hits:
            logger.info("cache hit", extra={"query_prefix": user_query[:50]})
            return CacheResult(
                response=hits[0]["response"],
                is_cache_hit=True,
                latency_ms=latency_ms,
                metadata=hits[0].get("metadata", {}),
            )

        logger.info("cache miss, calling LLM", extra={"query_prefix": user_query[:50]})
        llm_response = self._call_llm(
            system_prompt=system_prompt,
            user_query=user_query,
        )

        self._cache.store(
            prompt=cache_prompt,
            response=llm_response,
            metadata={"system_prompt_hash": str(hash(system_prompt))},
        )

        total_latency_ms = (time.monotonic() - start_ts) * 1000
        return CacheResult(
            response=llm_response,
            is_cache_hit=False,
            latency_ms=total_latency_ms,
        )

    def _call_llm(self, system_prompt: str, user_query: str) -> str:
        """
        呼叫 LLM API 的抽象方法（目前為 Dummy 實作）。

        替換為真實 OpenAI 呼叫時，修改此方法即可：
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
            )
            return response.choices[0].message.content
        """
        logger.warning("Using DUMMY LLM implementation.")
        time.sleep(0.05)
        return (
            f"[DUMMY LLM RESPONSE] "
            f"Query: '{user_query[:80]}' | "
            f"System: '{system_prompt[:40]}'"
        )
