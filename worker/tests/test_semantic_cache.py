"""
語義快取核心邏輯的單元測試。

測試策略：
- 使用 unittest.mock 模擬 redisvl SemanticCache，避免依賴真實 Redis
- 分別測試 Cache Hit / Cache Miss 路徑
- 測試 EvictionPolicy 介面的合約
"""
import pytest
from unittest.mock import MagicMock, patch
from semantic_cache import (
    SemanticCacheManager,
    EvictionPolicy,
    SphereLFUPolicy,
    CacheResult,
)


class TestEvictionPolicyInterface:
    """驗證 EvictionPolicy 是抽象介面，不可直接實例化"""

    def test_cannot_instantiate_abstract_policy(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            EvictionPolicy()

    def test_sphere_lfu_instantiates(self):
        """SphereLFU 可以實例化（即使功能是 stub）"""
        policy = SphereLFUPolicy()
        assert policy is not None

    def test_sphere_lfu_should_evict_raises_not_implemented(self):
        """SphereLFU.should_evict 尚未實作，呼叫應拋出 NotImplementedError"""
        policy = SphereLFUPolicy()
        with pytest.raises(NotImplementedError):
            policy.should_evict(key="test_key", access_count=5, last_access_ts=0.0)


class TestSemanticCacheManagerHit:
    """Cache Hit / Miss 路徑測試"""

    @patch("semantic_cache.SemanticCache")
    def test_cache_hit_returns_cached_response(self, MockSemanticCache):
        mock_cache_instance = MagicMock()
        MockSemanticCache.return_value = mock_cache_instance
        mock_cache_instance.check.return_value = [
            {"response": "Machine learning is a subset of AI.", "metadata": {}}
        ]

        manager = SemanticCacheManager(redis_url="redis://localhost:6379")

        result = manager.query(
            system_prompt="You are a helpful assistant.",
            user_query="What is machine learning?",
        )

        assert result.is_cache_hit is True
        assert result.response == "Machine learning is a subset of AI."
        assert result.latency_ms >= 0
        mock_cache_instance.check.assert_called_once()
        mock_cache_instance.store.assert_not_called()

    @patch("semantic_cache.SemanticCache")
    def test_cache_miss_calls_llm_and_stores(self, MockSemanticCache):
        mock_cache_instance = MagicMock()
        MockSemanticCache.return_value = mock_cache_instance
        mock_cache_instance.check.return_value = []

        manager = SemanticCacheManager(redis_url="redis://localhost:6379")

        with patch.object(manager, "_call_llm", return_value="Deep learning uses neural networks.") as mock_llm:
            result = manager.query(
                system_prompt="You are a helpful assistant.",
                user_query="Explain deep learning.",
            )

        assert result.is_cache_hit is False
        assert result.response == "Deep learning uses neural networks."
        mock_llm.assert_called_once_with(
            system_prompt="You are a helpful assistant.",
            user_query="Explain deep learning.",
        )
        mock_cache_instance.store.assert_called_once()

    @patch("semantic_cache.SemanticCache")
    def test_cache_key_combines_system_prompt_and_query(self, MockSemanticCache):
        """不同 system_prompt 的相同問題不應共用快取"""
        mock_cache_instance = MagicMock()
        MockSemanticCache.return_value = mock_cache_instance
        mock_cache_instance.check.return_value = []

        manager = SemanticCacheManager(redis_url="redis://localhost:6379")

        with patch.object(manager, "_call_llm", return_value="answer"):
            manager.query(system_prompt="You are a doctor.", user_query="What is aspirin?")
            manager.query(system_prompt="You are a chef.", user_query="What is aspirin?")

        assert mock_cache_instance.check.call_count == 2
        calls = mock_cache_instance.check.call_args_list
        prompt_0 = calls[0][1].get("prompt") or calls[0][0][0]
        prompt_1 = calls[1][1].get("prompt") or calls[1][0][0]
        assert prompt_0 != prompt_1
