"""FastAPI Worker 端點整合測試。"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app
from semantic_cache import CacheResult


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "worker_id" in data


class TestQueryEndpoint:
    def test_query_returns_cache_hit_response(self, client):
        mock_result = CacheResult(
            response="Paris is the capital of France.",
            is_cache_hit=True,
            latency_ms=5.2,
        )
        with patch("main.cache_manager") as mock_manager:
            mock_manager.query.return_value = mock_result
            response = client.post(
                "/query",
                json={"system_prompt": "You are a geography expert.", "query": "Capital of France?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Paris is the capital of France."
        assert data["cache_hit"] is True
        assert data["latency_ms"] == pytest.approx(5.2, abs=0.1)

    def test_query_missing_query_field_returns_422(self, client):
        response = client.post(
            "/query",
            json={"system_prompt": "You are helpful."},
        )
        assert response.status_code == 422

    def test_query_empty_system_prompt_is_allowed(self, client):
        mock_result = CacheResult(response="Hello!", is_cache_hit=False, latency_ms=100.0)
        with patch("main.cache_manager") as mock_manager:
            mock_manager.query.return_value = mock_result
            response = client.post(
                "/query",
                json={"system_prompt": "", "query": "Hello"},
            )
        assert response.status_code == 200
