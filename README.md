# Distributed Semantic Cache & Stateful Routing System

## The Problem

Every LLM API call costs money and adds latency. In production, a large share of those calls are **semantically redundant** — users ask the same question in slightly different words, and the system calls the LLM every single time.

Standard caching doesn't help. Exact-match key comparison misses "What is ML?" vs "Can you explain machine learning?" entirely. And even semantic caches break down at scale: when requests scatter randomly across multiple workers, no single worker accumulates enough cache history to be effective.

## What This Solves

| Pain Point | This System |
|------------|-------------|
| Paying for the same LLM answer repeatedly | Semantic vector cache returns stored responses for similar queries — no LLM call needed |
| Exact-match caches miss paraphrased questions | ANN search on 384-dim embeddings catches semantically equivalent queries (cosine similarity ≥ 0.8) |
| Cache hit rate collapses when scaled horizontally | Consistent hashing pins the same `system_prompt` context to the same worker — cache stays warm as you scale |
| Cold-start latency on first request | Embedding model is pre-loaded at container startup, not on first query |
| Vendor lock-in to one LLM provider | Swap the `_call_llm()` method — routing and caching logic is provider-agnostic |

## Key Strengths

- **7× latency reduction on cache hits** — measured ~180ms (LLM call) vs ~25ms (cache hit) in local testing
- **Stateful routing without sticky sessions** — consistent hashing achieves worker affinity without infrastructure-level session pinning
- **Graceful horizontal scaling** — adding a worker migrates only ~1/N keys (consistent hashing), avoiding cache thrashing that plagues modulo-based routing
- **Semantic namespace isolation** — `system_prompt` is part of the cache key, so a customer service bot and a medical advisor never share cached answers to the same question
- **Extensible eviction** — `EvictionPolicy` interface ships with a `SphereLFU` stub for future kernel density estimation-based eviction, replacing blunt TTL expiry

---

A polyglot architecture that combines a high-performance Go API Gateway with Python semantic cache workers to minimize redundant LLM API calls. Requests with the same context are pinned to the same worker via consistent hashing, maximizing vector cache hit rates.

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────┐
│     Go API Gateway :8080    │
│                             │
│  1. Parse system_prompt     │
│  2. Consistent Hash Ring    │  ◄── KV-Cache Indexer (extensible)
│  3. Select target Worker    │
│  4. Reverse Proxy forward   │
└─────────────────────────────┘
          │              │
          ▼              ▼
┌──────────────┐  ┌──────────────┐
│   Worker 1   │  │   Worker 2   │
│  :8001       │  │  :8001       │
│              │  │              │
│ sentence-    │  │ sentence-    │
│ transformers │  │ transformers │
│ + redisvl    │  │ + redisvl    │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
     ┌─────────────────────┐
     │   Redis Stack :6379  │
     │                      │
     │  HNSW Vector Index   │
     │  ANN Search + TTL    │
     └─────────────────────┘
```

**Routing key:** The `system_prompt` field is hashed via a consistent hash ring (FNV-1a, 150 virtual nodes per worker). Requests with the same system prompt always route to the same worker, so that worker's local semantic cache remains effective across repeated queries.

**Cache logic:** On each query, the worker generates an embedding with `sentence-transformers/all-MiniLM-L6-v2` and searches Redis Stack for semantically similar past responses (cosine distance ≤ 0.2, equivalent to similarity ≥ 0.8). Cache hits skip the LLM entirely; misses call the LLM, store the result with a TTL, and return.

## Tech Stack

| Layer | Technology |
|-------|------------|
| API Gateway | Go 1.22 + Gin |
| Semantic Cache Worker | Python 3.11 + FastAPI + uvicorn |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim) |
| Vector Cache | redisvl 0.16 + Redis Stack 7.2 (HNSW index) |
| Orchestration | Docker Compose v2 |

## Prerequisites

- Docker with Compose v2 (`docker compose version`)
- 4 GB RAM recommended (sentence-transformers model: ~80 MB per worker)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/redjackfred/distributed-semantic-cache-and-stateful-routing-system.git
cd distributed-semantic-cache-and-stateful-routing-system

# 2. Start all services (first run downloads ~1 GB of images + model weights)
docker compose up --build -d

# 3. Wait for all services to become healthy (~60-90 seconds)
docker compose ps

# 4. Send a query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "You are a helpful assistant.",
    "query": "What is machine learning?"
  }'
```

## API

### `POST /query`

The only endpoint you need. All traffic goes through the Gateway.

**Request:**
```json
{
  "system_prompt": "You are a helpful assistant.",
  "query": "What is machine learning?",
  "user_id": "optional-tracking-id"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `system_prompt` | string | No | Routing key — requests with identical `system_prompt` always route to the same worker |
| `query` | string | **Yes** | The natural language question |
| `user_id` | string | No | Audit/tracking only, does not affect routing |

**Response:**
```json
{
  "response": "Machine learning is a subset of AI...",
  "cache_hit": false,
  "latency_ms": 182.4,
  "worker_id": "worker1"
}
```

**Cache hit (second semantically similar query):**
```json
{
  "response": "Machine learning is a subset of AI...",
  "cache_hit": true,
  "latency_ms": 24.7,
  "worker_id": "worker1"
}
```

### `GET /health`

Returns gateway status and the current worker list.

```json
{
  "status": "ok",
  "workers": ["http://worker1:8001", "http://worker2:8001"]
}
```

## Connecting a Real LLM

The worker ships with a dummy LLM implementation. To use OpenAI:

**1. Edit `worker/semantic_cache.py`:**

```python
def _call_llm(self, system_prompt: str, user_query: str) -> str:
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
```

**2. Add the API key to `docker-compose.yml`:**

```yaml
worker1:
  environment:
    - OPENAI_API_KEY=sk-...
worker2:
  environment:
    - OPENAI_API_KEY=sk-...
```

**3. Rebuild workers:**

```bash
docker compose up -d --build worker1 worker2
```

## Scaling Workers

Add a third worker by extending `docker-compose.yml`:

```yaml
worker3:
  build:
    context: ./worker
    dockerfile: Dockerfile
  environment:
    - WORKER_ID=worker3
    - REDIS_URL=redis://redis:6379
  depends_on:
    redis:
      condition: service_healthy

gateway:
  environment:
    - WORKERS=http://worker1:8001,http://worker2:8001,http://worker3:8001
```

The consistent hash ring redistributes routing automatically. Because of consistent hashing, only ~1/N keys migrate when adding the Nth worker — cache thrashing is minimized.

## Cache Similarity Threshold

The default cosine distance threshold is `0.2` (equivalent to cosine similarity ≥ 0.8). Adjust in `config/config.yaml` or via the `CACHE_DISTANCE_THRESHOLD` environment variable per worker:

| `distance_threshold` | Cosine similarity | Behavior |
|----------------------|-------------------|----------|
| `0.1` | ≥ 0.9 | Strict — only near-identical queries hit cache |
| `0.2` | ≥ 0.8 | **Default** — balanced precision and recall |
| `0.3` | ≥ 0.7 | Loose — broader semantic matches hit cache |

## Monitoring

```bash
# Gateway routing logs (shows which worker each request goes to)
docker compose logs -f gateway

# Worker cache hit/miss logs
docker compose logs -f worker1 worker2

# RedisInsight — visualise the vector index in a browser
open http://localhost:8001
```

## Running Tests

```bash
# Go unit tests (consistent hash ring)
cd gateway && go test ./... -v

# Python unit tests (semantic cache logic + FastAPI endpoints)
cd worker && python3 -m pytest tests/ -v

# End-to-end validation (requires docker compose up)
bash scripts/e2e_test.sh
```

## Project Structure

```
.
├── gateway/                    # Go API Gateway
│   ├── main.go                 # Server entry point, DI wiring
│   ├── go.mod
│   ├── router/
│   │   ├── consistent_hash.go  # FNV-1a hash ring, virtual nodes
│   │   ├── consistent_hash_test.go
│   │   └── proxy.go            # Request parsing + reverse proxy
│   ├── indexer/
│   │   └── indexer.go          # KV-Cache Indexer interface (extensible)
│   └── Dockerfile
├── worker/                     # Python Semantic Cache Worker
│   ├── main.py                 # FastAPI app, /query + /health
│   ├── semantic_cache.py       # SemanticCacheManager, EvictionPolicy
│   ├── requirements.txt
│   ├── tests/
│   │   ├── test_main.py
│   │   └── test_semantic_cache.py
│   └── Dockerfile
├── config/
│   └── config.yaml             # Shared configuration
├── scripts/
│   └── e2e_test.sh             # End-to-end validation script
└── docker-compose.yml
```

## Extending the System

### KV-Cache Indexer (cache-aware routing)

`gateway/indexer/indexer.go` defines the `Indexer` interface. The default `NoopIndexer` falls back entirely to consistent hashing. Implement `RedisIndexer` to have workers register their cached prefixes with the Gateway, enabling routing decisions based on actual cache contents rather than hash position.

### SphereLFU Eviction Policy

`worker/semantic_cache.py` contains a `SphereLFUPolicy` stub. The intended algorithm uses kernel density estimation across the embedding space — evicting low-frequency vectors at the periphery of semantic clusters while preserving high-frequency core entries. See the docstring in `SphereLFUPolicy` for the implementation roadmap.

## Stopping

```bash
docker compose down       # Stop services, keep Redis data
docker compose down -v    # Stop services and delete all cached data
```
