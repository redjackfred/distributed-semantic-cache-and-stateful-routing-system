# Distributed Semantic Cache & Stateful Routing System — 實作計劃

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一個 Polyglot 混合架構系統：Go API Gateway 透過一致性雜湊將相同語境的請求穩定路由至同一台 Python Worker，Worker 以 RedisVL 語義向量快取避免重複呼叫 LLM API。

**Architecture:** Go Gateway 接收 JSON 請求，對 `system_prompt` 欄位做一致性雜湊決定目標 Worker，再以 `httputil.ReverseProxy` 反向代理轉發；Python Worker 使用 `sentence-transformers` 生成 Embedding，透過 `redisvl.SemanticCache` 做 ANN 搜尋，Cache Hit 直接回傳，Cache Miss 才呼叫 LLM 並寫入快取。KV-Cache Indexer 介面預留未來「快取感知路由」的擴充點。

**Tech Stack:** Go 1.22 + Gin、Python 3.11 + FastAPI + sentence-transformers + redisvl 0.3+、Redis Stack 7.2、Docker Compose v2

---

## 檔案結構

```
semantic-cache-router/
├── gateway/                          # Go API Gateway（控制層 + 路由層）
│   ├── go.mod                        # Go module 定義
│   ├── main.go                       # HTTP 伺服器啟動、路由初始化、DI 組裝
│   ├── router/
│   │   ├── consistent_hash.go        # 一致性雜湊環：Worker 節點管理 + key→node 映射
│   │   ├── consistent_hash_test.go   # 單元測試：分佈均勻性、節點增刪穩定性
│   │   └── proxy.go                  # 反向代理：解析請求體取路由鍵、呼叫雜湊環、轉發
│   └── indexer/
│       └── indexer.go                # KV-Cache Indexer 介面 + NoopIndexer（預設空實作）
├── worker/                           # Python Semantic Cache Worker（語義層）
│   ├── requirements.txt              # Python 依賴清單
│   ├── main.py                       # FastAPI 入口：/query 端點、健康檢查
│   ├── semantic_cache.py             # 核心：EvictionPolicy 抽象 + SemanticCacheManager
│   └── tests/
│       ├── test_semantic_cache.py    # 單元測試：cache hit/miss 邏輯、eviction policy 介面
│       └── test_main.py              # 整合測試：FastAPI 端點 E2E
├── config/
│   └── config.yaml                   # 共用設定：Workers URL 清單、Redis URL、快取閾值
└── docker-compose.yml                # 完整容器編排：Redis Stack + Gateway + 2x Worker
```

**架構互動關係：**
```
使用者請求
    │
    ▼
[Go Gateway :8080]
    │  1. 解析 JSON 取 system_prompt
    │  2. Hash(system_prompt) → 查詢一致性雜湊環
    │  3. 取得目標 Worker URL
    │  4. ReverseProxy 轉發
    ▼
[Python Worker :8001 or :8002]
    │  5. 生成 Embedding (sentence-transformers)
    │  6. ANN 搜尋 Redis Stack
    │  ├─ Cache Hit → 直接回傳
    │  └─ Cache Miss → 呼叫 LLM → 寫入 RedisVL → 回傳
    ▼
[Redis Stack :6379]
    向量索引 (HNSW) + TTL 管理
```

---

## Task 1: 專案骨架與共用設定

**Files:**
- Create: `gateway/go.mod`
- Create: `worker/requirements.txt`
- Create: `config/config.yaml`

- [ ] **Step 1: 建立目錄結構**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
mkdir -p gateway/router gateway/indexer worker/tests config docs/superpowers/plans
```

Expected: 無報錯，`ls` 可見上述目錄。

- [ ] **Step 2: 建立 `gateway/go.mod`**

```
module semantic-cache-router/gateway

go 1.22

require (
    github.com/gin-gonic/gin v1.10.0
)
```

- [ ] **Step 3: 建立 `worker/requirements.txt`**

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
sentence-transformers==3.0.1
redisvl==0.3.0
pydantic==2.7.1
httpx==0.27.0
pytest==8.2.2
pytest-asyncio==0.23.7
```

- [ ] **Step 4: 建立 `config/config.yaml`**

```yaml
gateway:
  port: 8080
  # 一致性雜湊虛擬節點數：越大分佈越均勻，但記憶體消耗略增
  hash_replicas: 150
  # 路由鍵欄位：用哪個欄位決定路由（system_prompt 或 prefix）
  routing_key_field: "system_prompt"

workers:
  # Docker Compose 服務名稱解析
  - "http://worker1:8001"
  - "http://worker2:8001"

redis:
  url: "redis://redis:6379"
  index_name: "llmcache"

cache:
  # 語義相似度門檻：向量餘弦距離 <= 0.2 等同於相似度 >= 0.8
  # redisvl 使用距離（越小越相似），此處設 0.2 對應使用者規格的「0.8 相似度」
  distance_threshold: 0.2
  ttl_seconds: 3600
```

- [ ] **Step 5: 初始化 Git 並提交骨架**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
git init
echo "gateway/vendor/\nworker/__pycache__/\nworker/.venv/\n*.pyc\n.DS_Store" > .gitignore
git add .
git commit -m "chore: initialise monorepo skeleton"
```

Expected: `main branch` 建立，首次 commit 成功。

---

## Task 2: Go — 一致性雜湊環

**Files:**
- Create: `gateway/router/consistent_hash.go`
- Create: `gateway/router/consistent_hash_test.go`

**設計決策：**
- 使用 FNV-1a 32-bit hash（零依賴、速度快）
- 每個實體節點擴展為 `replicas` 個虛擬節點，避免資料傾斜
- 以排序 `[]uint32` + `sort.Search` 實作 O(log n) 查詢

- [ ] **Step 1: 撰寫失敗測試 `gateway/router/consistent_hash_test.go`**

```go
package router

import (
	"fmt"
	"testing"
)

// TestAddAndGet 驗證：加入節點後，同一個 key 每次都路由到同一台 Worker
func TestAddAndGet(t *testing.T) {
	ring := NewConsistentHash(50)
	ring.Add("http://worker1:8001")
	ring.Add("http://worker2:8001")

	key := "You are a helpful assistant."
	first := ring.Get(key)
	if first == "" {
		t.Fatal("Get() should not return empty string")
	}
	for i := 0; i < 100; i++ {
		if got := ring.Get(key); got != first {
			t.Errorf("iteration %d: expected %s, got %s", i, first, got)
		}
	}
}

// TestEmptyRing 驗證：空雜湊環不應 panic
func TestEmptyRing(t *testing.T) {
	ring := NewConsistentHash(50)
	if got := ring.Get("any key"); got != "" {
		t.Errorf("empty ring should return empty string, got %s", got)
	}
}

// TestRemoveReducesNodes 驗證：移除節點後，其他 key 仍能路由（不 panic）
func TestRemoveReducesNodes(t *testing.T) {
	ring := NewConsistentHash(50)
	ring.Add("http://worker1:8001")
	ring.Add("http://worker2:8001")
	ring.Remove("http://worker1:8001")

	for i := 0; i < 50; i++ {
		key := fmt.Sprintf("query-%d", i)
		got := ring.Get(key)
		if got != "http://worker2:8001" {
			t.Errorf("after removing worker1, key %s routed to %s", key, got)
		}
	}
}

// TestDistribution 驗證虛擬節點使路由分佈均勻（兩節點各分配 30%~70% 之間）
func TestDistribution(t *testing.T) {
	ring := NewConsistentHash(150)
	ring.Add("http://worker1:8001")
	ring.Add("http://worker2:8001")

	counts := map[string]int{}
	total := 1000
	for i := 0; i < total; i++ {
		key := fmt.Sprintf("system_prompt_%d", i)
		counts[ring.Get(key)]++
	}

	for node, count := range counts {
		ratio := float64(count) / float64(total)
		if ratio < 0.30 || ratio > 0.70 {
			t.Errorf("node %s has unbalanced ratio %.2f (expected 0.30-0.70)", node, ratio)
		}
	}
}
```

- [ ] **Step 2: 確認測試目前失敗**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/gateway
go test ./router/... 2>&1
```

Expected: `FAIL` — `NewConsistentHash undefined`

- [ ] **Step 3: 實作 `gateway/router/consistent_hash.go`**

```go
// Package router 實作 API Gateway 的路由核心邏輯。
//
// 一致性雜湊環（Consistent Hash Ring）解決的問題：
//   普通取模路由（key % N）在節點增刪時會重新映射大量 key，
//   導致快取失效（Cache Thrashing）。
//   一致性雜湊只重新映射 key/N 比例的 key，
//   對語義快取系統而言，能確保「同一語境的請求穩定落在同一台 Worker」，
//   最大化 Worker 本地快取的命中率。
package router

import (
	"fmt"
	"hash/fnv"
	"sort"
	"sync"
)

// ConsistentHash 是執行緒安全的一致性雜湊環。
// 每個實體節點（Worker URL）展開為 replicas 個虛擬節點，
// 虛擬節點均勻散佈在雜湊環上，使負載分佈更均勻。
type ConsistentHash struct {
	replicas int
	ring     map[uint32]string // 雜湊值 → Worker URL
	keys     []uint32          // 已排序的雜湊值（用於二分查詢）
	mu       sync.RWMutex
}

// NewConsistentHash 建立一個新的一致性雜湊環。
// replicas：每個實體節點的虛擬節點數，建議 100~200。
// 數值越大，分佈越均勻，但記憶體消耗略增（每節點 replicas * 8 bytes）。
func NewConsistentHash(replicas int) *ConsistentHash {
	return &ConsistentHash{
		replicas: replicas,
		ring:     make(map[uint32]string),
	}
}

// Add 向雜湊環加入一個 Worker 節點。
// 同一個 node 重複加入是安全的（冪等操作）。
func (c *ConsistentHash) Add(node string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := 0; i < c.replicas; i++ {
		// 虛擬節點鍵："{index}-{node}"，確保每個虛擬節點的雜湊值不同
		h := c.hash(fmt.Sprintf("%d-%s", i, node))
		c.ring[h] = node
		c.keys = append(c.keys, h)
	}
	sort.Slice(c.keys, func(i, j int) bool { return c.keys[i] < c.keys[j] })
}

// Remove 從雜湊環移除一個 Worker 節點（例如節點下線）。
// 移除後，原本路由到此節點的 key 會平滑遷移到環上的下一個節點。
func (c *ConsistentHash) Remove(node string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := 0; i < c.replicas; i++ {
		h := c.hash(fmt.Sprintf("%d-%s", i, node))
		delete(c.ring, h)
	}

	// 重建排序鍵列表（過濾掉已刪除的虛擬節點）
	newKeys := c.keys[:0]
	for _, k := range c.keys {
		if _, exists := c.ring[k]; exists {
			newKeys = append(newKeys, k)
		}
	}
	c.keys = newKeys
}

// Get 根據 key 查詢應路由到哪台 Worker（URL 字串）。
// 雜湊環為空時回傳空字串，呼叫方應做 fallback 處理。
//
// 路由決策流程：
//  1. 計算 key 的 FNV-1a 雜湊值 H
//  2. 在排序的虛擬節點環上，找到第一個 >= H 的節點（順時針最近節點）
//  3. 若 H 大於所有節點，則環繞回第一個節點（環形結構）
func (c *ConsistentHash) Get(key string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.keys) == 0 {
		return ""
	}

	h := c.hash(key)
	// sort.Search 找第一個使 c.keys[i] >= h 的索引
	idx := sort.Search(len(c.keys), func(i int) bool { return c.keys[i] >= h })
	// 環繞：若 idx 超出範圍，回到環的起點
	if idx == len(c.keys) {
		idx = 0
	}
	return c.ring[c.keys[idx]]
}

// Nodes 回傳目前所有實體節點的清單（不含重複虛擬節點）。
// 用於 Gateway 健康檢查端點與監控。
func (c *ConsistentHash) Nodes() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	seen := make(map[string]struct{})
	var nodes []string
	for _, v := range c.ring {
		if _, ok := seen[v]; !ok {
			seen[v] = struct{}{}
			nodes = append(nodes, v)
		}
	}
	return nodes
}

// hash 以 FNV-1a 32-bit 演算法計算字串的雜湊值。
// 選用 FNV-1a 的原因：無依賴、速度快、對短字串的雪崩效應良好。
func (c *ConsistentHash) hash(key string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	return h.Sum32()
}
```

- [ ] **Step 4: 執行測試確認全部通過**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/gateway
go test ./router/... -v -run "TestAddAndGet|TestEmptyRing|TestRemoveReducesNodes|TestDistribution"
```

Expected:
```
--- PASS: TestAddAndGet (0.00s)
--- PASS: TestEmptyRing (0.00s)
--- PASS: TestRemoveReducesNodes (0.00s)
--- PASS: TestDistribution (0.00s)
PASS
```

- [ ] **Step 5: Commit**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
git add gateway/router/consistent_hash.go gateway/router/consistent_hash_test.go
git commit -m "feat(gateway): implement consistent hash ring with virtual nodes"
```

---

## Task 3: Go — KV-Cache Indexer 介面

**Files:**
- Create: `gateway/indexer/indexer.go`

**設計決策：** 此 Task 僅定義介面與 NoopIndexer，不實作真實後端。
Indexer 是 llm-d KV-Cache Indexer 的概念移植：
Worker 登錄自己持有的 prefix 快取 → Gateway 查詢 → 做快取感知路由（優先於純雜湊路由）。

- [ ] **Step 1: 建立 `gateway/indexer/indexer.go`**

```go
// Package indexer 定義全域 KV-Cache 索引器介面。
//
// 系統路由優先級：
//   1. Indexer.Lookup(prefix) → 若有快取感知建議，優先使用
//   2. ConsistentHash.Get(prefix) → Fallback 靜態雜湊路由
//
// 這使得系統在初期（NoopIndexer）表現為純一致性雜湊路由，
// 在未來實作 RedisIndexer/EtcdIndexer 後自動升級為快取感知路由，
// 無需修改 Gateway 主體邏輯（開放封閉原則）。
package indexer

import "context"

// CacheEntry 代表一筆 Worker 的快取登錄記錄。
type CacheEntry struct {
	// WorkerID 是 Worker 的唯一識別，格式為 "host:port"
	WorkerID string
	// Prefixes 是該 Worker 目前持有快取的 system_prompt 前綴清單
	Prefixes []string
	// HitCount 是累計快取命中次數，可用於路由優先度排序
	HitCount int64
}

// Indexer 定義全域 KV-Cache 索引查詢介面。
type Indexer interface {
	// Register 讓 Worker 在快取寫入後，向 Gateway 登錄新的快取前綴。
	Register(ctx context.Context, entry CacheEntry) error

	// Lookup 根據 system_prompt 前綴，回傳最適合的 Worker ID。
	// 若無快取感知建議，回傳 ("", nil)，由一致性雜湊決定。
	Lookup(ctx context.Context, prefix string) (workerID string, err error)

	// Unregister 在 Worker 下線時清除其所有快取登錄資料。
	Unregister(ctx context.Context, workerID string) error

	// Stats 回傳所有 Worker 的快取統計（用於監控 Dashboard）。
	Stats(ctx context.Context) ([]CacheEntry, error)
}

// NoopIndexer 是空實作，所有查詢直接回傳空值，
// 路由決策完全交給一致性雜湊環。這是預設行為。
type NoopIndexer struct{}

func (n *NoopIndexer) Register(_ context.Context, _ CacheEntry) error   { return nil }
func (n *NoopIndexer) Lookup(_ context.Context, _ string) (string, error) { return "", nil }
func (n *NoopIndexer) Unregister(_ context.Context, _ string) error      { return nil }
func (n *NoopIndexer) Stats(_ context.Context) ([]CacheEntry, error)     { return nil, nil }

// TODO: RedisIndexer 實作
// 當 Worker 寫入快取後，透過 HTTP 回呼 Gateway /indexer/register，
// Gateway 以 Redis HSET 儲存 {prefix → workerID} 映射。
// Lookup 時做前綴字串匹配（HSCAN + strings.HasPrefix），
// 找出命中率最高的 Worker 並回傳其 ID。
//
// type RedisIndexer struct {
//     client *redis.Client
//     prefix string // Redis key 前綴，避免命名衝突
// }
```

- [ ] **Step 2: 確認編譯無誤**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/gateway
go build ./indexer/...
```

Expected: 無輸出（編譯成功）

- [ ] **Step 3: Commit**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
git add gateway/indexer/indexer.go
git commit -m "feat(gateway): define KV-Cache Indexer interface with NoopIndexer"
```

---

## Task 4: Go — 反向代理邏輯

**Files:**
- Create: `gateway/router/proxy.go`

**設計決策：** 使用標準庫 `net/http/httputil.ReverseProxy`，
無需外部依賴。路由鍵取自請求 JSON Body 的 `system_prompt` 欄位；
若 Body 已被讀取，需以 `io.NopCloser` 重設 Body 供代理轉發。

- [ ] **Step 1: 建立 `gateway/router/proxy.go`**

```go
package router

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"

	"semantic-cache-router/gateway/indexer"
)

// QueryRequest 是從使用者接收的 JSON 請求格式。
// system_prompt 用作路由決策的主鍵（雜湊輸入），
// 相同 system_prompt 的請求將穩定路由到同一台 Worker，
// 使該 Worker 的語義快取對此類請求持續有效。
type QueryRequest struct {
	SystemPrompt string `json:"system_prompt"` // 路由決策鍵：LLM 角色設定
	Query        string `json:"query"`          // 使用者的自然語言查詢
	UserID       string `json:"user_id"`        // 可選：用於追蹤與稽核
}

// ProxyHandler 封裝完整的路由與代理邏輯。
// 持有一致性雜湊環與 Indexer，依路由優先級決定目標 Worker。
type ProxyHandler struct {
	hash    *ConsistentHash
	indexer indexer.Indexer
	logger  *slog.Logger
}

// NewProxyHandler 建立 ProxyHandler 實例。
// idx 傳入 &indexer.NoopIndexer{} 即為純雜湊路由模式。
func NewProxyHandler(hash *ConsistentHash, idx indexer.Indexer, logger *slog.Logger) *ProxyHandler {
	return &ProxyHandler{hash: hash, indexer: idx, logger: logger}
}

// ServeHTTP 實作 http.Handler，是 Gateway 的核心路由邏輯。
//
// 執行流程：
//  1. 讀取並解析 JSON Body，取得 system_prompt 作為路由鍵
//  2. 詢問 Indexer 是否有「快取感知」的路由建議
//  3. 若 Indexer 無建議，使用一致性雜湊環決定目標 Worker
//  4. 重設 Body（因為已被讀取），透過 ReverseProxy 轉發
func (p *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// 步驟 1：讀取 Body 並解析路由鍵
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	// 重要：讀取後必須重設 Body，否則 ReverseProxy 轉發時 Body 為空
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req QueryRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}

	routingKey := req.SystemPrompt
	if routingKey == "" {
		// Fallback：若無 system_prompt，使用 query 本身作為路由鍵
		// 這會降低快取命中率，但確保系統仍可正常運作
		routingKey = req.Query
	}

	// 步驟 2：詢問 Indexer（快取感知路由，優先級高於雜湊路由）
	targetWorker, err := p.indexer.Lookup(context.Background(), routingKey)
	if err != nil {
		p.logger.Warn("indexer lookup failed, falling back to consistent hash", "error", err)
	}

	// 步驟 3：若 Indexer 無建議，使用一致性雜湊環
	if targetWorker == "" {
		targetWorker = p.hash.Get(routingKey)
	}

	if targetWorker == "" {
		p.logger.Error("no available workers in the hash ring")
		http.Error(w, "no available workers", http.StatusServiceUnavailable)
		return
	}

	p.logger.Info("routing request",
		"routing_key_prefix", truncate(routingKey, 50),
		"target_worker", targetWorker,
	)

	// 步驟 4：建立 ReverseProxy 並轉發請求
	target, err := url.Parse(targetWorker)
	if err != nil {
		p.logger.Error("invalid worker URL", "url", targetWorker, "error", err)
		http.Error(w, "internal routing error", http.StatusInternalServerError)
		return
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	// 自訂錯誤處理：Worker 不可用時回傳 502 而非預設的空白回應
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		p.logger.Error("proxy error", "target", targetWorker, "error", err)
		http.Error(w, "worker unavailable", http.StatusBadGateway)
	}
	proxy.ServeHTTP(w, r)
}

// truncate 截短字串用於日誌輸出，避免過長的 system_prompt 污染 log。
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
```

- [ ] **Step 2: 確認編譯**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/gateway
go build ./router/...
```

Expected: 無輸出

- [ ] **Step 3: Commit**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
git add gateway/router/proxy.go
git commit -m "feat(gateway): implement reverse proxy with consistent hash routing"
```

---

## Task 5: Go — `main.go` 組裝與啟動

**Files:**
- Create: `gateway/main.go`

- [ ] **Step 1: 建立 `gateway/main.go`**

```go
// Package main 是 API Gateway 的入口點。
//
// 系統啟動流程：
//  1. 從環境變數讀取 Worker 清單（支援 Docker Compose 動態注入）
//  2. 建立一致性雜湊環，注入所有 Worker 節點
//  3. 建立 NoopIndexer（未來替換為 RedisIndexer）
//  4. 組裝 ProxyHandler 並啟動 Gin HTTP 伺服器
package main

import (
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
	"semantic-cache-router/gateway/indexer"
	"semantic-cache-router/gateway/router"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	// 從環境變數讀取 Worker 清單
	// 格式：WORKERS=http://worker1:8001,http://worker2:8001
	// Docker Compose 在啟動時注入此環境變數
	workersEnv := os.Getenv("WORKERS")
	if workersEnv == "" {
		// 本地開發 fallback
		workersEnv = "http://localhost:8001,http://localhost:8002"
	}
	workers := strings.Split(workersEnv, ",")

	// 建立一致性雜湊環
	// 150 個虛擬節點：在 2-10 個節點的情況下，分佈偏差 < 5%
	hashReplicas := 150
	ring := router.NewConsistentHash(hashReplicas)
	for _, w := range workers {
		w = strings.TrimSpace(w)
		if w != "" {
			ring.Add(w)
			logger.Info("registered worker in hash ring", "worker", w)
		}
	}

	// 建立 Indexer（目前為 Noop，未來替換為 RedisIndexer 實現快取感知路由）
	idx := &indexer.NoopIndexer{}

	// 組裝 ProxyHandler
	proxyHandler := router.NewProxyHandler(ring, idx, logger)

	// 建立 Gin Engine
	gin.SetMode(gin.ReleaseMode)
	engine := gin.New()
	engine.Use(gin.Recovery()) // Panic 保護

	// 主要路由：轉發所有 /query 請求到後端 Worker
	engine.POST("/query", func(c *gin.Context) {
		proxyHandler.ServeHTTP(c.Writer, c.Request)
	})

	// 健康檢查端點：Kubernetes/Docker 用於探測 Gateway 存活
	engine.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "ok",
			"workers": ring.Nodes(),
		})
	})

	// Indexer 登錄端點（預留）：Worker 在快取寫入後呼叫此端點
	// 未來實作 RedisIndexer 後，此端點將真正記錄快取前綴資訊
	engine.POST("/indexer/register", func(c *gin.Context) {
		var entry indexer.CacheEntry
		if err := c.ShouldBindJSON(&entry); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if err := idx.Register(c.Request.Context(), entry); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"registered": true})
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	logger.Info("gateway starting", "port", port, "worker_count", len(workers))
	if err := engine.Run(":" + port); err != nil {
		logger.Error("gateway failed to start", "error", err)
		os.Exit(1)
	}
}
```

- [ ] **Step 2: 下載依賴並確認編譯**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/gateway
go mod tidy
go build ./...
```

Expected: 生成 `go.sum`，無編譯錯誤

- [ ] **Step 3: 本地快速驗證（無 Worker 時應回傳 503）**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/gateway
WORKERS="" go run . &
sleep 1
curl -s http://localhost:8080/health | python3 -m json.tool
# 預期：{"status": "ok", "workers": []}
curl -s -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"system_prompt":"You are helpful","query":"hello"}' 
# 預期：HTTP 503 "no available workers"
kill %1
```

- [ ] **Step 4: Commit**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
git add gateway/main.go gateway/go.sum
git commit -m "feat(gateway): wire up gin server with health check and indexer endpoint"
```

---

## Task 6: Python — EvictionPolicy 抽象 + SemanticCacheManager

**Files:**
- Create: `worker/semantic_cache.py`
- Create: `worker/tests/test_semantic_cache.py`

**設計決策：**
- `EvictionPolicy` 使用 Python ABC 定義介面，強制子類實作 `should_evict`
- `SphereLFU` 是預留的核密度估計淘汰策略 stub，包含詳細 TODO 說明
- `SemanticCacheManager` 包裝 redisvl `SemanticCache`，隔離外部依賴

- [ ] **Step 1: 撰寫失敗測試 `worker/tests/test_semantic_cache.py`**

```python
"""
語義快取核心邏輯的單元測試。

測試策略：
- 使用 pytest-mock 模擬 redisvl SemanticCache，避免測試依賴真實 Redis
- 分別測試 Cache Hit / Cache Miss 路徑
- 測試 EvictionPolicy 介面的合約
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
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
    """Cache Hit 路徑：相似度足夠時直接回傳快取結果"""

    @patch("semantic_cache.SemanticCache")
    def test_cache_hit_returns_cached_response(self, MockSemanticCache):
        # Arrange
        mock_cache_instance = MagicMock()
        MockSemanticCache.return_value = mock_cache_instance
        # 模擬 SemanticCache.check() 回傳命中結果
        mock_cache_instance.check.return_value = [
            {"response": "Machine learning is a subset of AI.", "metadata": {}}
        ]

        manager = SemanticCacheManager(redis_url="redis://localhost:6379")

        # Act
        result = manager.query(
            system_prompt="You are a helpful assistant.",
            user_query="What is machine learning?",
        )

        # Assert
        assert result.is_cache_hit is True
        assert result.response == "Machine learning is a subset of AI."
        assert result.latency_ms >= 0
        mock_cache_instance.check.assert_called_once()
        mock_cache_instance.store.assert_not_called()  # Cache Hit 不應寫入

    @patch("semantic_cache.SemanticCache")
    def test_cache_miss_calls_llm_and_stores(self, MockSemanticCache):
        # Arrange
        mock_cache_instance = MagicMock()
        MockSemanticCache.return_value = mock_cache_instance
        mock_cache_instance.check.return_value = []  # 無快取命中

        manager = SemanticCacheManager(redis_url="redis://localhost:6379")

        with patch.object(manager, "_call_llm", return_value="Deep learning uses neural networks.") as mock_llm:
            # Act
            result = manager.query(
                system_prompt="You are a helpful assistant.",
                user_query="Explain deep learning.",
            )

        # Assert
        assert result.is_cache_hit is False
        assert result.response == "Deep learning uses neural networks."
        mock_llm.assert_called_once_with(
            system_prompt="You are a helpful assistant.",
            user_query="Explain deep learning.",
        )
        mock_cache_instance.store.assert_called_once()  # Cache Miss 應寫入快取

    @patch("semantic_cache.SemanticCache")
    def test_cache_key_combines_system_prompt_and_query(self, MockSemanticCache):
        """確認快取鍵包含 system_prompt，使不同角色設定的相同問題互不影響"""
        mock_cache_instance = MagicMock()
        MockSemanticCache.return_value = mock_cache_instance
        mock_cache_instance.check.return_value = []

        manager = SemanticCacheManager(redis_url="redis://localhost:6379")

        with patch.object(manager, "_call_llm", return_value="answer"):
            manager.query(system_prompt="You are a doctor.", user_query="What is aspirin?")
            manager.query(system_prompt="You are a chef.", user_query="What is aspirin?")

        assert mock_cache_instance.check.call_count == 2
        # 兩次呼叫的 prompt 參數應不同（包含不同的 system_prompt 前綴）
        calls = mock_cache_instance.check.call_args_list
        prompt_0 = calls[0][1].get("prompt") or calls[0][0][0]
        prompt_1 = calls[1][1].get("prompt") or calls[1][0][0]
        assert prompt_0 != prompt_1
```

- [ ] **Step 2: 確認測試失敗**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/worker
python -m pytest tests/test_semantic_cache.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'SemanticCacheManager' from 'semantic_cache'`

- [ ] **Step 3: 實作 `worker/semantic_cache.py`**

```python
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
    未來透過 EvictionPolicy 介面實作 SphereLFU（球形軟性頻率更新），
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


# ---------------------------------------------------------------------------
# 資料類型
# ---------------------------------------------------------------------------

@dataclass
class CacheResult:
    """封裝單次查詢的完整結果，包含快取命中狀態與延遲指標。"""
    response: str
    is_cache_hit: bool
    latency_ms: float
    worker_id: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvictionPolicy 抽象介面
# ---------------------------------------------------------------------------

class EvictionPolicy(ABC):
    """
    快取淘汰策略的抽象介面。

    子類需實作 should_evict()，返回 True 表示該條目應被淘汰。
    SemanticCacheManager 在每次寫入快取前，可呼叫此介面決定是否
    先騰出空間（目前 redisvl 以 TTL 為主，此介面為進階淘汰的擴充點）。
    """

    @abstractmethod
    def should_evict(self, key: str, access_count: int, last_access_ts: float) -> bool:
        """
        判斷指定快取條目是否應被淘汰。

        Args:
            key: 快取條目的向量鍵（通常是查詢文字的雜湊）
            access_count: 該條目的歷史存取次數
            last_access_ts: 最後一次存取的 Unix timestamp

        Returns:
            True 表示應淘汰，False 表示應保留
        """
        ...

    @abstractmethod
    def record_access(self, key: str) -> None:
        """記錄一次存取，更新內部頻率統計（用於頻率估計演算法）。"""
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
           （使用指數移動平均 EMA 實現「軟性」更新）
    """

    def __init__(self, radius: float = 0.15, bandwidth: float = 0.05):
        """
        Args:
            radius: 球形鄰域半徑（餘弦距離），建議 0.1~0.2
            bandwidth: KDE 核函數帶寬，控制密度估計的平滑程度
        """
        self.radius = radius
        self.bandwidth = bandwidth
        # TODO: 注入 Redis client 以存取鄰居向量的頻率計數

    def should_evict(self, key: str, access_count: int, last_access_ts: float) -> bool:
        """
        TODO: 實作基於核密度估計的淘汰決策。
        目前版本拋出 NotImplementedError，讓系統 fallback 到 TTL 淘汰。
        """
        raise NotImplementedError(
            "SphereLFU.should_evict() not yet implemented. "
            "System falls back to redisvl TTL-based eviction. "
            "See class docstring for implementation roadmap."
        )

    def record_access(self, key: str) -> None:
        """
        TODO: 實作鄰域衰減頻率更新。
        """
        raise NotImplementedError("SphereLFU.record_access() not yet implemented.")


class TTLOnlyPolicy(EvictionPolicy):
    """
    純 TTL 淘汰策略（預設使用）。
    直接委派給 redisvl 的 TTL 機制，should_evict 永遠回傳 False。
    """

    def should_evict(self, key: str, access_count: int, last_access_ts: float) -> bool:
        return False  # 完全依賴 TTL，不主動淘汰

    def record_access(self, key: str) -> None:
        pass  # TTL 策略不需要記錄存取頻率


# ---------------------------------------------------------------------------
# SemanticCacheManager
# ---------------------------------------------------------------------------

class SemanticCacheManager:
    """
    語義快取管理器：封裝 redisvl SemanticCache 的完整查詢流程。

    快取鍵設計：
        鍵 = f"{system_prompt}\\n\\n{user_query}"
        將 system_prompt 納入鍵，確保不同角色設定的查詢互相隔離，
        避免「客服機器人」和「醫療顧問」對同一問題共用錯誤的快取結果。

    距離門檻說明：
        redisvl 使用餘弦距離（Cosine Distance），值域 [0, 2]，
        值越小代表語義越相似。
        distance_threshold=0.2 等同於餘弦相似度 >= 0.8，
        對應使用者規格的「相似度門檻 0.8」。
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

        # 初始化 HuggingFace 向量化器（sentence-transformers 後端）
        # all-MiniLM-L6-v2：384 維向量，速度與品質的良好平衡點
        # 首次啟動會從 HuggingFace Hub 下載模型（約 80MB），之後快取本地
        vectorizer = HFTextVectorizer(model=model_name)

        # 初始化 redisvl SemanticCache
        # distance_threshold: 向量餘弦距離 <= 此值視為 Cache Hit
        # ttl: 快取條目的存活秒數（TTL 到期後 Redis 自動刪除）
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

        # 組合快取查詢鍵：system_prompt + 分隔符 + user_query
        # 確保不同 system_prompt 的相同 query 不互相污染快取
        cache_prompt = f"{system_prompt}\n\n{user_query}"

        # 向量化 + ANN 搜尋（redisvl 內部完成）
        hits = self._cache.check(prompt=cache_prompt, num_results=1)

        latency_ms = (time.monotonic() - start_ts) * 1000

        if hits:
            # Cache Hit：直接回傳快取結果，無需呼叫 LLM
            logger.info("cache hit", extra={"query_prefix": user_query[:50]})
            return CacheResult(
                response=hits[0]["response"],
                is_cache_hit=True,
                latency_ms=latency_ms,
                metadata=hits[0].get("metadata", {}),
            )

        # Cache Miss：呼叫 LLM（Dummy 模擬，實際替換為 OpenAI SDK）
        logger.info("cache miss, calling LLM", extra={"query_prefix": user_query[:50]})
        llm_response = self._call_llm(
            system_prompt=system_prompt,
            user_query=user_query,
        )

        # 將 LLM 結果寫入 RedisVL 快取（含 TTL）
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
        呼叫 LLM API 的抽象方法。

        目前為 Dummy 實作，回傳模擬回應。
        替換為真實 OpenAI 呼叫時，修改此方法即可，其餘邏輯不變。

        替換範例（OpenAI SDK）：
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
        logger.warning(
            "Using DUMMY LLM implementation. "
            "Replace _call_llm() with real OpenAI/Anthropic SDK call."
        )
        # Dummy 回應：模擬 LLM 處理延遲
        time.sleep(0.1)
        return (
            f"[DUMMY LLM RESPONSE] "
            f"Query: '{user_query[:80]}' | "
            f"System: '{system_prompt[:40]}'"
        )
```

- [ ] **Step 4: 執行測試確認通過**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/worker
pip install -r requirements.txt -q
python -m pytest tests/test_semantic_cache.py -v
```

Expected:
```
PASSED tests/test_semantic_cache.py::TestEvictionPolicyInterface::test_cannot_instantiate_abstract_policy
PASSED tests/test_semantic_cache.py::TestEvictionPolicyInterface::test_sphere_lfu_instantiates
PASSED tests/test_semantic_cache.py::TestEvictionPolicyInterface::test_sphere_lfu_should_evict_raises_not_implemented
PASSED tests/test_semantic_cache.py::TestSemanticCacheManagerHit::test_cache_hit_returns_cached_response
PASSED tests/test_semantic_cache.py::TestSemanticCacheManagerHit::test_cache_miss_calls_llm_and_stores
PASSED tests/test_semantic_cache.py::TestSemanticCacheManagerHit::test_cache_key_combines_system_prompt_and_query
```

- [ ] **Step 5: Commit**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
git add worker/semantic_cache.py worker/tests/test_semantic_cache.py
git commit -m "feat(worker): semantic cache manager with eviction policy interface"
```

---

## Task 7: Python — FastAPI Worker 主程式

**Files:**
- Create: `worker/main.py`
- Create: `worker/tests/test_main.py`

- [ ] **Step 1: 撰寫 API 端點測試 `worker/tests/test_main.py`**

```python
"""FastAPI Worker 端點整合測試。"""
import pytest
from unittest.mock import patch, MagicMock
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
            json={"system_prompt": "You are helpful."},  # 缺少 query 欄位
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
```

- [ ] **Step 2: 確認測試失敗**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/worker
python -m pytest tests/test_main.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'app' from 'main'`

- [ ] **Step 3: 實作 `worker/main.py`**

```python
"""
Python Worker FastAPI 入口。

Worker 在 Polyglot 架構中的角色：
    - 接收 Go Gateway 透過一致性雜湊路由過來的請求
    - 因為一致性雜湊的「穩定性」，相同 system_prompt 的請求
      會持續路由到同一台 Worker，使本地 Redis 快取效益最大化
    - 每台 Worker 擁有獨立的 Redis 快取命名空間（以 worker_id 區分），
      避免多 Worker 共用快取時的快取污染問題（未來擴充）

水平擴充策略：
    增加 Worker 數量時，Go Gateway 自動將新 Worker 加入雜湊環。
    因一致性雜湊特性，只有 1/N 的 key 需要遷移，快取命中率不會大幅下降。
"""
import logging
import os
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from semantic_cache import SemanticCacheManager

# 設定結構化日誌
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# Worker 唯一識別（Docker 容器 hostname 即為服務名稱）
WORKER_ID = os.getenv("WORKER_ID", socket.gethostname())
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# 全域 SemanticCacheManager（應用程式生命週期內共享）
cache_manager: SemanticCacheManager = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 管理器：在應用程式啟動時初始化快取管理器。
    
    使用 lifespan 而非 @app.on_event("startup") 是 FastAPI 0.93+ 的建議做法，
    確保啟動/關閉邏輯在同一個函數內，易於資源管理。
    """
    global cache_manager
    logger.info(f"Worker {WORKER_ID} starting, connecting to Redis: {REDIS_URL}")
    cache_manager = SemanticCacheManager(
        redis_url=REDIS_URL,
        distance_threshold=float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.2")),
        ttl=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
    )
    logger.info(f"Worker {WORKER_ID} ready")
    yield
    # 關閉時釋放資源（redisvl 連線池）
    logger.info(f"Worker {WORKER_ID} shutting down")


app = FastAPI(
    title="Semantic Cache Worker",
    description="語義向量快取 Worker 節點，接收 Go Gateway 路由的 LLM 查詢請求",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 請求 / 回應 Schema
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """來自 Go Gateway 的查詢請求格式（與 Gateway 端 QueryRequest 結構一致）。"""
    system_prompt: str = ""
    query: str
    user_id: str = ""


class QueryResponse(BaseModel):
    """回傳給 Go Gateway 的查詢結果格式。"""
    response: str
    cache_hit: bool
    latency_ms: float
    worker_id: str


# ---------------------------------------------------------------------------
# API 端點
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest) -> QueryResponse:
    """
    主要查詢端點。

    執行流程：
    1. 呼叫 SemanticCacheManager.query()
    2. 內部做向量搜尋 → Cache Hit/Miss 判斷
    3. Cache Miss 時呼叫 LLM（Dummy）並寫入 Redis
    4. 回傳統一格式的 QueryResponse

    Gateway 在收到此回應後，直接透過 ReverseProxy 轉發給使用者，
    無需額外的格式轉換（Gateway 對 Worker 的回應格式透明）。
    """
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
    """
    健康檢查端點。
    Go Gateway 的 /health 端點會聚合所有 Worker 的狀態。
    Docker Compose healthcheck 也使用此端點確認 Worker 已就緒。
    """
    return {
        "status": "ok",
        "worker_id": WORKER_ID,
        "redis_url": REDIS_URL,
    }
```

- [ ] **Step 4: 執行測試確認通過**

```bash
cd /Users/peiwen/Projects/semantic-cache-router/worker
python -m pytest tests/test_main.py -v
```

Expected: 所有 5 個測試 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
git add worker/main.py worker/tests/test_main.py
git commit -m "feat(worker): FastAPI worker with semantic cache integration"
```

---

## Task 8: Docker Compose 容器編排

**Files:**
- Create: `docker-compose.yml`
- Create: `worker/Dockerfile`
- Create: `gateway/Dockerfile`

- [ ] **Step 1: 建立 `worker/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 先複製依賴清單（利用 Docker layer cache：依賴不變則跳過安裝）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY main.py semantic_cache.py ./

# 預先下載 sentence-transformers 模型（避免首次請求時的冷啟動延遲）
RUN python -c "from redisvl.utils.vectorize import HFTextVectorizer; HFTextVectorizer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
```

- [ ] **Step 2: 建立 `gateway/Dockerfile`**

```dockerfile
FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum* ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o gateway .

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=builder /app/gateway .

EXPOSE 8080
CMD ["./gateway"]
```

- [ ] **Step 3: 建立 `docker-compose.yml`**

```yaml
version: "3.9"

# ============================================================
# 分散式語義快取路由系統 — 完整服務編排
#
# 服務拓撲：
#   使用者 → [gateway:8080] → 一致性雜湊 → [worker1:8001] ↘
#                                          → [worker2:8001] → [redis:6379]
#
# 路由穩定性驗證：
#   同一 system_prompt 的請求，無論發送幾次，
#   都應路由到同一台 Worker（觀察 worker_id 欄位是否固定）。
# ============================================================

services:

  # Redis Stack：向量資料庫 + TTL 管理 + RediSearch 向量索引
  # redis/redis-stack 包含 RedisSearch、RedisJSON 等模組，
  # redisvl 需要 RediSearch 模組才能建立 HNSW 向量索引
  redis:
    image: redis/redis-stack:7.2.0-v9
    container_name: semantic-cache-redis
    ports:
      - "6379:6379"    # Redis 協議
      - "8001:8001"    # RedisInsight Web UI（向量索引視覺化工具）
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  # Python Worker 1：語義快取處理節點
  worker1:
    build:
      context: ./worker
      dockerfile: Dockerfile
    container_name: semantic-cache-worker1
    environment:
      - WORKER_ID=worker1
      - REDIS_URL=redis://redis:6379
      - CACHE_DISTANCE_THRESHOLD=0.2
      - CACHE_TTL_SECONDS=3600
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 10s
      timeout: 5s
      retries: 5
    # 注意：Worker 不對外暴露 port，只允許 Gateway 透過 Docker network 存取
    # 這是安全設計：防止繞過 Gateway 直接呼叫 Worker

  # Python Worker 2：用於測試一致性雜湊負載分配
  worker2:
    build:
      context: ./worker
      dockerfile: Dockerfile
    container_name: semantic-cache-worker2
    environment:
      - WORKER_ID=worker2
      - REDIS_URL=redis://redis:6379
      - CACHE_DISTANCE_THRESHOLD=0.2
      - CACHE_TTL_SECONDS=3600
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Go API Gateway：控制層與路由層
  gateway:
    build:
      context: ./gateway
      dockerfile: Dockerfile
    container_name: semantic-cache-gateway
    ports:
      - "8080:8080"    # 對外暴露的唯一入口
    environment:
      # 一致性雜湊 Worker 清單，使用 Docker Compose 服務名稱解析
      - WORKERS=http://worker1:8001,http://worker2:8001
      - PORT=8080
    depends_on:
      worker1:
        condition: service_healthy
      worker2:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  redis_data:
    driver: local

networks:
  default:
    name: semantic-cache-network
```

- [ ] **Step 4: 驗證 compose 設定語法**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
docker compose config --quiet
```

Expected: 無報錯（語法正確）

- [ ] **Step 5: Commit**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
git add docker-compose.yml worker/Dockerfile gateway/Dockerfile
git commit -m "feat: add docker compose with redis stack, gateway and 2 workers"
```

---

## Task 9: 端對端驗證腳本

**Files:**
- Create: `scripts/e2e_test.sh`

目的：在 `docker compose up` 後，驗證一致性雜湊路由穩定性與語義快取命中。

- [ ] **Step 1: 建立 `scripts/e2e_test.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

GATEWAY="http://localhost:8080"
echo "=== 語義快取路由系統 E2E 驗證 ==="

# 測試 1：健康檢查
echo -e "\n[1] Gateway 健康檢查"
curl -sf "$GATEWAY/health" | python3 -m json.tool

# 測試 2：一致性雜湊穩定性
# 相同 system_prompt 應路由到同一台 Worker（worker_id 固定）
echo -e "\n[2] 一致性雜湊穩定性（同一 system_prompt 應路由到同一 Worker）"
WORKER_IDS=()
for i in {1..5}; do
  RESP=$(curl -sf -X POST "$GATEWAY/query" \
    -H "Content-Type: application/json" \
    -d '{"system_prompt":"You are a geography expert.","query":"Capital of France?"}')
  WID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['worker_id'])")
  WORKER_IDS+=("$WID")
  echo "  Request $i → worker_id: $WID"
done

UNIQUE=$(printf '%s\n' "${WORKER_IDS[@]}" | sort -u | wc -l)
if [ "$UNIQUE" -eq 1 ]; then
  echo "  PASS: 所有請求路由到同一台 Worker（${WORKER_IDS[0]}）"
else
  echo "  FAIL: 路由不穩定，命中了 $UNIQUE 台不同 Worker"
  exit 1
fi

# 測試 3：語義快取命中
echo -e "\n[3] 語義快取命中測試（第二次相同查詢應命中快取）"
curl -sf -X POST "$GATEWAY/query" \
  -H "Content-Type: application/json" \
  -d '{"system_prompt":"You are helpful.","query":"What is the speed of light?"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  First call  → cache_hit: {d[\"cache_hit\"]}, latency: {d[\"latency_ms\"]:.1f}ms')"

curl -sf -X POST "$GATEWAY/query" \
  -H "Content-Type: application/json" \
  -d '{"system_prompt":"You are helpful.","query":"What is the speed of light?"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Second call → cache_hit: {d[\"cache_hit\"]}, latency: {d[\"latency_ms\"]:.1f}ms')"

echo -e "\n=== 驗證完成 ==="
```

- [ ] **Step 2: 授予執行權限並 commit**

```bash
chmod +x /Users/peiwen/Projects/semantic-cache-router/scripts/e2e_test.sh
cd /Users/peiwen/Projects/semantic-cache-router
git add scripts/e2e_test.sh
git commit -m "test: add e2e validation script for routing stability and cache hit"
```

- [ ] **Step 3: 執行完整系統（選做，需要 Docker）**

```bash
cd /Users/peiwen/Projects/semantic-cache-router
docker compose up --build -d
# 等待所有服務健康（約 60-90 秒，Worker 需下載 sentence-transformers 模型）
docker compose ps
./scripts/e2e_test.sh
```

---

## 自我審查：Spec 覆蓋率

| Spec 需求 | 對應 Task |
|-----------|-----------|
| Monorepo 目錄結構 | Task 1 |
| Go Gateway + JSON 請求接收 | Task 5 |
| 前綴感知路由 / 一致性雜湊 | Task 2 |
| 反向代理到 Worker | Task 4 |
| KV-Cache Indexer 介面 | Task 3 |
| Python FastAPI Worker | Task 7 |
| sentence-transformers Embedding | Task 6 (`HFTextVectorizer`) |
| redisvl SemanticCache 整合 | Task 6 |
| 相似度門檻 0.8（距離 0.2） | Task 6 |
| TTL 機制 | Task 6 |
| Cache Hit/Miss 流程 | Task 6 + Task 7 |
| LLM Dummy 函數 | Task 6 (`_call_llm`) |
| EvictionPolicy 介面 | Task 6 |
| SphereLFU TODO stub | Task 6 |
| docker-compose 多 Worker | Task 8 |
| Redis Stack | Task 8 |
| 中文詳細註解 | 全部 Task |

**無遺漏。**
