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

func (n *NoopIndexer) Register(_ context.Context, _ CacheEntry) error    { return nil }
func (n *NoopIndexer) Lookup(_ context.Context, _ string) (string, error) { return "", nil }
func (n *NoopIndexer) Unregister(_ context.Context, _ string) error       { return nil }
func (n *NoopIndexer) Stats(_ context.Context) ([]CacheEntry, error)      { return nil, nil }

// TODO: RedisIndexer 實作
// 當 Worker 寫入快取後，透過 HTTP 回呼 Gateway /indexer/register，
// Gateway 以 Redis HSET 儲存 {prefix → workerID} 映射。
// Lookup 時做前綴字串匹配，找出命中率最高的 Worker 並回傳其 ID。
//
// type RedisIndexer struct {
//     client *redis.Client
//     prefix string
// }
