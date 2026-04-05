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
func NewConsistentHash(replicas int) *ConsistentHash {
	return &ConsistentHash{
		replicas: replicas,
		ring:     make(map[uint32]string),
	}
}

// Add 向雜湊環加入一個 Worker 節點（冪等操作）。
func (c *ConsistentHash) Add(node string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := 0; i < c.replicas; i++ {
		h := c.hash(fmt.Sprintf("%d-%s", i, node))
		c.ring[h] = node
		c.keys = append(c.keys, h)
	}
	sort.Slice(c.keys, func(i, j int) bool { return c.keys[i] < c.keys[j] })
}

// Remove 從雜湊環移除一個 Worker 節點。
// 移除後，原本路由到此節點的 key 平滑遷移到環上的下一個節點。
func (c *ConsistentHash) Remove(node string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := 0; i < c.replicas; i++ {
		h := c.hash(fmt.Sprintf("%d-%s", i, node))
		delete(c.ring, h)
	}

	newKeys := c.keys[:0]
	for _, k := range c.keys {
		if _, exists := c.ring[k]; exists {
			newKeys = append(newKeys, k)
		}
	}
	c.keys = newKeys
}

// Get 根據 key 查詢應路由到哪台 Worker URL。
// 雜湊環為空時回傳空字串。
//
// 路由決策流程：
//  1. 計算 key 的 FNV-1a 雜湊值 H
//  2. 在排序的虛擬節點環上，找到第一個 >= H 的節點（順時針最近節點）
//  3. 若 H 大於所有節點，環繞回第一個節點
func (c *ConsistentHash) Get(key string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.keys) == 0 {
		return ""
	}

	h := c.hash(key)
	idx := sort.Search(len(c.keys), func(i int) bool { return c.keys[i] >= h })
	if idx == len(c.keys) {
		idx = 0
	}
	return c.ring[c.keys[idx]]
}

// Nodes 回傳目前所有實體節點清單（去重）。
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

// hash 以 FNV-1a 32-bit 計算字串雜湊值。
func (c *ConsistentHash) hash(key string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	return h.Sum32()
}
