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
