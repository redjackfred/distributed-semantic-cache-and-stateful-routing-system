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
// SystemPrompt 用作路由決策的主鍵（雜湊輸入），
// 相同 SystemPrompt 的請求將穩定路由到同一台 Worker，
// 使該 Worker 的語義快取對此類請求持續有效。
type QueryRequest struct {
	SystemPrompt string `json:"system_prompt"`
	Query        string `json:"query"`
	UserID       string `json:"user_id"`
}

// ProxyHandler 封裝完整的路由與代理邏輯。
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
//  4. 重設 Body，透過 ReverseProxy 轉發
func (p *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	// 讀取後必須重設 Body，否則 ReverseProxy 轉發時 Body 為空
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req QueryRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}

	routingKey := req.SystemPrompt
	if routingKey == "" {
		// Fallback：若無 system_prompt，使用 query 本身作為路由鍵
		routingKey = req.Query
	}

	// 詢問 Indexer（快取感知路由，優先級高於雜湊路由）
	targetWorker, err := p.indexer.Lookup(context.Background(), routingKey)
	if err != nil {
		p.logger.Warn("indexer lookup failed, falling back to consistent hash", "error", err)
	}

	// 若 Indexer 無建議，使用一致性雜湊環
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

	target, err := url.Parse(targetWorker)
	if err != nil {
		p.logger.Error("invalid worker URL", "url", targetWorker, "error", err)
		http.Error(w, "internal routing error", http.StatusInternalServerError)
		return
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
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
