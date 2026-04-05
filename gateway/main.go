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
	workersEnv := os.Getenv("WORKERS")
	if workersEnv == "" {
		workersEnv = "http://localhost:8001,http://localhost:8002"
	}
	workers := strings.Split(workersEnv, ",")

	// 建立一致性雜湊環（150 個虛擬節點）
	ring := router.NewConsistentHash(150)
	for _, w := range workers {
		w = strings.TrimSpace(w)
		if w != "" {
			ring.Add(w)
			logger.Info("registered worker in hash ring", "worker", w)
		}
	}

	// 建立 Indexer（目前為 Noop，路由完全由一致性雜湊決定）
	idx := &indexer.NoopIndexer{}

	// 組裝 ProxyHandler
	proxyHandler := router.NewProxyHandler(ring, idx, logger)

	// 建立 Gin Engine
	gin.SetMode(gin.ReleaseMode)
	engine := gin.New()
	engine.Use(gin.Recovery())

	// 主要路由：轉發所有 /query 請求到後端 Worker
	engine.POST("/query", func(c *gin.Context) {
		proxyHandler.ServeHTTP(c.Writer, c.Request)
	})

	// 健康檢查端點
	engine.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "ok",
			"workers": ring.Nodes(),
		})
	})

	// Indexer 登錄端點（預留）：Worker 寫入快取後呼叫此端點
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
