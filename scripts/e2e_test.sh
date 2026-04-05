#!/usr/bin/env bash
# E2E 驗證腳本：測試一致性雜湊路由穩定性與語義快取命中
#
# 使用方式：
#   docker compose up --build -d
#   ./scripts/e2e_test.sh
#
# 測試項目：
#   1. Gateway 健康檢查
#   2. 一致性雜湊穩定性：相同 system_prompt → 同一台 Worker
#   3. 語義快取命中：第二次相同查詢應命中快取（延遲更低）

set -euo pipefail

GATEWAY="http://localhost:8080"
PASS=0
FAIL=0

# ANSI 顏色（在 CI 環境可關閉）
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL+1)); }

echo "=== 語義快取路由系統 E2E 驗證 ==="
echo "Gateway: $GATEWAY"
echo ""

# ---------------------------------------------------------------
# 測試 1：健康檢查
# ---------------------------------------------------------------
echo "[1] Gateway 健康檢查"
HEALTH=$(curl -sf "$GATEWAY/health" || echo "")
if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
  pass "Gateway /health returns {status: ok}"
  echo "    Workers: $(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('workers','?'))")"
else
  fail "Gateway /health failed or returned unexpected response: $HEALTH"
fi

echo ""

# ---------------------------------------------------------------
# 測試 2：一致性雜湊穩定性
# 相同 system_prompt 應路由到同一台 Worker（worker_id 固定）
# ---------------------------------------------------------------
echo "[2] 一致性雜湊穩定性（5 次相同 system_prompt → 應路由到同一台 Worker）"
WORKER_IDS=()
for i in {1..5}; do
  RESP=$(curl -sf -X POST "$GATEWAY/query" \
    -H "Content-Type: application/json" \
    -d '{"system_prompt":"You are a geography expert.","query":"Capital of France?"}' || echo "")
  WID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('worker_id','ERROR'))" 2>/dev/null || echo "ERROR")
  WORKER_IDS+=("$WID")
done

UNIQUE=$(printf '%s\n' "${WORKER_IDS[@]}" | sort -u | wc -l | tr -d ' ')
if [ "$UNIQUE" -eq 1 ] && [ "${WORKER_IDS[0]}" != "ERROR" ]; then
  pass "Consistent routing: all 5 requests → ${WORKER_IDS[0]}"
else
  fail "Routing unstable: got $UNIQUE distinct worker_ids (${WORKER_IDS[*]})"
fi

echo ""

# ---------------------------------------------------------------
# 測試 3：語義快取命中
# 第一次：Cache Miss（呼叫 LLM）
# 第二次：Cache Hit（直接回傳，latency 更低）
# ---------------------------------------------------------------
echo "[3] 語義快取命中（第二次相同查詢應命中快取）"
QUERY_PAYLOAD='{"system_prompt":"You are helpful.","query":"What is the speed of light?"}'

RESP1=$(curl -sf -X POST "$GATEWAY/query" \
  -H "Content-Type: application/json" \
  -d "$QUERY_PAYLOAD" || echo "")
HIT1=$(echo "$RESP1" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cache_hit','?'))" 2>/dev/null || echo "?")
LAT1=$(echo "$RESP1" | python3 -c "import sys,json; print(round(json.load(sys.stdin).get('latency_ms',0),1))" 2>/dev/null || echo "?")

RESP2=$(curl -sf -X POST "$GATEWAY/query" \
  -H "Content-Type: application/json" \
  -d "$QUERY_PAYLOAD" || echo "")
HIT2=$(echo "$RESP2" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cache_hit','?'))" 2>/dev/null || echo "?")
LAT2=$(echo "$RESP2" | python3 -c "import sys,json; print(round(json.load(sys.stdin).get('latency_ms',0),1))" 2>/dev/null || echo "?")

echo "    1st call → cache_hit: $HIT1, latency: ${LAT1}ms"
echo "    2nd call → cache_hit: $HIT2, latency: ${LAT2}ms"

if [ "$HIT2" = "True" ]; then
  pass "2nd call cache_hit=True (semantic cache working)"
else
  fail "2nd call cache_hit=$HIT2 (expected True)"
fi

echo ""

# ---------------------------------------------------------------
# 結果摘要
# ---------------------------------------------------------------
echo "=== 驗證結果摘要 ==="
echo -e "  ${GREEN}PASS: $PASS${NC}  ${RED}FAIL: $FAIL${NC}"
if [ "$FAIL" -eq 0 ]; then
  echo -e "  ${GREEN}全部通過 ✓${NC}"
  exit 0
else
  echo -e "  ${RED}有 $FAIL 項測試失敗${NC}"
  exit 1
fi
