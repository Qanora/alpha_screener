#!/bin/bash
# PR Monitor — fixed, no more ad-hoc jq tweaks.
# Usage: ./watch-pr.sh <pr_number> [timeout_rounds]
# Exit: 0=merged/approved, 1=CI failure, 2=stuck/timeout, 3=gh command failed, 4=changes requested, 5=missing tools, 6=invalid timeout

set -euo pipefail

for cmd in gh jq; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is required but not installed"
    exit 5
  fi
done

REPO="${REPO:-Qanora/alpha_screener}"
PR="${1:?Usage: $0 <pr_number> [timeout_rounds]}"
TIMEOUT="${2:-40}"

# 校验 TIMEOUT 为正整数
case "$TIMEOUT" in
  ''|*[!0-9]*|0) echo "ERROR: timeout_rounds must be a positive integer, got: $TIMEOUT"; exit 6 ;;
esac

ROUND=0
STDERR_FILE=$(mktemp)
trap 'rm -f "$STDERR_FILE"' EXIT INT TERM

while true; do
  ROUND=$((ROUND + 1))

  # 清空旧 stderr，保证每次迭代干净
  : > "$STDERR_FILE"

  if ! RESULT=$(gh pr view "$PR" --repo "$REPO" \
    --json statusCheckRollup,reviewDecision,mergedAt \
    --jq '{
      failing: [(.statusCheckRollup // [])[] |
        select(.status == "COMPLETED" and (.conclusion == "FAILURE" or .conclusion == "TIMED_OUT" or .conclusion == "CANCELLED" or .conclusion == "ACTION_REQUIRED" or .conclusion == "STARTUP_FAILURE")) |
        "\(.name):\(.conclusion)"
      ],
      pending: [(.statusCheckRollup // [])[] |
        select(.status != "COMPLETED" and .status != null) |
        .name
      ],
      review: .reviewDecision,
      merged: .mergedAt
    }' 2>"$STDERR_FILE"); then
    echo "[$ROUND] $(date +%H:%M:%S) gh pr view failed"
    cat "$STDERR_FILE"
    exit 3
  fi

  REVIEW=$(echo "$RESULT" | jq -r '.review')
  MERGED=$(echo "$RESULT" | jq -r '.merged')
  FAILING=$(echo "$RESULT" | jq -r '.failing | join(",")')
  PENDING=$(echo "$RESULT" | jq -r '.pending | join(",")')

  echo "[$ROUND] $(date +%H:%M:%S) review=$REVIEW pending=${PENDING:-none} failing=${FAILING:-none}"

  # Terminal states
  if [ "$MERGED" != "null" ]; then
    echo "=== MERGED at $MERGED ==="
    exit 0
  fi

  if [ "$REVIEW" = "APPROVED" ] && [ -z "$PENDING" ] && [ -z "$FAILING" ]; then
    echo "=== APPROVED with all CI passed — auto-merge should follow ==="
    exit 0
  fi

  if [ -n "$FAILING" ]; then
    echo "=== CI FAILURES: $FAILING ==="
    exit 1
  fi

  if [ "$REVIEW" = "CHANGES_REQUESTED" ]; then
    echo "=== CHANGES_REQUESTED — need close-reopen ==="
    exit 4
  fi

  if [ "$ROUND" -ge "$TIMEOUT" ]; then
    echo "=== STUCK after ${ROUND} rounds ==="
    exit 2
  fi

  sleep 30
done
