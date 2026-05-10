#!/bin/bash
# PR Monitor — fixed, no more ad-hoc jq tweaks.
# Usage: ./watch-pr.sh <pr_number> [timeout_rounds]
# Exit: 0=merged/approved, 1=CI failure, 2=stuck/timeout, 3=gh command failed, 4=changes requested

set -euo pipefail

PR="${1:?Usage: $0 <pr_number> [timeout_rounds]}"
TIMEOUT="${2:-40}"
ROUND=0

while true; do
  ROUND=$((ROUND + 1))

  if ! RESULT=$(gh pr view "$PR" --repo Qanora/alpha_screener \
    --json statusCheckRollup,reviewDecision,mergedAt \
    --jq '{
      failing: [.statusCheckRollup[] |
        select(.status == "COMPLETED" and .conclusion != "SUCCESS" and .conclusion != "SKIPPED") |
        "\(.name):\(.conclusion)"
      ],
      pending: [.statusCheckRollup[] |
        select(.status != "COMPLETED" and .status != null) |
        .name
      ],
      review: .reviewDecision,
      merged: .mergedAt
    }' 2>&1); then
    echo "[$ROUND] $(date +%H:%M:%S) gh pr view failed: $RESULT"
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

  if [ "$REVIEW" = "CHANGES_REQUESTED" ]; then
    echo "=== CHANGES_REQUESTED — need close-reopen ==="
    exit 4
  fi

  if [ -n "$FAILING" ]; then
    echo "=== CI FAILURES: $FAILING ==="
    exit 1
  fi

  if [ "$ROUND" -ge "$TIMEOUT" ]; then
    echo "=== STUCK after ${ROUND} rounds ==="
    exit 2
  fi

  sleep 30
done
