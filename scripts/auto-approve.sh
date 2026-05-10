#!/bin/bash
# Check if CodeRabbit review on a PR meets auto-approve criteria.
# Usage: ./auto-approve.sh <pr_number>
# Exit: 0=approved, 1=rejected (major/critical or too many findings)
#
# Auto-approve rules:
#   1. 0 actionable → approve
#   2. major=0 AND critical=0 AND ≤ 2 trivial/minor AND new fix commit → approve
#   3. major/critical > 0 OR > 2 trivial → reject
#   4. ≤ 2 trivial but no fix commit yet → skip (waiting)

set -euo pipefail

PR="${1:?Usage: $0 <pr_number>}"
REPO="${REPO:-Qanora/alpha_screener}"

REVIEW=$(gh pr view "$PR" --repo "$REPO" --json reviews --jq '
  [.reviews[] | select(.author.login == "coderabbitai[bot]")] | last // empty
')

if [ -z "$REVIEW" ]; then
  echo "No CodeRabbit review found on PR #$PR"
  exit 0
fi

REVIEW_STATE=$(echo "$REVIEW" | jq -r '.state')
REVIEW_BODY=$(echo "$REVIEW" | jq -r '.body')
REVIEW_COMMIT=$(echo "$REVIEW" | jq -r '.commit.oid // ""')

if [ "$REVIEW_STATE" = "APPROVED" ]; then
  echo "Already approved."
  exit 0
fi

ACTIONABLE=$(echo "$REVIEW_BODY" | sed -n 's/.*\*\*Actionable comments posted: \([0-9]\{1,\}\).*/\1/p')
ACTIONABLE=${ACTIONABLE:-0}

# Parse severity from review inline comments (review body is markdown, not JSON)
# Use REST API for numeric ID — GraphQL node IDs don't match pull_request_review_id
REVIEW_ID=$(gh api "repos/$REPO/pulls/$PR/reviews" --jq '[.[] | select(.user.login == "coderabbitai[bot]")] | last | .id')
COMMENTS=$(gh api "repos/$REPO/pulls/$PR/comments" --jq '[.[] | select(.pull_request_review_id == '"$REVIEW_ID"') | .body] | join("\n")' 2>/dev/null || echo "")
MAJORS=$(echo "$COMMENTS" | grep -ciF '_🟠 Major_' || true)
MAJORS=${MAJORS:-0}
CRITICALS=$(echo "$COMMENTS" | grep -ciF '_🔴 Critical_' || true)
CRITICALS=${CRITICALS:-0}

PR_HEAD=$(gh pr view "$PR" --repo "$REPO" --json commits --jq '.commits[-1].oid // ""')
HAS_NEW_COMMIT="false"
if [ -n "$PR_HEAD" ] && [ -n "$REVIEW_COMMIT" ] && [ "$PR_HEAD" != "$REVIEW_COMMIT" ]; then
  HAS_NEW_COMMIT="true"
fi

echo "PR #$PR: actionable=$ACTIONABLE majors=$MAJORS criticals=$CRITICALS new_commit=$HAS_NEW_COMMIT"

# Case 1: 0 actionable → approve
if [ "$ACTIONABLE" -eq 0 ]; then
  echo "No actionable — approving."
  gh pr review "$PR" --repo "$REPO" --approve \
    --body "Auto-approved: no actionable findings."
  exit 0
fi

# Case 2: major/critical or > 2 trivial → reject
if [ "$MAJORS" -gt 0 ] || [ "$CRITICALS" -gt 0 ] || [ "$ACTIONABLE" -gt 2 ]; then
  echo "Unacceptable findings — rejecting."
  gh pr review "$PR" --repo "$REPO" --request-changes \
    --body "Auto-rejected: major/critical=$((MAJORS + CRITICALS)) actionable=$ACTIONABLE. Fix and close-reopen."
  exit 1
fi

# Case 3: ≤ 2 trivial + fix commit → approve
if [ "$HAS_NEW_COMMIT" = "true" ]; then
  echo "Fix commit pushed after trivial-only review — approving."
  gh pr review "$PR" --repo "$REPO" --approve \
    --body "Auto-approved: fix commit pushed, $ACTIONABLE trivial/minor finding(s)."
  exit 0
fi

# Case 4: ≤ 2 trivial, no fix commit → wait
echo "≤ 2 trivial/minor but no fix commit yet — waiting."
exit 0
