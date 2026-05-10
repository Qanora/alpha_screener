#!/bin/bash
# Check if CodeRabbit review on a PR meets auto-approve criteria.
# Usage: ./auto-approve.sh <pr_number>
# Exit: 0=approved, 1=needs human attention
#
# Auto-approve rules:
#   1. 0 actionable findings → approve
#   2. major=0 AND critical=0 AND trivial/minor ≤ 2 AND new fix commit after review → approve
#   3. Everything else → skip

set -euo pipefail

PR="${1:?Usage: $0 <pr_number>}"
REPO="${REPO:-Qanora/alpha_screener}"

# Get latest CodeRabbit review
REVIEW=$(gh pr view "$PR" --repo "$REPO" --json reviews --jq '
  [.reviews[] | select(.author.login | test("coderabbit"))]
  | last // empty
')

if [ -z "$REVIEW" ]; then
  echo "No CodeRabbit review found on PR #$PR"
  exit 0
fi

REVIEW_STATE=$(echo "$REVIEW" | jq -r '.state')
REVIEW_BODY=$(echo "$REVIEW" | jq -r '.body')
REVIEW_COMMIT=$(echo "$REVIEW" | jq -r '.commit_id // ""')

if [ "$REVIEW_STATE" = "APPROVED" ]; then
  echo "Already approved."
  exit 0
fi

# Parse actionable count and severity
ACTIONABLE=$(echo "$REVIEW_BODY" | sed -n 's/.*\*\*Actionable comments posted: \([0-9]\{1,\}\).*/\1/p')
ACTIONABLE=${ACTIONABLE:-0}

MAJORS=$(echo "$REVIEW_BODY" | grep -c '"severity":"major"' || echo 0)
CRITICALS=$(echo "$REVIEW_BODY" | grep -c '"severity":"critical"' || echo 0)

# Check if a new commit was pushed after this review
PR_HEAD=$(gh pr view "$PR" --repo "$REPO" --json commits --jq '.commits[-1].oid // ""')
HAS_NEW_COMMIT="false"
if [ -n "$PR_HEAD" ] && [ -n "$REVIEW_COMMIT" ] && [ "$PR_HEAD" != "$REVIEW_COMMIT" ]; then
  HAS_NEW_COMMIT="true"
fi

echo "PR #$PR: actionable=$ACTIONABLE majors=$MAJORS criticals=$CRITICALS new_commit=$HAS_NEW_COMMIT"

# Case 1: 0 actionable → approve
if [ "$ACTIONABLE" -eq 0 ]; then
  echo "No actionable comments — approving."
  gh pr review "$PR" --repo "$REPO" --approve \
    --body "Auto-approved: no actionable findings."
  exit 0
fi

# Case 2: has major or critical → skip
if [ "$MAJORS" -gt 0 ] || [ "$CRITICALS" -gt 0 ]; then
  echo "Has major/critical findings — skipping."
  exit 1
fi

# Case 3: > 2 trivial/minor → skip
if [ "$ACTIONABLE" -gt 2 ]; then
  echo "$ACTIONABLE trivial/minor findings (> 2) — skipping."
  exit 1
fi

# Case 4: ≤ 2 trivial/minor + fix commit → approve
if [ "$HAS_NEW_COMMIT" = "true" ]; then
  echo "≤ 2 trivial/minor + fix commit → approving."
  gh pr review "$PR" --repo "$REPO" --approve \
    --body "Auto-approved: $ACTIONABLE trivial/minor finding(s), fix commit pushed."
  exit 0
fi

# Case 5: ≤ 2 trivial/minor, no fix commit → skip
echo "≤ 2 trivial/minor but no fix commit — skipping (waiting for fix)."
exit 0
