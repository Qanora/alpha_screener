#!/bin/bash
# Close old PR and reopen with squashed 1-commit fresh start.
# Usage: ./close-reopen.sh <old_pr_number> <old_branch> [new_suffix]
set -euo pipefail

OLD_PR="${1:?}"
OLD_BRANCH="${2:?}"

# 检查工作区是否干净（含 untracked 文件），避免未提交变更被 squash 丢失
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "ERROR: working tree is dirty — commit or stash changes first"
  echo "$DIRTY"
  exit 1
fi

# POSIX-safe version extraction: extract trailing v<N>, default to 0
CURRENT_VERSION=$(echo "$OLD_BRANCH" | sed -n 's/.*v\([0-9]\{1,\}\)$/\1/p')
CURRENT_VERSION=${CURRENT_VERSION:-0}
NEXT_VERSION=$((CURRENT_VERSION + 1))
SUFFIX="${3:-$NEXT_VERSION}"

if echo "$OLD_BRANCH" | grep -q 'v[0-9]\{1,\}$'; then
  NEW_BRANCH=$(echo "$OLD_BRANCH" | sed "s/v[0-9]\{1,\}$/v$SUFFIX/")
else
  NEW_BRANCH="${OLD_BRANCH}-v${SUFFIX}"
fi

echo "Creating $NEW_BRANCH from squashed $OLD_BRANCH..."

# 直接从 origin/master 创建新分支，避免移动本地 master
git fetch origin master
git checkout -b "$NEW_BRANCH" origin/master
git merge --squash "$OLD_BRANCH"

# 恢复 guardrails hook（squash merge 可能丢失此文件的最新版本）
git checkout "$OLD_BRANCH" -- .claude/hooks/block-dangerous-git.sh 2>/dev/null || true

COMMIT_MSG=$(git log -1 --pretty=format:'%s' "$OLD_BRANCH")
git commit -m "$COMMIT_MSG"

echo "Pushing $NEW_BRANCH..."
git push origin "$NEW_BRANCH"

# 先创建新 PR，成功后再关闭旧 PR 和删除旧分支
TITLE=$(git log -1 --pretty=format:'%s')
echo "Creating new PR..."
gh pr create --repo Qanora/alpha_screener --title "$TITLE" --body "1 commit." --base master

echo "Closing old PR #$OLD_PR..."
gh pr close "$OLD_PR" --repo Qanora/alpha_screener

echo "Deleting old remote branch $OLD_BRANCH..."
git push origin --delete "$OLD_BRANCH" 2>/dev/null || echo "  (already deleted or never pushed)"

echo "Deleting old local branch $OLD_BRANCH..."
git branch -D "$OLD_BRANCH" 2>/dev/null || echo "  (already deleted)"

echo "Done. New PR created."
