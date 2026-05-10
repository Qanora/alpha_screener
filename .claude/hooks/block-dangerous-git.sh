#!/bin/bash

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

# Patterns that are always blocked
DANGEROUS_PATTERNS=(
  "git reset --hard"
  "git clean -fd"
  "git clean -f"
  "git branch -D"
  "git checkout \."
  "git restore \."
  "push --force"
  "reset --hard"
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
  if echo "$COMMAND" | grep -qE "$pattern"; then
    echo "BLOCKED: '$COMMAND' matches dangerous pattern '$pattern'. The user has prevented you from doing this." >&2
    exit 2
  fi
done

# git push: allow non-master branches, block master/main and bare push
if echo "$COMMAND" | grep -qE '^git push'; then
  # Block push --force to any branch
  if echo "$COMMAND" | grep -qE 'push.*--force'; then
    echo "BLOCKED: git push --force is forbidden." >&2
    exit 2
  fi

  # Only allow: git push origin <branch> (with optional -u flag)
  if ! echo "$COMMAND" | grep -qE '^git push( -u)? origin [^[:space:]:]+$'; then
    echo "BLOCKED: git push without explicit branch. Use 'git push origin <branch>'." >&2
    exit 2
  fi

  # Block push to master or main (including refs/heads/* and HEAD:* refspecs)
  if echo "$COMMAND" | grep -qE '^git push( -u)? origin (master|main|refs/heads/master|refs/heads/main|([^[:space:]:]+:)?(master|main))$'; then
    echo "BLOCKED: git push to master/main is forbidden. Use feature branches + PR." >&2
    exit 2
  fi
fi

exit 0
