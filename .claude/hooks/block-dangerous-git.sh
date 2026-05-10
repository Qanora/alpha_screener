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

# git push: allow feature branches, block master/main and bare push
# Strip trailing redirects before checking
PUSH_CMD=$(echo "$COMMAND" | sed 's/ *2>&1 *$//; s/ *>[^ ]* *$//')
if echo "$PUSH_CMD" | grep -qE '(^|[[:space:]])git push'; then
  if echo "$PUSH_CMD" | grep -qE 'git push.*(--force|-f)'; then
    echo "BLOCKED: git push --force is forbidden." >&2
    exit 2
  fi

  if ! echo "$PUSH_CMD" | grep -qE '(^|[[:space:]])git push( -u)? origin feature/[^[:space:]:]+$'; then
    echo "BLOCKED: only 'git push origin feature/<name>' is allowed." >&2
    exit 2
  fi

  if echo "$PUSH_CMD" | grep -qE '(^|[[:space:]])git push( -u)? origin (master|main|refs/heads/master|refs/heads/main|([^[:space:]:]+:)?(master|main))$'; then
    echo "BLOCKED: git push to master/main is forbidden. Use feature branches + PR." >&2
    exit 2
  fi
fi

exit 0
