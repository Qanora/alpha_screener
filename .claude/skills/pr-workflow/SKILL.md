---
name: pr-workflow
description: PR 创建→CI 调试→AI 评审→修复→Auto-Merge 完整循环。使用当用户要求创建 PR、检查 CI、修复 AI 评审意见、处理 guardrails 冲突、或 merge PR 时。
---

# PR Workflow

## 标准流程

```bash
# 1. 本地 CLI 评审（无 PR 开销）
cr review --agent --base origin/master
# → 修复所有 finding → 再评 → 0 findings

# 2. 推 PR
git push origin feature/<name>
gh pr create --title "..." --body "..." --base master

# 3. Monitor（固定脚本，不再手写 jq）
bash scripts/watch-pr.sh <N>
```

## Git 约束

- commit 必须关联 `#N`
- `git push` 只允许 `git push origin feature/<name>`
- feature 分支 → PR → squash merge
- 禁止 force push；不要 `&&` 连接 push 和建 PR

## 循环模式

**Bot CHANGES_REQUESTED** → `autofix` skill 读 threads → 修复 → 关旧开新:

```bash
bash scripts/close-reopen.sh <old-N> <old-branch>
bash scripts/watch-pr.sh <new-N>
```

**与 master 冲突** → 同上（新分支 rebase 到 master 后再 squash）。

**Bot 无响应** → 等。若超 20 分钟仍是 REVIEW_REQUIRED → 关旧开新。

## 速查

| 症状 | 修复 |
|------|------|
| push 被拦 | 分支名 `feature/` 开头 |
| CI test ImportError | pyproject.toml 动态提取 deps |
| 网络测试 CI 失败 | `@pytest.mark.network` |
| Bot 不 approve | `.coderabbit.yaml`: `assertive` + `request_changes_workflow` |

见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
