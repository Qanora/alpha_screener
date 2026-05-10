---
name: pr-workflow
description: PR 创建→CI 调试→AI 评审→修复→Auto-Merge 完整循环。使用当用户要求创建 PR、检查 CI、修复 AI 评审意见、处理 guardrails 冲突、或 merge PR 时。
---

# PR Workflow

## 标准流程

### 1. 本地验证（CRITICAL：修到干净再推）

在 push 之前用本地 CLI 评审反复修到干净，**不要跟 GitHub bot 跳舞**。

LLM 评审每次从不同视角审视，可能分多轮发现新问题。实用策略：

- **第 1 轮**: `cr review --agent --base origin/master` → 修所有 finding
- **第 2 轮**: 再跑 → 修所有 finding
- **第 3 轮**: 再跑 → 若只剩 1-2 个 trivial 且无可修复的 actual bug → 可推
- **最多 3-4 轮**，之后即使有剩余 trivial 也直接推（diminishing returns）

```bash
# 循环修复
cr review --agent --base origin/master
# → 有 finding 就修 → git add -A && git commit --amend --no-edit → 再跑
```

**停止规则**：0 major/critical + ≤ 2 trivial/minor → 可推。

### 2. 推送 + 创建 PR

```bash
git push origin feature/<name>
gh pr create --title "..." --body "..." --base master
```

### 3. Monitor

```bash
bash scripts/watch-pr.sh <N>
```

## Git 约束

- commit 必须关联 `#N`
- `git push` 只允许 `git push origin feature/<name>`
- feature 分支 → PR → squash merge
- 禁止 force push；不要 `&&` 连接 push 和建 PR

## 循环模式

### Bot 评审结果分级处理

拿到 CodeRabbit review 后先看 severity 分布，不是无脑关旧开新：

| 情况 | 处理 |
|------|------|
| 0 findings | 等 CI → auto-merge（bot 自动 approve） |
| major=0 AND critical=0 AND trivial/minor ≤ 2 | 修 → commit → push 同分支 → bot 自动 approve → merge |
| ≥ 1 major/critical 或 trivial/minor > 2 | 修 → amend → `close-reopen.sh` |

```bash
# ≤ 2 trivial/minor：正常 commit 推同分支，不用关旧开新，bot 自动 approve
git add -A && git commit -m "fix: address trivial review findings (#N)"
git push origin feature/<name>

# major/critical > 0 或 trivial/minor > 2：修 → amend → 关旧开新
bash scripts/close-reopen.sh <old-N> <old-branch>
bash scripts/watch-pr.sh <new-N>
```

**与 master 冲突** → 同上（新分支从 master 创建后 squash merge 旧分支）。

**Bot 无响应** → 等。若超 20 分钟仍是 REVIEW_REQUIRED → 关旧开新。

## 速查

| 症状 | 修复 |
|------|------|
| push 被拦 | 分支名 `feature/` 开头 |
| CI test ImportError | pyproject.toml 动态提取 deps |
| 网络测试 CI 失败 | `@pytest.mark.network` |
| Bot 不 approve | `.coderabbit.yaml`: `assertive` + `request_changes_workflow` |
| Bot trivial review 阻塞 merge | Auto-Approve workflow 自动 approve（`.github/workflows/auto-approve.yml`） |

见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
