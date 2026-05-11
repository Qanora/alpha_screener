---
name: pr-workflow
description: PR 创建→CI 调试→AI 评审→修复→Auto-Merge 完整循环。使用当用户要求创建 PR、检查 CI、修复 AI 评审意见、处理 guardrails 冲突、或 merge PR 时。
---

# PR Workflow

## 标准流程

### 1. Simplify

```bash
if command -v claude-dp &> /dev/null; then
  claude-dp -p "请对当前分支 $(git branch --show-current) 相对于 origin/master 的所有 commit 改动执行 /simplify"
else
  echo "claude-dp 不可用，跳过简化步骤"
fi
```

对本次所有 commit 做代码精简、复用审查、质量优化。完成后 `git diff` 检查改动，确认无误后继续下一步。若 claude-dp 不可用则自动跳过。

### 2. 本地验证

```bash
cr review --agent --base origin/master
```

分级处理：

| findings | 处理 |
|----------|------|
| 0 major, 0 critical, **0 trivial/minor** | 直接推 PR |
| 0 major, 0 critical, **> 0 trivial/minor** | 修复 → `git commit --amend --no-edit` → 直接推 PR（不重跑 review） |
| ≥ 1 major/critical | 修复 → amend → **重跑 review**，直到只剩 trivial/minor |

### 3. 推送 + 创建 PR

```bash
git push origin feature/<name>
gh pr create --title "..." --body "..." --base master
```

### 4. Monitor

```bash
bash scripts/watch-pr.sh <N>
```

`watch-pr.sh` 持续轮询直到 PR 成功 **MERGED** 才退出（exit 0）。任何非零退出都需修复后重跑：

| 退出码 | 症状 | 修复 |
|--------|------|------|
| 1 | CI 失败 | 定位原因 → 修 → commit → push → 重跑 monitor |
| 2 | 超时 stuck | 检查 PR 状态，手动排查 |
| 3 | gh 命令失败 | 检查 GitHub 连接 / token |
| 4 | CHANGES_REQUESTED | 修 → amend → `close-reopen.sh` → 重跑 monitor |
| 5 | 缺少工具 | 安装 `gh` / `jq` |
| 6 | 参数错误 | 修正 timeout 参数 |

> **核心循环：monitor → 失败 → 修复 → 重跑 monitor → 直到 MERGED。**

### 5. 收尾

Monitor 以 `MERGED` 退出后，清理本地：

```bash
git checkout master && git pull origin master && git branch -D feature/<name>
```

> squash merge 后 `-d` 会失败（commit 未被直接合并），需用 `-D` 强制删除本地分支。

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
| 0 major/critical, > 0 trivial/minor | 修 → commit → push 同分支 → bot 自动 approve → merge |
| ≥ 1 major/critical | 修 → amend → `close-reopen.sh` |

```bash
# trivial/minor：正常 commit 推同分支，bot 自动 approve
git add -A && git commit -m "fix: address trivial review findings (#N)"
git push origin feature/<name>

# major/critical：修 → amend → 关旧开新
bash scripts/close-reopen.sh <old-N> <old-branch>
bash scripts/watch-pr.sh <new-N>
```

**与 master 冲突** → 同上（新分支从 master 创建后 squash merge 旧分支）。

**Bot 无响应** → 等 CI。若超 20 分钟仍是 REVIEW_REQUIRED（bot 不触发）→ 关旧开新。

**Auto-Approve Bot**（`.github/workflows/auto-approve.yml`）:
- 监听 CodeRabbit review 和 push 事件
- 符合 trivial 条件 → 自动 approve → auto-merge
- 有 major/critical → 自动 request-changes 阻塞 merge

## 速查

| 症状 | 修复 |
|------|------|
| push 被拦 | 分支名 `feature/` 开头 |
| CI test ImportError | pyproject.toml 动态提取 deps |
| 网络测试 CI 失败 | `@pytest.mark.network` |
| Bot 不 approve | `.coderabbit.yaml`: `assertive` + `request_changes_workflow` |
| Bot trivial review 阻塞 merge | Auto-Approve workflow 自动 approve（`.github/workflows/auto-approve.yml`） |

见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
