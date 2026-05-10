## Agent skills

### Issue tracker

Issues live as GitHub Issues on Qanora/alpha_screener, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

All five triage roles use the default label names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Workflow

- 每个 commit 必须在 message 中关联 issue (e.g. `#3`, `closes #3`)
- `git push` 允许推送到非 master 分支（master 被 guardrails 保护）
- 所有代码变更走 feature 分支 → PR → squash merge 流程

### Feature branch workflow

```bash
git checkout -b feature/<name>         # 从 master 创建分支
# ... 开发、commit（每个 commit 关联 issue）...
git push -u origin <branch>            # 推送分支到 remote
gh pr create --title "..." --body "..." --base master
gh pr merge <N> --squash --delete-branch  # 合并后自动删除远程分支
git checkout master && git pull origin master
```

### 处理冲突

当 PR 分支与 master 冲突时，**禁止 force push**（被 guardrails 拦截）。用新分支解决：

```bash
# 1. rebase 到最新 master
git fetch origin
git rebase origin/master
# 解决冲突...

# 2. 从 rebase 后的 commit 创建新分支
git checkout -b feature/<name>-v2

# 3. 推送新分支（无需 --force）
git push -u origin feature/<name>-v2

# 4. 关旧 PR，建新 PR
gh pr close <old-N>
gh pr create --title "..." --body "..." --base master
```

### Guardrails

- `git push` 被 `.claude/hooks/block-dangerous-git.sh` 拦截：只允许 `git push origin <feature-branch>`，禁止 master/main、force push
- 遇到冲突时用新分支方式，重启 PR```
