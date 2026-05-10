## Agent skills

### Issue tracker

Issues live as GitHub Issues on Qanora/alpha_screener, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

All five triage roles use the default label names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Workflow

- 每个 commit 必须在 message 中关联 issue (e.g. `#3`, `closes #3`)
- `git push` 被 git guardrails 拦截，推送用 `bash scripts/push-branch.sh`
- 所有代码变更走 feature 分支 → PR → squash merge 流程

### Feature branch workflow

```bash
git checkout -b feature/<name>         # 从 master 创建分支
# ... 开发、commit（每个 commit 关联 issue）...
bash scripts/push-branch.sh            # 推送分支到 remote
gh pr create --title "..." --body "..." --base master
gh pr merge <N> --squash --delete-branch  # 合并后自动删除远程分支
git checkout master && git pull origin master
```
