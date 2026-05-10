## Agent skills

### Issue tracker

Issues live as GitHub Issues on Qanora/alpha_screener, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

All five triage roles use the default label names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Workflow

- 每个 commit 必须在 message 中关联 issue (e.g. `#3`, `closes #3`)
- `git push` 只允许 `git push [-u] origin feature/<name>`（master/main/force push 被 guardrails 拦截）
- 所有代码变更走 feature 分支 → PR → squash merge 流程
- **提交 PR、查看 CI、修复 AI 评审意见 必须使用 `/pr-workflow` skill**
