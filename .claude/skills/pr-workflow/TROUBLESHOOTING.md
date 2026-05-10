# PR Workflow 故障排查

## CI 快速修复

- **gitleaks**: `env.GITHUB_TOKEN` 必须传入 action
- **mypy INTERNAL ERROR**: mypy 2.x 引入了新的默认 flag 行为 (如 `--local-partial-types`, `--strict-bytes`)，代码可能需要适配，详见 [mypy changelog](https://mypy.readthedocs.io/en/stable/changelog.html) (2026-05)。如遇兼容问题可临时 pin `mypy<2`
- **test ModuleNotFoundError**: 不要手写依赖列表——从 pyproject.toml 动态提取（需 Python 3.11+ 的 tomllib，在仓库根目录运行）：
  ```bash
  python3.11 -c "
import tomllib
with open('pyproject.toml','rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
print('\n'.join(deps))
" > deps.txt
  while read -r pkg; do pip install "$pkg"; done < deps.txt
  ```
  (旧版 Python 可用 `pip install tomli && python -c "import tomli; ..."`)
  GNU 环境可选：`xargs -d '\n' -r pip install < deps.txt`
- **test 网络超时**: Stooq/yfinance API 不可达 → `@pytest.mark.network` + conftest CI skip

## Guardrails Hook

push 被误拦的原因：
1. 分支名不是 `feature/` 前缀
2. `2>&1` 或 `&&` 在命令中 → 分开执行，不要组合

常见误解：
- commit message 中含 "git push" 文本 → hook 只检查 `^git push` 开头的命令，不会被误拦

## .coderabbit.yaml 配置

```yaml
language: zh-CN
reviews:
  profile: assertive
  request_changes_workflow: true
  high_level_summary: true
  auto_review:
    enabled: true
```

注意：`auto_approve` 不存在，`summarization` 不存在。摘要行为由 `high_level_summary` (及 `high_level_summary_instructions`) 控制。`request_changes_workflow` 和 `auto_review` 有效。`profile` 仅接受 `chill`/`assertive`。详见 [CodeRabbit 配置文档](https://docs.coderabbit.ai/reference/configuration) (2026-05)。

## Bot 交互

以下方式经实测不可靠（2026-05），CHANGES_REQUESTED 后必须用 `close-reopen.sh`：
- `@coderabbitai full review` — 不会触发第二次评审
- dismiss via API + retrigger — bot 不响应

**Bot 每 PR 只做一次有效评审。** CHANGES_REQUESTED 后 → 修代码 → `close-reopen.sh`。

## 关旧开新脚本

```bash
bash scripts/close-reopen.sh <old-N> <old-branch> [new-suffix]
# 自动: close PR → squash → new branch → push → create PR → delete old branch
# 可选第三个参数指定新分支版本后缀，默认自动递增
```
