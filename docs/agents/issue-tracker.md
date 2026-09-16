# Issue tracker: GitHub

本仓库的 issues 和 specs 存放在
[kaiyintj/DOG 的 GitHub Issues](https://github.com/kaiyintj/DOG/issues)。
相关操作使用 `gh` CLI；在本仓库内运行，或显式指定 `--repo kaiyintj/DOG`。

## Conventions

- 创建：`gh issue create --title "..." --body-file <path>`。
- 查看：`gh issue view <number> --comments`；需要结构化字段时使用 `--json`，包含 labels。
- 列出：`gh issue list --state open`，按需使用状态和标签过滤。
- 评论：`gh issue comment <number> --body-file <path>`。
- 添加或删除标签：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`。
- 关闭：`gh issue close <number>`。

多行正文先保存为文本文件，再通过 `--body-file` 传入，保留实际换行。
这些命令描述操作方式；是否执行写入遵循当前用户授权。

## Pull requests as a triage surface

PRs as a request surface: no.

## When a skill says "publish to the issue tracker"

创建一个 GitHub issue。

## When a skill says "fetch the relevant ticket"

运行 `gh issue view <number> --comments`，同时获取需要的标签信息。
