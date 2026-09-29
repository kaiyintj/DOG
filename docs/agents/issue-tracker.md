# Issue tracker：GitHub

仓库为 [kaiyintj/DOG](https://github.com/kaiyintj/DOG/issues)，与本地 origin 一致。
在仓库内执行 `gh`，或显式指定 `--repo kaiyintj/DOG`。

- 读取：`gh issue view <number> --comments`；列出：`gh issue list --state open`。
- 创建：`gh issue create --title "..." --body-file <path>`。
- 评论：`gh issue comment <number> --body-file <path>`。
- 标签：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`；定义见 [triage-labels](triage-labels.md)。
- 关闭：`gh issue close <number>`。

多行正文使用文件，保留换行。写入操作须有用户授权；技能提及 publish 不构成授权。
任务来源使用 Issue，不将 PR 视为任务分派入口。
