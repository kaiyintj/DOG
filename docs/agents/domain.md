# Domain Docs

本仓库采用 single-context 布局：根目录 `CONTEXT.md` 定义领域术语，
`docs/adr/` 存放架构决策。

## Before exploring, read these

- 根目录的 `CONTEXT.md`（如果存在）。
- `docs/adr/` 中与当前工作相关的 ADR（如果存在）。

文件不存在时静默继续，不把缺失本身标记为问题或提前建议创建。
当领域术语或决策实际确定时，由 `domain-modeling` 按需创建。
现有 `AGENTS.md` 对项目状态、实机和仿真文档的读取规则继续适用。

## Use the glossary's vocabulary

命名领域概念时使用 `CONTEXT.md` 中的术语，包括 Issue 标题、设计建议和测试名称。
需要的概念尚未定义时，先确认是否确为项目概念；确有缺口则记录供领域建模处理。

## Flag ADR conflicts

输出与已有 ADR 冲突时，指出对应 ADR 和重新讨论的理由，保留决策可追溯性。
