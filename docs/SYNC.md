# 本地与服务器通过 GitHub 协作

本地维护方案和脚本 → push 到 GitHub 私有仓库 → 服务器 pull 后运行 → 服务器把小型摘要 push 到独立分支 → 本地读取。

## 本地更新实践方案

只在独立的 `alignment-preparedness-lab` 目录运行：

```bash
git status
git pull --ff-only
git add docs configs scripts
git diff --cached --stat
git commit -m "Update experiment plan"
git push origin main
```

提交前查看实际修改；按需指定文件。学习软件的修改不进入此仓库。

## 服务器获取更新

```bash
git switch main
git pull --ff-only
git log -1 --oneline
```

如果有未提交修改，先保存到自己的分支；不要使用 `reset --hard` 清理。已经运行的训练保持原版本，下一次任务才使用新代码。

## 服务器回传第一份检查摘要

只在仓库配置 Git 作者信息，避免修改共享服务器全局配置：

```bash
git config user.name "TingxuanHan"
git config user.email "TingxuanHan@users.noreply.github.com"
git switch -c "codex/server-check-$(date +%Y%m%d-%H%M%S)"
cp runs/preflight-gpu.json handoff/preflight-gpu.json
git add handoff/preflight-gpu.json
git diff --cached
git commit -m "Record server GPU preflight"
git push -u origin HEAD
```

推送前确认报告仅含必要的版本、设备规格、检查结果，不含凭据、内部数据或敏感主机信息。完整训练日志和输出保留服务器，按 `templates/RUN_RECORD.md` 摘要。

推送后把分支名发回当前对话，本地可通过 gh / git 读取；不需要服务器运行 Codex。日后每次回传用新的分支，完成后 `git switch main`。

## GitHub 浏览器也能使用

在已登录浏览器打开仓库可以直接读所有 Markdown。修改文档后记得在另一台机器 pull。下载 ZIP 仅是副本，不自动同步；正常协作使用 clone。

此仓库私有，默认不授予第三方访问。不要上传 API key、`.env`、模型、数据集、客户资料和虚拟环境。
