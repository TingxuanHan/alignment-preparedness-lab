# Alignment & Preparedness Lab

一个月的安全与对齐求职实践：**安全 SFT / DPO 的泛化与正常任务代价**。

这是独立的实践方案仓库，包含理论笔记、日程、实验协议、配置草案和 Ubuntu/Debian 初始化工具。服务器通过 GitHub 获取内容，无需 Codex，也无需连接本地电脑。

## 先从这里开始

1. [服务器配置指南](docs/SERVER_SETUP.md)：登录 GitHub、克隆、检查环境、安装依赖、第一次模型推理。
2. [30 天计划](docs/PLAN.md)：每天 3–4 小时，明确学习内容、操作和验收产物。
3. [实验协议](docs/PROTOCOL.md)：基线、数据划分、指标、消融、32B 验证及证据边界。
4. [32 节基础知识](docs/FOUNDATIONS.md)：训练、偏好、RL、安全评测、红队、Agent、监控与风险准备。
5. [配置说明](docs/CONFIGURATION.md)：H20 分配、参数、依赖版本、常见错误。
6. [两台机器同步](docs/SYNC.md)：本地推方案，服务器拉取；小型结果通过独立分支回传。

## 当前交付范围

| 内容 | 状态 |
| --- | --- |
| 计划、理论、实验口径、结果模板 | 已提供 |
| 无依赖 CPU 公式检查、计划配置检查、服务器预检 | 已提供可运行脚本 |
| 独立虚拟环境及固定版本依赖安装 | 已提供脚本，需在目标 Ubuntu 服务器执行 |
| 单张 H20 的 Qwen3-8B 推理与版本记录 | 已提供入口，目标 GPU 尚未实测 |
| 数据清洗、正式 SFT / DPO 训练、官方基准评测适配 | 后续按日程实现；当前配置为草案 |
| 安全提升、训练耗时、32B 结论 | 尚无实验结果 |

本次先把实践方案和配置入口准备好。仓库中的 CPU 数学例子不属于模型实验结果，`configs/experiment.json` 不是可直接交给 TRL 的训练参数文件。

## 配置完成后的第一条命令

在仓库根目录：

```bash
python3 scripts/doctor.py
bash scripts/bootstrap.sh
source .venv/bin/activate
python scripts/cpu_checks.py
python scripts/validate_plan.py
```

完成 GPU 依赖安装后再执行模型检查，详见服务器指南。脚本不会自动启动训练、修改驱动、开启服务端口或调用付费 API。

## 最终作品目标

8B：原模型 / 安全 SFT / 安全 DPO＋一个消融＋持出攻击。
32B：原模型＋在开发集上选定的方案。

交付可复现仓库、4–6 页报告、8–10 页汇报、逐条评分清单和 3–5 个失败案例。所有指标只填写真实测量值。可选 Agent 拓展在主线完成后安排。

## 数据位置

模型和数据存放在服务器的 `LAB_DATA_ROOT`；环境变量参考 `env.example`。本仓库只同步代码、配置和文档。经检查的小型摘要放 `handoff/`；模型权重、原始数据、密钥和虚拟环境不进入 Git。
