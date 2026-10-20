# 验证记录

2026-10-20，在本地 Windows 上完成：

- `python scripts/cpu_checks.py`：SFT mask、DPO 数值、Wilson 区间、模拟授权规则通过。
- `python scripts/validate_plan.py`：草案结构通过；6 项版本/清单字段尚未填写，明确报告 NOT READY。
- `python -m unittest discover -s tests -v`：5 项测试通过。
- 所有 Python 脚本通过语法编译。
- `bash -n scripts/bootstrap.sh`：Bash 语法检查通过。
- `doctor.py` 的无 GPU 检查路径实际执行并生成 JSON；未安装训练依赖的情况可正确报告。

目标 Ubuntu/Debian 上尚未验证：系统工具安装、设备授权、CUDA wheel 与驱动兼容、GPU 矩阵检查、Qwen3 模型下载/推理、训练及基准评测。
`model_smoke.py` 已检查语法和官方接口文档，没有实际下载模型或占用目标 H20。

这是配置准备记录，不是模型安全结果。服务器首次执行后，将实际报告提交至 `handoff/` 并保留环境版本。
