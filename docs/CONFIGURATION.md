# 配置、版本与排错

## 已提供与待实现

当前仓库用于先准备实践方案和服务器配置。CPU 检查、服务器预检、模型推理检查可运行。`configs/experiment.json` 是实验协议草案，尚无正式 SFT / DPO trainer 或基准适配入口；后续按 30 天计划实现并验证。

## 固定候选环境

| 包 | 版本 | 作用 |
| --- | --- | --- |
| Python | 推荐 3.10–3.12 | 独立环境 |
| torch | 2.8.0，cu126 wheel | GPU 张量与模型 |
| transformers | 4.57.1 | Qwen3 加载和模板 |
| trl | 0.24.0 | 后续 SFT / DPO |
| peft | 0.17.1 | LoRA |
| datasets | 4.3.0 | 数据读写 |
| accelerate | 1.11.0 | 多卡训练基础 |
| huggingface-hub | 0.35.3 | 模型版本和下载 |

这是可追溯的起点，不是已在目标服务器验证的组合，也不声明是最新版本。安装后 `pip check` 检查声明依赖，CUDA 预检检查实际 GPU 运算；真实模型与训练仍需分别验证。实际间接依赖保存在 `runs/requirements-resolved.txt`。

采用独立 venv，不安装 vision/audio、FlashAttention、DeepSpeed 或 vLLM。后续确有需求再单独加依赖并记录版本，避免在第一天引入不必要的编译问题。

## 实验参数起点

| 参数 | 起点 | 需要检查 |
| --- | --- | --- |
| 主模型 | Qwen3-8B | commit、tokenizer、非思考模板 |
| 核验模型 | Qwen3-32B | 仅关键对照，单独测显存 |
| dtype | BF16 | GPU 运算、加载成功 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | target modules 和可训练参数 |
| 最大长度 | 2048；调试 512 | 截断比例与监督目标是否保留 |
| 每卡 batch | 1 | 峰值显存 |
| 梯度累积 | 16 | 有效 batch 随 GPU 数变化 |
| SFT / DPO 学习率 | 1e-4 / 5e-6 | 开发集选择，非最佳值承诺 |
| DPO beta | 0.1 | 冻结参考、长度偏差与 KL |
| 初始 epochs | 1 | 有效 Token、训练步数 |
| 正式种子 | 17、29、43 | 先一个种子，关键对照再重复 |

单卡 8B 调试之后，可将 H20 分为 2 卡 SFT、2 卡 DPO、2 卡评测、2 卡复核。实际根据显存与时间测量调整。DDP 每卡仍保留模型副本；32B 若需要分片，必须另行配置 FSDP / ZeRO 并验证。3090 不与 H20 混为同一同步训练组。

## 常见问题

| 现象 | 先做什么 |
| --- | --- |
| 私有仓库提示不存在 / 404 | `gh auth status` 确认登录账号与仓库权限 |
| 浏览器不能在服务器打开 | 本地浏览器完成 GitHub device code 授权 |
| `No module named venv` | 使用正确 Python，安装对应 python3-venv 或现有 conda |
| `torch.cuda.is_available()` 为 False | 检查驱动、容器、调度分配和 GPU wheel，保留预检错误 |
| OOM | 缩短长度、减 batch，检查其他占用与梯度检查点；先记录峰值 |
| 模型下载失败 | 区分 GitHub、PyPI 与 Hugging Face 连通性；三者互不保证 |
| `KeyError: qwen3` | 确认已激活项目 venv 和固定 Transformers 版本 |
| `git pull --ff-only` 拒绝 | 先 `git status`，把自己的改动提交到结果分支，不强制覆盖 |
| NCCL / 多卡异常 | 先保留单卡可运行配置，再查设备拓扑和多卡通信日志 |

正式训练前还需实现：数据筛选、模板与 mask 验证、独立评测器、结果 schema。上述超参数不替代这些工作。

参考：[TRL 0.24 SFT](https://huggingface.co/docs/trl/v0.24.0/sft_trainer)、[TRL 0.24 DPO](https://huggingface.co/docs/trl/v0.24.0/dpo_trainer)、[PEFT](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)。
