# Ubuntu / Debian：从 GitHub 配置实践环境

两台电脑只需分别能访问 GitHub。服务器不需要 Codex，不需要开放端口，也无需连接你的本地电脑。以下命令在服务器终端执行；若通过跳板机进入，进入实际有 H20 的机器后再执行 GPU 部分。

## 1. 安装或确认基础工具

```bash
git --version
python3 --version
gh --version
nvidia-smi
```

缺少 Git、gh、venv 或 tmux 且拥有 sudo 时：

```bash
sudo apt update
sudo apt install -y git gh python3 python3-venv tmux
```

没有 sudo：使用服务器现有 Git/Python/conda，或让管理员安装以上工具。gh 的用户目录安装可参考 [官方 Linux 安装方式](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)。不会为本项目修改 NVIDIA 驱动或共享 CUDA。

Python 推荐 3.10–3.12；若系统 Python 版本不合适，可用已有 conda 创建 `python=3.11` 环境，再让 `LAB_PYTHON` 指向其中的解释器。

## 2. 让服务器登录 GitHub

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

终端会显示一次性授权代码和 `https://github.com/login/device`。服务器没有浏览器时，在本地电脑的浏览器打开该地址，登录拥有私有仓库权限的 `TingxuanHan` 账号，输入代码并授权。服务器终端等待完成即可。设备代码只用于本次授权，不提交到仓库。

这只要求两端各自访问 GitHub，不要求两台机器能互连。不要复制本地 `gh` 凭据文件到服务器。

若服务器为多人共用账号，优先改用仅授权本仓库的细粒度令牌或 [deploy key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)。只拉取可用只读权限，回传摘要才需要写权限；认证工具在服务器保存凭据，`.env` 不放 GitHub 凭据。

## 3. 克隆独立实践仓库

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
gh repo clone TingxuanHan/alignment-preparedness-lab
cd alignment-preparedness-lab
git remote -v
git log -1 --oneline
```

看到 README、docs、configs、scripts 即正确。这里没有 PyPractice 桌面程序。
已经克隆过时，进入已有目录执行 `git pull --ff-only`，不重复建立环境。

## 4. 先运行无需第三方库的检查

```bash
python3 scripts/doctor.py --output runs/preflight-before.json
bash scripts/bootstrap.sh
source .venv/bin/activate
python scripts/cpu_checks.py
python scripts/validate_plan.py
```

`bootstrap.sh` 默认只建立本项目 `.venv` 并运行 CPU 检查，不安装 CUDA、不下载模型、不启动训练。

如果 Python 需要指定：

```bash
LAB_PYTHON=/path/to/python3.11 bash scripts/bootstrap.sh
```

先将占位路径替换为真实解释器路径。已有 `.venv` 会保留。

## 5. 配置服务器数据目录

```bash
cp -n env.example .env
nano .env
```

把 `LAB_DATA_ROOT` 改成有足够空间且属于你的磁盘目录；默认是家目录下的独立目录。至少为 8B 权重、训练数据、适配器、缓存和日志预留空间；32B 还会增加几十 GB 的权重载荷，具体以实际下载为准。

```bash
source .env
mkdir -p "$LAB_DATA_ROOT"
df -h "$LAB_DATA_ROOT"
```

`.env` 被 Git 忽略。它只是环境配置脚本，只有主动 `source .env` 才会生效。

## 6. 安装 GPU 训练依赖

先看 `nvidia-smi`，确认驱动与服务器分配给你的空闲 GPU。`nvidia-smi` 顶部 CUDA Version 表示驱动支持能力，并不等于当前 Python 使用的 CUDA runtime。

```bash
bash scripts/bootstrap.sh --gpu
source .venv/bin/activate
python -m pip check
python scripts/doctor.py --require-gpu --output runs/preflight-gpu.json
```

脚本采用明确的候选环境：PyTorch 2.8.0 / CUDA 12.6 wheel、Transformers 4.57.1、TRL 0.24.0、PEFT 0.17.1。包版本与声明依赖已核对，尚未在你的 H20 上实测。先通过 CUDA 矩阵运算检查，再下载模型。

如果 CUDA 初始化失败，不改驱动、不连续重装。保留错误和预检报告，按 [配置与排错](CONFIGURATION.md) 检查驱动兼容性、调度分配、容器 GPU 透传和 PyTorch wheel。候选版本并非最新版本，选择它们是为了固定实验环境。

## 7. 第一张 H20 的模型推理检查

独占机器：把下面的 `0` 换成 `nvidia-smi` 中你确认可用的卡。调度系统已分配 GPU 时保留调度器的 `CUDA_VISIBLE_DEVICES`，不要覆盖分配。

```bash
export CUDA_VISIBLE_DEVICES=0
source .env
python scripts/model_smoke.py --model Qwen/Qwen3-8B
```

首次会下载模型，要求能访问 Hugging Face 的模型下载服务。脚本解析并记录模型 commit，用非思考模式、BF16 在单卡上做一条正常问题推理，记录结果、生成 Token、时间和峰值显存到 `runs/`。

这只是端到端加载与推理检查，不是安全评测，也不意味着训练显存已经验证。重跑同一版本可传 `--revision` 和报告中记录的 commit。

需要下载授权的模型时，使用服务器上的 `hf auth login`。当前公开模型通常不要求该步骤；不把 Token 写进命令参数、配置或提交记录。

## 8. 配置后该做什么

- 阅读 `docs/PLAN.md` 第 3–7 天和 `docs/PROTOCOL.md`。
- 配置草案在 `configs/experiment.json`，含 8B 三组、32B 复核和参数起点。
- 正式训练入口、数据清洗与基准适配按计划后续实现；不要将这份 JSON 直接传给 TRL。
- 将预检报告和模型检查的必要摘要放入 `handoff/`，按 `docs/SYNC.md` 回传。
- GitHub 不能直接访问时，先由服务器管理员解决允许的出网方式；不要关闭 TLS 验证。

## 9. 以后开终端的固定动作

```bash
cd "$HOME/projects/alignment-preparedness-lab"
git switch main
git pull --ff-only
source .venv/bin/activate
source .env
git log -1 --oneline
```

长任务可在 `tmux new -s alignment` 中执行；`Ctrl+B` 后按 `D` 离开，`tmux attach -t alignment` 返回。后台任务并不会因为 `git pull` 自动切换到新版本，开始运行时记录 commit。

认证说明依据 [gh auth login](https://cli.github.com/manual/gh_auth_login)，PyTorch wheel 依据 [官方历史版本](https://pytorch.org/get-started/previous-versions/)，模型模板依据 [Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-8B)。
