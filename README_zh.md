# NAC 中文运行说明

本文件是 `nac_project` 的中文补充 README，重点解释：

- 如何一键运行所有实验；
- 每个复现脚本分别跑什么；
- 什么时候应该单独运行某个脚本；
- 运行后的结果在哪里、如何核对。

项目主 README：`README.md`。  
主实验入口：`nac_fair_experiment.py`。  
快速单次运行入口：`run.sh` / `run.ps1`。  
一键运行入口：`run_all.sh`。

---

## 1. 项目在做什么

NAC（Nesterov Accelerated Counterattack）是对 TTC 测试时防御方法的改进：

- TTC 用标准 sign-PGD 在对抗图像上做反攻击；
- NAC 把 PGD 替换为 Nesterov 加速梯度，只在 `delta + mu * velocity` 位置计算梯度；
- 目标函数是最大化 `||f_I(x_adv + delta) - f_I(x_adv)||^2`；
- 主实验防御步数通常为 `K=2`，动量 `mu=0.9`，防御预算 `eps_ttc=4/255`。

论文主稿：`paper_backup/NAC_paper_sivp_8page.tex`。

---

## 2. 环境准备

推荐环境：

- Python 3.11
- PyTorch 2.5.1 + CUDA
- GPU 显存至少 8 GB
- 数据集和 CLIP 权重首次运行会自动下载

安装依赖：

```bash
cd F:/codex/NAC/nac_project
pip install -r requirements.txt
```

Windows 用户建议使用 Git Bash、WSL，或者直接运行 `run.ps1`。  
所有 `scripts/*.sh` 脚本需要 Bash 环境。

---

## 3. 一键运行所有实验

在 `nac_project` 目录下：

```bash
cd F:/codex/NAC/nac_project

# 完整版：按默认配置运行全部论文复现脚本
bash run_all.sh

# 单 seed 快速版：先把整个流程跑通
SEEDS=0 bash run_all.sh

# 跳过 AFT（因为 AFT 需要 TeCoA/FARE 外部权重）
bash run_all.sh --skip-aft

# 显存小时降低 batch size
BATCH_SIZE=8 bash run_all.sh
```

`run_all.sh` 会依次运行：

1. `scripts/reproduce_main.sh`
2. `scripts/reproduce_multi_eps.sh`
3. `scripts/reproduce_nstep.sh`
4. `scripts/reproduce_autoattack.sh`
5. `scripts/reproduce_cross_arch.sh`
6. `scripts/reproduce_ablation.sh`
7. `scripts/reproduce_mu_scan.sh`
8. `scripts/reproduce_aft.sh`
9. `scripts/compare_ttc_nac_doc.sh`

运行日志写入：`results/run_all.log`。  
每个子实验的结果写入对应的 `results/<实验目录>/`。

> 注意：默认 `SEEDS="0 1 2"` 是完整多 seed 模式，运行时间较长。  
> 建议先执行 `SEEDS=0 bash run_all.sh` 验证环境是否正常，再跑完整版。

---

## 4. 每个脚本运行什么

### 4.1 `scripts/reproduce_main.sh`

对应论文：Table 1，主结果。

运行内容：

- 攻击：PGD，`eps=1/255`，10 步；
- 防御：TTC 和 NAC，`K=2`；
- 数据集：CIFAR-10、CIFAR-100、STL-10、Flowers-102、DTD、ImageNet-100；
- 默认 `SEEDS=0`，可用 `SEEDS="0 1 2"` 扩展。

单独运行：

```bash
bash scripts/reproduce_main.sh
SEEDS="0 1 2" bash scripts/reproduce_main.sh
```

输出：

- `results/eps1/pgd_eps_1.0/ttc/seed_0.log`
- `results/eps1/pgd_eps_1.0/nac_m0.9/seed_0.log`
- 以及脚本顶层 `results/eps1/*.log`

适用场景：先看主结果时运行。

---

### 4.2 `scripts/reproduce_multi_eps.sh`

对应论文：Table 2，多攻击强度。

运行内容：

- 攻击强度：`eps=1/255, 2/255, 4/255`；
- 防御方法：TTC、NAC，`K=2`；
- 数据集：6 个数据集，ImageNet-100 只跑 seed 0；
- 默认 `SEEDS="0 1 2"`，可用 `SEEDS=0` 改为单 seed。

单独运行：

```bash
bash scripts/reproduce_multi_eps.sh
SEEDS=0 bash scripts/reproduce_multi_eps.sh
```

输出：`results/multi_eps/pgd_eps_*/...`

适用场景：需要看不同攻击强度下的增益趋势时运行。

---

### 4.3 `scripts/reproduce_nstep.sh`

对应论文：Table 3，N-step 扩展。

运行内容：

- 数据集：CIFAR-10、STL-10；
- 防御步数：`K=2`、`K=4`；
- 防御方法：TTC、NAC；
- 攻击：PGD，`eps=4/255`。

单独运行：

```bash
bash scripts/reproduce_nstep.sh
SEEDS=0 bash scripts/reproduce_nstep.sh
```

输出：`results/nstep/pgd_eps_4.0/...`

适用场景：需要证明 `K=2` 到 `K=4` 的收益变化时运行。

---

### 4.4 `scripts/reproduce_autoattack.sh`

对应论文：Table 4，AutoAttack 强攻击。

运行内容：

- 数据集：CIFAR-10；
- 攻击：AutoAttack，`eps=4/255`；
- 防御方法：TTC、NAC，`K=2`。

单独运行：

```bash
bash scripts/reproduce_autoattack.sh
SEEDS=0 bash scripts/reproduce_autoattack.sh
```

输出：`results/autoattack/autoattack_eps_4.0/...`

注意：需要先安装 `autoattack`。

---

### 4.5 `scripts/reproduce_cross_arch.sh`

对应论文：Table 5，跨架构实验。

运行内容：

- 数据集：CIFAR-10；
- 模型：CLIP ViT-B/32、ViT-B/16；
- 防御方法：TTC、NAC，`K=2`。

单独运行：

```bash
bash scripts/reproduce_cross_arch.sh
SEEDS=0 bash scripts/reproduce_cross_arch.sh
```

输出：`results/cross_arch/pgd_eps_4.0/...`

适用场景：说明方法不只在默认 ViT-B/32 上有效时运行。

---

### 4.6 `scripts/reproduce_ablation.sh`

对应论文：Table 7，消融实验。

运行内容：

- 数据集：CIFAR-10、STL-10；
- 方法：TTC、standard momentum、NAC；
- 攻击：PGD，`eps=4/255`，`K=2`。

单独运行：

```bash
bash scripts/reproduce_ablation.sh
SEEDS=0 bash scripts/reproduce_ablation.sh
```

输出：`results/ablation/pgd_eps_4.0/...`

适用场景：需要回答“收益是不是单纯来自动量”时运行。

---

### 4.7 `scripts/reproduce_mu_scan.sh`

对应论文：动量系数 `mu` 扫描。

运行内容：

- 数据集：CIFAR-10；
- `mu = 0, 0.1, 0.5, 0.7, 0.9, 0.99`；
- `mu=0` 等价于 TTC；
- 攻击：PGD，`eps=4/255`，`K=2`。

单独运行：

```bash
bash scripts/reproduce_mu_scan.sh
SEEDS=0 bash scripts/reproduce_mu_scan.sh
```

输出：`results/mu_scan/pgd_eps_4.0/...`

适用场景：需要说明 `mu=0.9` 的选择依据时运行。

---

### 4.8 `scripts/reproduce_aft.sh`

对应论文：Table 6，AFT 叠加实验。

运行内容：

- 在 TeCoA、FARE 对抗微调模型上叠加 TTC/NAC；
- 攻击：PGD，`eps=4/255`，`K=2`；
- 需要外部权重文件：
  - `./checkpoints/tecoa_vit_b32.pth`
  - `./checkpoints/fare_vit_b32.pth`

单独运行：

```bash
bash scripts/reproduce_aft.sh
SEEDS=0 bash scripts/reproduce_aft.sh
```

输出：`results/aft/pgd_eps_4.0/...`

注意：如果权重不存在，脚本会尝试自动下载；网络受限时可能失败。  
一键运行时可用 `bash run_all.sh --skip-aft` 跳过。

---

### 4.9 `scripts/compare_ttc_nac_doc.sh`

对应论文：TTC vs NAC vs DOC 公平对比。

运行内容：

- 数据集：CIFAR-10；
- 攻击：PGD，`eps=4/255`；
- Config A：`K=2`，步长 `1/255`；
- 方法：TTC、NAC、DOC；
- DOC 使用 `learnable_tau=0.155`、`temperature=75.0`。

单独运行：

```bash
# 默认 seed 0
bash scripts/compare_ttc_nac_doc.sh

# 指定 seed
bash scripts/compare_ttc_nac_doc.sh 1
```

输出：`results/compare/pgd_eps_4.0/{ttc,nac_m0.9,doc}/seed_0.log`

注意：脚本注释里 Config B 的提示是编辑参数后单独运行，`run_all.sh` 默认只自动跑 Config A。

---

## 5. 结果聚合与检查

运行完相关脚本后，可以用聚合脚本生成汇总（自动遍历所有 `results/*/pgd_eps_*/<method>/seed_*.log`，按 seed 求均值±标准差，生成 `results/aggregated_all.json` 和 `results/aggregated_all.md`）：

```bash
python scripts/aggregate_all.py
```

只需重跑 ImageNet-100（Table 1 + Table 2，修复数据/类名后）时：`bash scripts/rerun_imagenet.sh`

主结果 JSON：

```bash
cat results/main_results.json
```

也可以直接查看单个日志，例如：

```bash
cat results/eps1/pgd_eps_1.0/nac_m0.9/seed_0.log
cat results/compare/pgd_eps_4.0/nac_m0.9/seed_0.log
```

---

## 6. 单次快速运行

如果不想跑完整脚本，只跑一个方法、一个数据集：

```bash
# Bash
bash run.sh nac cifar10 4
bash run.sh ttc cifar10 4
```

PowerShell：

```powershell
.\run.ps1 -Method nac -Dataset cifar10 -Eps 4
.\run.ps1 -Method ttc -Dataset cifar10 -Eps 4
```

也可以直接调主入口：

```bash
python.exe nac_fair_experiment.py \
  --test_set cifar10 \
  --test_eps 4 \
  --counterattack nac \
  --batch_size 32
```

---

## 7. 运行顺序建议

如果只是想复现论文：

1. 先跑 `SEEDS=0 bash run_all.sh --skip-aft`，快速检查主要表格；
2. 确认环境正常后，再跑 `bash run_all.sh` 完整版；
3. 如果只需要某张表，按第 4 节单独跑对应脚本；
4. 跑完后执行聚合脚本，并对照 `results/main_results.json` 核对。

---

## 8. 当前已知注意事项

- 论文主表采用 `seed 0`；五数据集 multi-seed pilot 最大 std `<=1.16pp`。
- 主表 batch size：五个数据集 `bs=64`，ImageNet-100 `bs=32`。
- 当前部分 `scripts/*.sh` 通过 `common.sh` 自动选择 batch size；若要严格复现论文中不同表的 `bs=32/64` 口径，建议在对应命令前显式设置 `BATCH_SIZE`。
- ImageNet-100 自动下载依赖网络；网络受限时请手动准备数据。
- AFT 需要外部模型权重，网络受限时请提前下载到 `checkpoints/`。
- AutoAttack 需要额外安装 `autoattack`。
- 完整多 seed 模式运行时间较长，8 GB 显卡上预计可能达到 15 小时级别。
- 如果 Git Bash 报 `python: command not found`，请确认 `F:\\anaconda3\\python.exe` 存在；`scripts/common.sh` 已尝试自动把 `/f/anaconda3` 加入 PATH。

---

## 9. 关键文件速查

| 文件 | 作用 |
|------|------|
| `run_all.sh` | 一键运行所有论文复现脚本 |
| `run.sh` / `run.ps1` | 单次快速实验 |
| `nac_fair_experiment.py` | 主实验框架 |
| `evaluate.py` | 轻量单数据集评估 |
| `nac.py` | NAC 独立实现 |
| `ttc.py` | TTC 独立实现 |
| `attacks.py` | PGD/AutoAttack 攻击生成 |
| `utils.py` | 数据加载与评估工具 |
| `setup_datasets.py` | 数据集准备 |
| `plot_figures.py` | 论文图表生成 |
| `scripts/aggregate_results.py` | 汇总日志结果 |
| `README.md` | 英文主 README |
| `README_zh.md` | 本中文说明 |