# Adaptive Signal Pilot — 预登记协议（写于任何 pilot 数据产生之前）

登记时间：2026-09-14
目标：为 NAC 寻找"第二个创新点"——样本级自适应机制。以数据决定 adaptive rule，不预设。

## 一、固定协议（与论文主实验一致）
- model = CLIP ViT-B/32（zero-shot, 无 AFT checkpoint）
- dataset = CIFAR-10 test set（10000 张）
- attack = PGD-10, eps=4/255, stepsize=1/255, seed=0
- counterattack: momentum=0.9, tau=0.2, beta=2.0, stepsize=1/255（全部 config 共用）
- 网格：K ∈ {1,2,3,4} × eps_ca ∈ {1,2,4}/255 → 12 个 NAC config + 1 个 TTC(K=2, eps=4) 协议校验 cell
- 关键控制：
  - 对抗样本每 batch 只生成一次，12+1 个 config 复用同一 `attacked` 张量（消除攻击随机性混淆）
  - 每个 counterattack 调用前 `torch.manual_seed(10000+batch_idx)` → 所有 config 共享相同初始 delta（消除初始化混淆）
  - 插装版 NAC 与原版 nac_counterattack 数学路径逐行一致，先做小批量等价性校验

## 二、逐样本信号定义（探测 config = K=2, eps=4/255，同时所有 config 均记录）
| # | 信号 | 定义 |
|---|---|---|
| 1 | feature_drift_norm | ‖f(x_adv)−f(x_clean)‖₂/‖f(x_clean)‖₂（=TTC τ 信号） |
| 2 | cos_clean_adv | cos(f(x_adv), f(x_clean)) |
| 3 | grad_norm0 / grad_norm_mean | 步0/全程 ‖∇_δ L‖₂ |
| 4 | grad_cos_succ | 相邻梯度 cos(g_t, g_{t+1}) 均值（自相关） |
| 5 | step_align | 取步方向 cos(v_t, g_{t+1}) 均值 |
| 6 | disp_growth | 嵌入位移增长率 (D₂−D₁)/max(D₁,ε)，D_t=‖f(la_t)−f(la_{t−1})‖ |
| 7 | gain_rate | (L_K−L_0)/K，L_t=‖f(la_t)−f(x_adv)‖² |
| 8 | lookahead_pred_err | cos(g_look_t, g_{t+1}) 均值（look-ahead 梯度预测力） |
| 9 | drift_abs | ‖f(x_adv)−f(x_clean)‖₂ 绝对值 |
| 10 | pred_instability | 分析侧：同一 eps=4 下 K∈{1..4} 防御后预测类概率的 std |
| 11 | cos_atkdelta_g0 | cos(δ_adv, g_0)（初始攻击扰动与反攻击梯度的夹角） |
| 12 | velocity_cos / clean_margin / emb_norm_clean | cos(v_t,v_{t+1}) / 干净预测 top1−top2 概率差 / ‖f(x_clean)‖ |

每个样本另存：clean_correct、adv_correct、各 config 的 defended_correct / margin / def_shift / def_cos。

## 三、分析计划（数据落地前锁定）
1. 主分析人群：adv_wrong 样本（防御仅对该人群有增益意义；全样本结果一并报告）
2. 真值定义：best_K@eps4 = argmax_K correctness（平局取小 K）；best_eps@K2 同理；benefit = correct(best) − correct(K2,eps4)
3. 统计：Pearson（信号 vs 0/1 正确性）、Spearman（信号 vs best_K 序数）、AUROC（信号预测 benefit>0，Mann-Whitney 秩和实现，不用 scipy）
4. 分组统计：信号四分位 → 各组 best-K 分布 / 平均正确率
5. 图：信号 vs 防御正确率（分箱曲线）、信号 vs best-K 分布
6. **预登记判定阈值（结果出来后不得更改）**：
   - 强证据：|Spearman| ≥ 0.30 或 AUROC ≥ 0.65（adv_wrong 人群，n≈10k）
   - 弱证据：0.15 ≤ |Spearman| < 0.30 或 0.55 ≤ AUROC < 0.65
   - **失败：所有 12 信号均低于弱证据线 → 如实报告"该方向失败"，不强行构造 adaptive rule**
7. 规则选择：仅从有证据的信号中构造；候选形态（adaptive K / step size / eps / weighting / early stop / solver 选择）由证据形态决定（单调→adaptive K；U型→solver 选择…），不预设

## 四、公平比较方案（pilot 通过后执行）
TTC / NAC-fixed(K=2,eps=4) / NAC-adaptive，同模型同攻击同数据同种子同最大预算（adaptive 的平均前向数 ≤ 固定 K=2 的预算或单独报告预算-精度 Pareto），报告 clean/adv/defended/gain/平均K/latency/peak memory。

## 五、诚实条款
- 若插装版与原版等价性校验失败 → 修复后再跑，不以插装结果为准
- 若某信号与结果无稳定关系 → 直接报告失败
- 不因结果不好而修改信号定义重跑（信号定义已在上表锁定）
