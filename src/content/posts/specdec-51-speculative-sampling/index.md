---
title: "5.1 Speculative Decoding 核心原理：先猜后验的无损加速"
published: 2026-10-07T11:00:00
description: "讲透 Speculative Decoding 的 Draft-Verify 框架：为什么验证 γ 个 token 与生成 1 个几乎同价，Rejection Sampling 如何从数学上保证输出分布严格不变，附 numpy 分布一致性实验与 α/γ/c 加速比公式推导。"
tags: [投机解码, 推理优化, Speculative Decoding, LLM推理, AIInfraGuide]
category: AIInfraGuide·投机解码
author: pplk
draft: false
---

# 5.1 Speculative Decoding 核心原理：先猜后验的无损加速

> **系列导航｜《AIInfraGuide》模块四·第 5 章：Speculative Decoding**
> 1. **5.1 Speculative Decoding 核心原理：先猜后验的无损加速（本篇）**
> 2. [5.2 Draft 模型方案：独立小模型的选型与接受率](../specdec-52-draft-model/)
> 3. [5.3 Self-Draft 方案：Medusa 多头预测与 EAGLE-2 动态草稿树](../specdec-53-medusa-eagle2/)
> 4. [5.4 收益边界与限制：什么时候投机解码不划算](../specdec-54-limits/)
> 5. [5.5 动手实验：vLLM 实测代码生成 vs 开放对话的接受率](../specdec-55-vllm-lab/)

## 本章简介

本篇是《AIInfraGuide》模块四"推理优化"第 5 章"Speculative Decoding"的第 1 篇（共 5 篇）。第 4 章量化回答的问题是"每生成一个 token，能不能少搬几个字节的权重"；本章换一个维度追问：**一次前向能不能多产出几个 token？**

要解决的问题有三个：

1. 为什么"验证 γ 个候选 token"和"生成 1 个 token"的成本几乎相同？这个不对称性是整个方法的地基；
2. Rejection Sampling（拒绝采样）凭什么保证投机解码的输出与 Target 模型原始输出的分布**严格一致**——即所谓的"无损"到底在数学上意味着什么？
3. 接受率 α、猜测长度 γ、草稿成本比 c 三个变量如何共同决定加速比？γ 是不是越大越好？

读完你应该能：徒手推一遍 Speculative Sampling 的正确性等式；用 numpy 在 30 行内复现"投机采样输出分布 = Target 分布"；给出一个接受率和草稿成本，算出最优 γ 和期望加速比。

## 1. 串行瓶颈再审视：Decode 的"一次一个"

第 4.3 篇已经算过 decode 的带宽账：decode 阶段每步只生成一个 token，GEMM 退化成 GEMV，每产出一个 token 都要把全部权重从 HBM 完整读一遍，是 memory-bound。以 Qwen2.5-7B 为例（FP16 权重约 15.2 GB，H100 SXM HBM3 带宽 3.35 TB/s），单请求 decode 的理论时间下限约 4.5 ms/token——**算力大量闲置，时间全花在搬权重上**。

这里藏着整个投机解码赖以成立的不对称性：

- **"生成下一个 token"是严格串行的**：第 t+1 个 token 依赖第 t 个的输出，没有并行空间；
- **但对一段"已经知道内容"的 token 序列做前向，是完全可并行的**——这正是 prefill 在做的事：输入 1000 个 token，一次前向算出所有位置的 logits，权重照样只读一遍。

把两件事放在一起看：如果有一个"预言家"告诉你接下来的 5 个 token 是什么，你用一次前向（约 4.5 ms，和生成 1 个 token 同价）就能同时验证这 5 个位置——因为验证只是对已知序列的前向计算，是 prefill 式的并行操作。**memory-bound 区间里，FLOPs 近乎免费，字节数才是硬通货；同一次权重读取，顺带多算几个位置几乎不花钱。**

```text
朴素 decode:  [读权重 4.5ms] -> 1 个 token
              [读权重 4.5ms] -> 1 个 token
              ...                 （γ+1 次前向 = γ+1 个 token）

投机 decode:  [draft 猜 γ 个，小模型，便宜]
              [读权重 4.5ms] -> 一次验证 γ+1 个位置
                                接受其中 0~γ 个 + 1 个修正/bonus token
                                （1 次大模型前向 = 平均 2~3+ 个 token）
```

> **一句话总结**：decode 贵不是因为"算一个 token"贵，而是"每算一个就要搬一遍权重"贵；既然横竖要搬一遍，不如一次多验证几个——这就是投机解码的第一性原理。

## 2. Draft–Verify：把串行解码改造成"批量验证"

Speculative Decoding（投机解码，亦称推测解码）的两篇奠基论文——Google 的 Leviathan et al.（Fast Inference from Transformers via Speculative Decoding，ICML 2023）与 DeepMind 的 Chen et al.（Accelerating Large Language Model Decoding with Speculative Sampling）——在 2022 年底至 2023 年初先后独立提出了同一个框架：

```text
一轮迭代的四个步骤（γ = 猜测长度，本篇取 γ=4 为例）:

① Draft:   小模型（Draft 模型）自回归地猜 γ 个 token:
           x̂₁, x̂₂, x̂₃, x̂₄          （γ 次小模型前向，廉价）

② Verify:  Target 模型对 [前缀 + x̂₁..x̂₄] 做一次并行前向，
           得到 5 个位置的分布: p₁, p₂, p₃, p₄, p₅

③ Accept:  从左到右逐位置判定:
           位置 i 接受 ⇔ 按投机采样规则（第 3 节）接受 x̂ᵢ
           一旦某个位置被拒绝，其后所有猜测作废

④ Correct: 若位置 k 首次被拒，从修正分布 p'ₖ 采一个 token；
           若全部接受，从 p₅ 采一个 "bonus token"
```

关键性质：**每一轮迭代至少产出 1 个 token（bonus/修正 token），至多产出 γ+1 个**。哪怕 draft 全军覆没，这一轮也等价于一次普通的 target decode——损失只是 draft 的开销和验证多算的几个位置（在 memory-bound 区间内几乎为零）。

从调度视角看，这是用"批量验证"替代"逐一生成"：原来 γ+1 次串行前向压缩成了 1 次并行前向 + γ 次廉价的小模型前向。压缩能不能转化成真实的墙钟加速，取决于第 4 节的成本模型。

一个容易被忽略的实现细节是 KV Cache：验证时 target 模型会为所有草稿位置算好 KV，被拒绝位置之后的 KV 直接截断作废（PagedAttention 下只需回退 block table 的指针，代价可忽略）；draft 模型的 KV 同样跨轮保留，每轮按接受结果回滚未命中的后缀后继续自回归，并不需要整体重建。投机解码的额外显存主要就是一份 draft 模型权重与它的 KV。

## 3. Speculative Sampling：数学上严格无损

"无损"是投机解码区别于绝大多数推理加速技术的招牌：量化拿精度换带宽，投机解码声称**输出分布和 Target 模型逐点一致**。这个声称靠 Rejection Sampling 保证，本节把它推到底。

### 3.1 接受-重采样规则

记号：某一步的真实分布为 p（Target 模型给出），draft 模型的分布为 q，draft 提出的候选 token x ∼ q。投机采样（Speculative Sampling）规则：

```text
以概率 min(1, p(x)/q(x)) 接受 x;
若拒绝, 从修正分布 p'(x) = norm(max(0, p(x) - q(x))) 重新采样
```

直觉：draft 在哪里的概率给得比 target 高，就在哪里以一定概率驳回；驳回后从"p 超出 q 的那部分残差"里补采。规则保证每个 token 的"出价"恰好被校正到 p。

### 3.2 正确性：一行代数

令 β 为拒绝概率，β = Σₓ q(x)·(1 - min(1, p(x)/q(x))) = 1 - Σₓ min(p(x), q(x))。修正分布归一化常数恰好也是 β，因为 Σₓ max(0, p(x) - q(x)) = Σₓ [p(x) - min(p(x), q(x))] = 1 - Σₓ min(p(x), q(x)) = β。于是任意 token x 的最终产出概率为：

```text
P(x) = P(提议 x 且接受) + P(拒绝后从修正分布采到 x)
     = q(x) · min(1, p(x)/q(x))  +  β · max(0, p(x)-q(x)) / β
     = min(p(x), q(x))           +  max(0, p(x) - q(x))
     = p(x)                                        ∎
```

**逐点相等，没有任何近似**。同时得到单步接受率的理论值：

```text
α_step = Σₓ min(p(x), q(x)) = 1 - D_TV(p, q)     （D_TV 为总变差距离）
```

接受率就是两个分布的"重叠面积"——draft 与 target 分布越接近，重叠越大，接受率越高。这个等式是第 5.2 篇一切 Draft 选型工作的理论靶心。

### 3.3 30 行 numpy 复现分布一致性

纸上得来终觉浅。下面在合成分布上验证：target 分布 p 取随机 logits 的 softmax（尖峰形态，模拟真实 next-token 分布），draft 分布 q 是 p 加噪声的退化版本；然后跑 20 万次投机采样，看产出分布是否收敛到 p：

```python
import numpy as np

rng = np.random.default_rng(0)
V = 16

# target 分布 p：随机 logits 的 softmax（尖峰，模拟真实 next-token 分布）
logits_p = rng.standard_normal(V) * 2.0
p = np.exp(logits_p - logits_p.max()); p /= p.sum()

# draft 分布 q：p 的 logits 加噪声（一个不完美的"小模型"）
noise = rng.standard_normal(V) * 0.8
logits_q = logits_p + noise
q = np.exp(logits_q - logits_q.max()); q /= q.sum()

def spec_sample(p, q, rng):
    """投机采样一步：x~q 提议，min(1, p/q) 接受；拒绝则从 norm(max(0,p-q)) 补采"""
    x = rng.choice(len(p), p=q)
    if rng.random() < min(1.0, p[x] / q[x]):
        return x, True
    r = np.maximum(0.0, p - q); r /= r.sum()
    return rng.choice(len(p), p=r), False

N = 200_000
counts = np.zeros(V, dtype=int); acc = 0
for _ in range(N):
    x, a = spec_sample(p, q, rng)
    counts[x] += 1; acc += a

emp = counts / N
print("token  p(target)  empirical  q(draft)")
for i in np.argsort(-p)[:6]:
    print(f"{i:5d}  {p[i]:.4f}     {emp[i]:.4f}    {q[i]:.4f}")
print(f"max |empirical - p| = {np.abs(emp - p).max():.4f}")
print(f"empirical acceptance   = {acc / N:.4f}")
print(f"theoretical acceptance = {np.minimum(p, q).sum():.4f}")
```

本机真实输出：

```text
token  p(target)  empirical  q(draft)
    6  0.4218     0.4205    0.2292
    7  0.2066     0.2087    0.2531
    2  0.1119     0.1126    0.1438
    5  0.0641     0.0640    0.1768
    0  0.0400     0.0394    0.0239
    3  0.0383     0.0376    0.0816
max |empirical - p| = 0.0021
empirical acceptance   = 0.7570
theoretical acceptance = 0.7565
```

两个看点：其一，q 和 p 差异巨大（token 6 上 q 只给了 p 一半的概率，token 5 上 q 又是 p 的近 3 倍），但 20 万次采样后产出分布与 p 的最大偏差只有 0.0021——纯蒙特卡洛噪声量级，**拒绝采样把 draft 的所有偏差都修正掉了**；其二，经验接受率 0.7570 与理论值 Σ min(p,q) = 0.7565 吻合到小数点后三位，重叠面积公式不是近似，是精确等式。

再固定噪声方向、扫噪声强度 σ，验证"draft 越差接受率越低"的单调关系：

```text
noise sigma= 0.1  theoretical=0.9866  empirical=0.9865
noise sigma= 0.3  theoretical=0.9613  empirical=0.9607
noise sigma= 0.5  theoretical=0.9378  empirical=0.9364
noise sigma= 1.0  theoretical=0.8776  empirical=0.8800
noise sigma= 2.0  theoretical=0.7362  empirical=0.7358
```

### 3.4 贪心解码的特例：逐 token 一致

工程上更常用的是温度 0 的贪心解码，此时投机验证退化为最朴素的形式：**接受 x̂ 当且仅当 x̂ = argmax p**（draft 猜的就是 target 会选的）；一旦不匹配，用 argmax p 替换并终止本轮。输出与纯 target 贪心解码**逐 token 完全相同**——不是分布一致，是字符串级一致。这也是生产系统敢默认开投机解码的原因：对 greedy 流量，它是严格免费的彩票，中奖率取决于 draft 质量。

## 4. 加速比公式：α、γ、c 的三角博弈

### 4.1 期望加速比推导

做一个标准简化：假设每个草稿位置的接受概率独立同分布为 α（现实中 α 随位置衰减、随内容波动，5.2 篇会回到这个假设上），则一轮迭代接受的草稿 token 数服从截断几何分布，加上必有的 1 个 bonus/修正 token，期望产出：

```text
E[本轮产出 token 数] = Σ_{k=0}^{γ} α^k = (1 - α^(γ+1)) / (1 - α)
```

成本侧：设 draft 单次前向耗时是 target 的 c 倍（c ≈ 小模型权重字节数 / 大模型权重字节数，memory-bound 下近似成立），一轮迭代耗时 γ·c + 1 个"target 前向单位"。期望加速比：

```text
Speedup(α, γ, c) = (1 - α^(γ+1)) / [(1 - α) · (γ·c + 1)]
```

### 4.2 数值表：三个变量的敏感度

用上面的公式算一张表（c = draft/target 单步耗时比）：

```text
gamma     c | a=0.5  a=0.6  a=0.7  a=0.8  a=0.9
    3  0.05 | 1.63x  1.89x  2.20x  2.57x  2.99x
    3  0.10 | 1.44x  1.67x  1.95x  2.27x  2.65x
    3  0.20 | 1.17x  1.36x  1.58x  1.85x  2.15x
    5  0.05 | 1.57x  1.91x  2.35x  2.95x  3.75x
    5  0.10 | 1.31x  1.59x  1.96x  2.46x  3.12x
    5  0.20 | 0.98x  1.19x  1.47x  1.84x  2.34x
    8  0.05 | 1.43x  1.77x  2.28x  3.09x  4.38x
    8  0.10 | 1.11x  1.37x  1.78x  2.40x  3.40x
    8  0.20 | 0.77x  0.95x  1.23x  1.66x  2.36x
```

三条读法：

1. **α 是绝对主角**。α 从 0.5 提到 0.9，同样配置下加速比接近翻倍再翻倍；社区所有 Draft 方案（第 5.2、5.3 篇）本质都是在"保持 c 小"的约束下把 α 顶上去。
2. **c 决定 γ 的可行域**。c=0.20 时 γ=5、α=0.5 的组合加速比 0.98x——**投机反而变慢**；draft 不够便宜时多猜是负收益。
3. **γ 存在最优值，且最优值随 α 变化**。固定 α=0.7、c=0.1 扫 γ：

```text
gamma=1: E[tokens]=1.700, cost=1.10, speedup=1.55x
gamma=2: E[tokens]=2.190, cost=1.20, speedup=1.82x
gamma=3: E[tokens]=2.533, cost=1.30, speedup=1.95x
gamma=4: E[tokens]=2.773, cost=1.40, speedup=1.98x   <- 峰值
gamma=5: E[tokens]=2.941, cost=1.50, speedup=1.96x
gamma=6: E[tokens]=3.059, cost=1.60, speedup=1.91x
gamma=8: E[tokens]=3.199, cost=1.80, speedup=1.78x
```

期望产出随 γ 增长但边际递减（几何级数收敛），成本却线性增长，交点就是最优 γ。α=0.7 时峰值在 γ≈4，这正是 vLLM、TensorRT-LLM 等引擎默认猜测长度普遍取 4~5 的算术根源。

### 4.3 论文数字的坐标系

Leviathan et al. 论文报告在 T5-XXL（11B）翻译与摘要任务上取得 2×~3× 的端到端加速，且输出与原模型逐点一致（greedy 与采样两种模式均验证）；Chen et al. 在 Chinchilla 70B 上报告约 2×~2.5×（均为论文报告值）。注意这些数字的坐标系：**单请求或极小 batch、memory-bound 场景**。大 batch 下 decode 转向 compute-bound，"验证近乎免费"的前提失效——这个边界条件留给第 5.4 篇展开。

## 5. 常见问题（FAQ）

**Q1：验证为什么能并行？一次前向怎么同时算出 γ+1 个位置的分布？** 就是 prefill 的机制：把 [已确认前缀 + γ 个草稿 token] 拼成一条序列做 teacher-forcing 前向，因果注意力 mask 保证位置 i 的分布只依赖它左边的 token——与"这些 token 是怎么来的"无关。每个位置的 logits 都是"如果前缀是这样，下一个 token 是什么"，天然并行。

**Q2：被拒绝位置之后的 KV Cache 怎么办？** 截断丢弃。验证时 target 为所有草稿位置计算了 KV，第 k 位被拒则 k 之后的 KV 作废；PagedAttention 下这只是把序列的 block table 长度回退几个块，没有额外计算。draft 模型的 KV 跨轮保留，同样按接受结果回滚后缀后继续自回归，不整体重建。

**Q3：说"无损"，是指每次运行输出都一样吗？** 精确含义是**分布逐点相等**（采样模式）或**逐 token 相等**（贪心模式）。现实中还有一层工程噪声：并行验证改变了 kernel 的归约顺序和 batch 形状，浮点非结合性会让 logits 出现 ULP 级差异，极低概率下翻转一次 argmax——这与"同模型不同 batch 输出略有差异"是同一类现象，不是投机解码引入的新问题。

**Q4：batch 里每个请求的接受率不同，怎么调度？** 每轮各请求独立接受不同数量的 token，序列长度会错开，需要引擎支持变长步进；且大 batch 下验证不再免费。这是投机解码与 Continuous Batching 叠加的核心复杂度，第 5.4 篇专章讨论。

**Q5：温度采样（temperature > 0）下还能用吗？** 可以，3.1 节的规则就是为一般分布写的，p、q 换成温度缩放后的分布即可。但要注意：高温让分布变平坦，draft 与 target 的重叠面积通常下降，接受率随温度升高而走低。

**Q6：这和 Beam Search 有什么区别？** 正相反的设计目标：Beam Search 用更多计算**改变**（希望改善）输出；投机解码用更多并行计算**保持**输出不变、只换时间。两者可以叠加但解决的是不同问题。

## 本章小结

1. **投机解码的地基是 decode 的带宽不对称**：生成串行、验证并行，而 memory-bound 下并行验证 γ 个位置与生成 1 个 token 同价——一次权重读取可以顺带验证多个位置。
2. **Draft–Verify 每轮产出 1 ~ γ+1 个 token**：draft 猜 γ 个，target 一次前向并行验证，接受最长正确前缀，拒绝处从修正分布补采；全中另有 bonus token，每轮至少产 1 个。
3. **无损是代数恒等式不是经验结论**：P(x) = min(p,q) + max(0,p−q) = p，逐点成立；单步接受率 α_step = Σ min(p,q) = 1 − D_TV(p,q)，numpy 实验把两个等式都验到了小数点后三位。贪心模式下输出逐 token 一致。
4. **加速比由 α、γ、c 三角决定**：Speedup = (1−α^(γ+1)) / [(1−α)(γc+1)]；α 是主角，c 决定 γ 的可行域，γ 存在内点最优（α=0.7、c=0.1 时 γ*≈4）。draft 不够便宜时投机可以为负收益。
5. **坐标系意识**：论文 2×~3× 的数字基于单请求 memory-bound 场景；大 batch 下前提失效，这是 5.4 篇的主题。接下来的 5.2 篇回答第一个工程问题：独立的 Draft 小模型怎么选、怎么训、接受率怎么测。

## 延伸阅读

- **Fast Inference from Transformers via Speculative Decoding**（Leviathan, Kalman, Matias, 2022 / ICML 2023）：投机解码的奠基论文之一，第 3、4 节的接受规则与加速比公式均出自此文，附录有完整正确性证明；
- **Accelerating Large Language Model Decoding with Speculative Sampling**（Chen, Borges, Piot et al., DeepMind, 2023）：与上文同期独立提出同一框架，对修正分布的推导更清晰，并在 70B 规模验证了无损性；
- **DistillSpec: Improving Speculative Decoding via Knowledge Distillation**（Zhou et al., 2023）：当 draft 与 target 分布差距大时如何用蒸馏对齐，是第 5.2 篇"能力匹配"问题的先修；
- **本系列**：第 4.3 篇《Weight-only INT4：GPTQ、AWQ 与 Marlin Kernel》（decode 带宽账的完整推导，本篇第 1 节的前提）；后续第 5.2 篇讨论独立 Draft 模型，第 5.3 篇讨论 Medusa/EAGLE-2 等 Self-Draft 方案。

## 参考文献

- Fast Inference from Transformers via Speculative Decoding — https://arxiv.org/abs/2211.17192
- Accelerating Large Language Model Decoding with Speculative Sampling — https://arxiv.org/abs/2302.01318
- DistillSpec: Improving Speculative Decoding via Knowledge Distillation — https://arxiv.org/abs/2310.08461
- vLLM Speculative Decoding 文档 — https://docs.vllm.ai/en/latest/features/spec_decode.html
- HuggingFace Assisted Generation（transformers 侧的投机解码实现） — https://huggingface.co/blog/assisted-generation
