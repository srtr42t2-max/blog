---
title: "5.3 Self-Draft 方案：Medusa 多头预测与 EAGLE-2 动态草稿树"
published: 2026-10-07T15:00:00
description: "拆掉独立小模型：Medusa 用多个解码头并行预测未来 token，EAGLE 在特征空间做轻量自回归、EAGLE-2 用置信度校准动态草稿树；附链式→树式联合验证的演进脉络与三种方案横向对比。"
tags: [投机解码, 推理优化, Medusa, EAGLE, AIInfraGuide]
category: AIInfraGuide·投机解码
author: pplk
draft: false
---

# 5.3 Self-Draft 方案：Medusa 多头预测与 EAGLE-2 动态草稿树

> **系列导航｜《AIInfraGuide》模块四·第 5 章：Speculative Decoding**
> 1. [5.1 Speculative Decoding 核心原理：先猜后验的无损加速](../specdec-51-speculative-sampling/)
> 2. [5.2 Draft 模型方案：独立小模型的选型、对齐与接受率](../specdec-52-draft-model/)
> 3. **5.3 Self-Draft 方案：Medusa 多头预测与 EAGLE-2 动态草稿树（本篇）**
> 4. [5.4 收益边界与限制：什么时候投机解码不划算](../specdec-54-limits/)
> 5. [5.5 动手实验：vLLM 实测代码生成 vs 开放对话的接受率](../specdec-55-vllm-lab/)

## 本章简介

本篇是《AIInfraGuide》模块四"推理优化"第 5 章"Speculative Decoding"的第 3 篇（共 5 篇）。第 5.2 篇的独立 Draft 方案有一个挥之不去的结构性成本：**维护两个模型**——两份权重、两套 KV、两次调度，词表还必须同源。Self-Draft 的思路是把草稿能力直接"长"在 Target 模型自己身上：复用它的 backbone、词表和分布，只加一个很轻的预测模块。

要解决的问题有三个：

1. Medusa 的多个 Decoding Head 怎么训练、怎么一次前向验证一棵候选树？"典型接受"和严格无损差在哪？
2. EAGLE 为什么把草稿从 token 空间搬到特征空间？这个改动为什么能把接受长度从链式草稿的 2~3 抬到 4 以上（EAGLE-2）？
3. 从链式草稿到静态树再到 EAGLE-2 的动态草稿树，"Token 级 → Block 级联合验证"这条演进线解决了什么问题？

读完你应该能：画出 Medusa 的头结构与 Tree Attention 的验证流程；讲清 EAGLE 特征自回归的输入输出与训练目标；说清楚 EAGLE-2 的动态树凭什么比静态树省验证预算；并能为自己的部署场景在"独立 draft / Medusa / EAGLE"之间做出有依据的选择。

## 1. 为什么要 Self-Draft：独立 Draft 的三笔账

第 5.2 篇结尾提到的天花板，具体是三笔账：

- **显存账**：独立 draft 要额外加载一份权重和一套 KV。70B 配 1.5B 还好，但 draft 越大甜点越难找——5.2 的扫描显示 α 与 c 的乘积约束把最优 draft 尺寸压死在 1/15~1/25；
- **运维账**：两个模型的版本、量化格式、采样参数要成对维护，任何一边升级都要重测接受率；
- **分布账**：独立小模型与 target 的分布 gap 只能靠蒸馏缓解，而 Self-Draft 模块**天然共享 target 的 backbone 表示**——草稿看到的特征就是 target 自己算出来的，分布对齐的起点高得多。

Medusa 与 EAGLE 是 Self-Draft 的两条代表路线：一个给 target 加"多个输出头"并行猜多个未来 token，一个加"一层特征自回归"串行猜但猜得更准。

## 2. Medusa：给 Target 模型装几个"预言头"

### 2.1 结构：一个 backbone，K 个头

Medusa（Cai et al., 2024 / ICML 2024，"Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）的做法直白到令人愉快：冻结 target 的 transformer backbone，在其最后一层 hidden state 上额外接 K 个解码头（每个头通常是一层带残差的前馈网络 + 共享原 LM head 的词表投影），**第 k 个头负责预测第 t+k+1 个 token**（原 LM head 预测 t+1）。一次 target 前向同时产出 K+1 个位置的分布——草稿不再需要第二个模型，也不需要串行跑 γ 步小模型前向。

训练分两档：

- **Medusa-1（冻结 backbone）**：只训 K 个头。backbone 不动，target 原始能力零影响；训练数据可以直接用 target 的 SFT 语料，甚至用 target 自己生成的语料（Self-Distillation）——5.2 篇"on-policy 数据"思想的头版应用。几个小时 GPU 就能出一个头；
- **Medusa-2（联合微调）**：backbone 与头一起训，头的接受率更高，但需要全量训练资源，且要小心别把 backbone 训偏（论文用多头损失加权缓解）。

### 2.2 从多头到候选树：Tree Attention 一次验一棵

K 个头各出一个 token，得到的只是一条链（链式草稿，γ=K）。Medusa 的第二步是**取每个头的 top-k 候选做笛卡尔积**，组成一棵候选树——比如 4 个头各取 top-2，理论上 2⁴=16 条路径，实际按各头概率的联合得分剪枝保留几十条候选路径。然后用一次前向验证整棵树，关键机制是 **Tree Attention**：把树拍平成一个序列，配合精心构造的注意力 mask，让每个候选节点只能 attend 到它在树上的祖先路径——同一层级的兄弟分支互相不可见。这样一次前向就为树上所有节点算出了 target 分布，验证成本仍是"一次权重读取 + γ 级别的额外位置计算"。

```text
链式草稿 (γ=3):        h1 -> h2 -> h3          1 条路径，验 3 个位置

树状草稿 (top-2 示意):        ┌- h2a - h3a
                  h1a ┤
                      └- h2b        ┌- h2a
                  h1b ──────────────┘
                  Tree Attention: 一次前向并行验证所有分支,
                  mask 保证 h3a 只看得到 h1a->h2a 这条祖先路径
```

验证时沿树找到"target 也认可"的最长路径（贪心模式下即逐节点比对 argmax），一次接受该路径上的全部节点。树的价值在于**同一个验证预算下覆盖更多可能**：链式草稿第 2 位猜错就全废，树里第 2 位备了多个候选，总有一条命中的概率大得多——期望接受长度因此显著高于同深度的链。

### 2.3 "典型接受"与严格无损的分界线

必须说清楚的一点：Medusa 论文的默认加速数字用的不是 5.1 篇那种严格的拒绝采样，而是**典型接受（typical acceptance）**——按 target 分布的概率与熵设阈值，"够典型"的候选就接受。这让接受率更高、加速更猛，但输出分布与 target **不再严格一致**（论文用一系列任务指标论证质量无明显退化）。Medusa 同时支持切换回标准投机采样以保证无损。**选型时要先问自己：这个业务要的是"数学无损"还是"指标无损"**——代码补全、内部工具可以要后者，对外 API 的 greedy 流量通常要前者。

### 2.4 数字坐标

Medusa 论文报告（论文报告值，随实现与硬件变化）：Vicuna 7B/13B/33B 上 Medusa-1 约 2.2× 加速、Medusa-2 约 2.3×~2.8×；配合典型接受最高约 3.6×。典型接受长度 τ 在 2.5~3.5 区间——这是"加几个头"这种近乎零架构侵入的方案能拿到的上限，也标定了 EAGLE 要超越的基线。

## 3. EAGLE：把草稿搬进特征空间

### 3.1 核心洞见：特征比 token 好预测

EAGLE（Li et al., ICML 2024，"Speculative Sampling Requires Rethinking Feature Uncertainty"）从一个观察出发：**token 级预测的本质困难在于采样引入的不确定性**——下一步 token 是从分布里采出来的，draft 永远猜不准"骰子的结果"；但 target 的倒数第二层特征 f（LM head 之前的 hidden state）是确定性的连续向量，**特征序列的自回归结构远比 token 序列规则**。

于是 EAGLE 的草稿模块（一个极轻的 decoder 层，参数量通常不到 target 的 5%）做这样的事：

```text
输入:  target 的倒数第二层特征序列 f₁..f_t（验证时白拿，零成本）
       + 下一个 token 的 embedding e_{t+1}（已确认/token 已知）
输出:  预测的下一个特征 f̂_{t+1}，过 target 的 LM head 得到草稿分布 q
自回归: 把 (f̂, e) 继续喂回草稿层, 串行猜出整棵草稿树
```

三个设计细节值得记住：其一，**LM head 与 backbone 全部复用 target 的**，新增参数只有那一层 decoder 层——词表问题、分布对齐问题一起消失；其二，输入里混入"下一个 token 的 embedding"是为了消掉采样的随机性——特征 f_t 加上实际采到的 token 恰好确定下一步特征；其三，训练目标是特征回归损失 + 分布分类损失的加权和，用 target 自己前向跑出的 (f, token) 对做监督——天然的 on-policy 数据，几卡小时的量级。

### 3.2 EAGLE-2：置信度校准的动态草稿树

EAGLE-1 的草稿树是**静态**的：固定的树形、固定的节点数，每个请求都用同一棵。EAGLE-2（Li et al., EMNLP 2024，"Faster Inference of Language Models with Dynamic Draft Trees"）指出这是浪费：上下文简单时（代码缩进、固定搭配）应该多猜深猜，上下文分歧大时应该少猜省验证预算。做法分两步：

- **扩展阶段**：草稿层自回归展开时，用草稿头输出的概率近似"该节点最终被接受的概率"——路径上各节点概率连乘得到路径级置信度，只对置信度高的分支继续展开。直觉：草稿分布越尖峰（自信），target 越可能落在同一处；
- **重排阶段**：验证前把全部候选节点按全局置信度排序，取 top-N 组成最终验证树——验证预算（一次前向的位置数）固定，但花在了最可能命中的节点上。

论文报告 EAGLE-2 比 EAGLE-1 再快约 20%~40%，在 MT-bench、GSM8K、HumanEval 等基准上对 7B~70B 模型的端到端加速约 3.0×~4.3×、平均接受长度约 4~5.5（论文报告值），且全程使用标准投机采样验证——**无损**。这是"验证数学不变、只改候选构造"的红利：候选越聪明，同一套拒绝采样给出的 τ 越大。

### 3.3 EAGLE-3：松开"特征必须复现"的约束

EAGLE-3（2025）进一步拿掉了"草稿必须预测 target 特征"的约束：改用多层特征融合 + training-time test（训练时模拟多步草稿暴露的分布漂移），论文报告接受长度与加速比再上台阶（部分设置下论文报告超过 5×~6×），并已被 vLLM、SGLang 等引擎跟进支持。细节随版本演进快，落地时以引擎文档与论文最新版为准；但它的存在说明这条路线仍有红利可挖。

## 4. 从 Token 级到 Tree（Block）级联合验证

把三章的脉络收拢，草稿形态经历了一条清晰的演进线：

```text
链式 (Chain)            静态树 (Static Tree)         动态树 (Dynamic Tree)
逐 token 串行猜测   →   固定树形, 多头 top-k 笛卡尔积  →  按置信度自适应展开+重排
独立 draft / EAGLE-1    Medusa Tree Attention         EAGLE-2 / Sequoia
验证 = 逐位置比对       验证 = Tree Attention 并行     验证 = 同左, 预算花在刀刃上
```

三个不变量贯穿始终：**验证侧的数学没有变过**——始终是对 target 分布做逐位置（逐节点）的拒绝采样或 argmax 比对，无损性由 5.1 篇的等式保证；变的是候选的构造方式，而候选构造的唯一目标是**在固定验证预算内最大化期望接受长度**。Sequoia（Chen et al., 2024）把树形选择本身做成了硬件感知的优化问题（给定树规模和硬件，用动态规划找最优树结构），可作为这条线的延伸阅读。

从系统视角看，"Token 级 → Block 级联合验证"的本质是把 decode 的计量单位从"单 token"放大成"token 块"：调度器每轮处理一个可变长的块，吞吐与延迟的账本都要按块重算——这正是第 5.4 篇与 Continuous Batching 叠加时复杂度的来源。

## 5. 横向对比：独立 Draft / Medusa / EAGLE-2

| 维度 | 独立小模型（5.2） | Medusa | EAGLE-2 |
|---|---|---|---|
| 额外参数 | 整个小模型（1%~10%+） | K 个头（约 1%~2%） | 一层 decoder（<5%） |
| 词表要求 | 必须与 target 一致 | 天然共享 | 天然共享 |
| 训练成本 | 零（现成）或一次蒸馏 | 冻结 backbone 训头，小时级 | 训一层，小时~天级 |
| 草稿生成 | 串行 γ 步小模型前向 | 一次前向并行出头 | 串行但每层极轻 |
| 无损性 | 严格无损（拒绝采样） | 可选：典型接受有损，RS 无损 | 严格无损 |
| 典型加速（论文报告值） | 2×~3× | 2.2×~3.6× | 3.0×~4.3× |
| 引擎支持 | vLLM/TRT-LLM/SGLang 广泛 | vLLM/SGLang 等（Medusa 头） | vLLM/SGLang/TRT-LLM（EAGLE/EAGLE-3） |
| 适合场景 | 快速验证、无训练资源 | 单模型自托管、可微调 | 追求极限接受长度、可训一层 |

> **选型决策**：要零成本验证收益 → 独立小模型或 n-gram；要自托管单模型且能跑微调 → Medusa-1 起步；要极限 τ 且愿意训一层 + 跟进引擎支持 → EAGLE 系。所有路线的验证侧数学相同，可以按"候选构造质量/成本比"单一维度比较。

## 6. 常见问题（FAQ）

**Q1：Medusa 的头数和每个头的 top-k 怎么定？** 论文常用 K=4~5 个头；top-k 与树规模是验证预算与命中率的权衡——树越大，一次前向验证的位置越多（大 batch 下有成本，见 5.4），但命中概率越高。论文给出过按硬件与模型调好的候选树配置，直接抄再按实测微调即可。

**Q2：Medusa-2 联合微调会把原模型训坏吗？** 风险存在但可控：论文用 backbone 原损失 + 各头损失的加权和联合优化，并报告联合训练后原模型指标无明显退化。稳妥做法是先 Medusa-1 冻结上线，有资源再评估 Medusa-2。

**Q3：EAGLE 的草稿层为什么要吃"下一个 token 的 embedding"？** 为了消除采样随机性：特征 f_t 是确定性的，但下一个 token 是从分布采出来的；不把实际采到的 token 告诉草稿层，它就既要学会预测特征、又要猜骰子——混入 embedding 后，"特征 + 已知 token → 下一特征"变成确定性映射，可学性强得多。这就是论文标题里 "Rethinking Feature Uncertainty" 的含义。

**Q4：EAGLE-2 的置信度近似可靠吗？** 它用的是草稿头自身概率的连乘，是"被接受概率"的近似而非精确值——精确值依赖 target 分布，验证前拿不到。论文的消融表明这个近似足以指导树的展开与重排；理解为启发式打分即可，不影响无损性（无损由验证侧保证，候选构造只影响速度）。

**Q5：Self-Draft 和独立 draft 能叠加吗？** 可以但少有必要：EAGLE 头本身接受长度已接近验证预算的甜点，再套一层独立 draft 会把调度复杂度翻倍。更常见的"叠加"是 EAGLE + 量化、EAGLE + PD 分离这类正交优化。

**Q6：我的模型没有现成的 EAGLE/Medusa 头怎么办？** 社区已为 Llama、Qwen、Vicuna、Mixtral 等主流模型发布训练好的头（EAGLE 官方仓库与 HuggingFace 可检索）；没有现成头时，Medusa-1 冻结训练是成本最低的自建路径——几小时 GPU + 一份 SFT 语料。

## 本章小结

1. **Self-Draft 拆掉独立小模型的三笔账**：显存、运维、分布对齐；草稿模块共享 target 的 backbone、词表与特征，起点天然更高。
2. **Medusa = 多头并行 + 树验证**：K 个头各预测一个未来位置，top-k 组成候选树，Tree Attention 一次前向验整棵树；默认的典型接受有损，切拒绝采样才严格无损——选型先想清楚要哪种"无损"。
3. **EAGLE = 特征空间自回归**：特征确定、规则、可回归，token 有采样随机性；一层小 decoder 吃特征+embedding 串行出草稿（EAGLE-1 接受长度约 3.5，EAGLE-2 约 4~5.5）；EAGLE-2 用置信度做动态树，把固定验证预算花在最可能命中的分支上，论文报告 3.0×~4.3× 且无损。
4. **演进主线是"候选构造"，验证数学不变**：链 → 静态树 → 动态树，目标都是在固定验证预算内最大化期望接受长度；无损性始终由 5.1 的拒绝采样等式兜底。
5. **下一篇回到系统**：接受率由任务决定、验证在大 batch 下不再免费、与量化和 Continuous Batching 叠加的账本——投机解码的收益边界。

## 延伸阅读

- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**（Cai et al., 2024 / ICML 2024）：多头结构与 Tree Attention 的原始文献，典型接受的阈值设计在正文，训练配方在附录；
- **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**（Li et al., ICML 2024）：特征级自回归的动机推导与训练目标，理解"为什么特征比 token 好猜"必读；
- **EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees**（Li et al., EMNLP 2024）：动态草稿树的展开与重排算法，置信度近似的消融实验；
- **Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding**（Chen et al., 2024）：把树形结构选择做成硬件感知优化问题，树验证的系统性研究；
- **本系列**：第 5.1 篇（验证数学）、第 5.2 篇（独立 draft 的对照系）、第 5.4 篇（大 batch 下树验证的成本边界）。

## 参考文献

- Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads — https://arxiv.org/abs/2401.10774
- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty — https://arxiv.org/abs/2401.15077
- EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees — https://arxiv.org/abs/2406.16858
- EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test — https://arxiv.org/abs/2503.01840
- Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding — https://arxiv.org/abs/2402.12374
- EAGLE 官方实现与预训练草稿头 — https://github.com/SafeAILab/EAGLE
- vLLM Speculative Decoding 文档 — https://docs.vllm.ai/en/latest/features/spec_decode.html
