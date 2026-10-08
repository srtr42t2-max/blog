---
title: "5.4 收益边界与限制：什么时候投机解码不划算"
published: 2026-10-07T17:00:00
description: "投机解码的边界条件全景：任务类型与温度如何决定接受率，大 batch 下验证何时从免费变成负担（Roofline 定量分析），与量化叠加为什么 1+1<2，与 Continuous Batching 叠加的调度复杂度，附启用决策清单。"
tags: [投机解码, 推理优化, Roofline, ContinuousBatching, AIInfraGuide]
category: AIInfraGuide·投机解码
author: pplk
draft: false
---

# 5.4 收益边界与限制：什么时候投机解码不划算

> **系列导航｜《AIInfraGuide》模块四·第 5 章：Speculative Decoding**
> 1. [5.1 Speculative Decoding 核心原理：先猜后验的无损加速](../specdec-51-speculative-sampling/)
> 2. [5.2 Draft 模型方案：独立小模型的选型、对齐与接受率](../specdec-52-draft-model/)
> 3. [5.3 Self-Draft 方案：Medusa 多头预测与 EAGLE-2 动态草稿树](../specdec-53-medusa-eagle2/)
> 4. **5.4 收益边界与限制：什么时候投机解码不划算（本篇）**
> 5. [5.5 动手实验：vLLM 实测代码生成 vs 开放对话的接受率](../specdec-55-vllm-lab/)

## 本章简介

本篇是《AIInfraGuide》模块四"推理优化"第 5 章"Speculative Decoding"的第 4 篇（共 5 篇）。前三篇把方法讲透了：验证并行、数学无损、候选构造一路从链进化到动态树。但论文里的 2×~4× 都长在同一类坐标系里——**低并发、memory-bound、高接受率负载**。生产系统不总在这个坐标系里。

要解决的问题有四个：

1. 哪些任务接受率天然高、哪些天然低？温度在其中扮演什么角色？
2. batch 变大后，"验证近乎免费"的前提什么时候失效？失效点能不能定量算出来？
3. 投机解码和量化叠加，为什么收益是次可加的？"无损"的参照系会发生什么变化？
4. 和 Continuous Batching 一起上生产，调度复杂度具体涨在哪里？

读完你应该能：用 Roofline 算出自己的部署在什么 batch 规模后不该再开投机；讲清"投机 × 量化"叠加的三层风险；并拿着决策清单判断一个新业务该不该开、开多大 γ。

## 1. 任务决定接受率：α 的场景分布

接受率 α = Σ min(p, q) 是"draft 与 target 分布的重叠面积"，而重叠面积本质由**下一步 token 的可预测性**决定——这正是任务属性：

| 场景 | 接受率水平 | 原因 |
|---|---|---|
| 代码生成 / 补全 | 高 | 语法确定、局部重复多、缩进与模板可预测；n-gram 草稿都能命中 |
| 翻译 / 摘要 / 复述 | 较高 | 输出被输入强约束，歧义空间小 |
| 格式化输出（JSON/SQL/工具调用） | 高 | 结构 token（括号、键名、逗号）几乎必然 |
| 数学推理链 | 中高 | 模板化推导步骤多，但关键数字处分歧大 |
| 开放对话 / 问答 | 中 | 知识性内容可预测，措辞分歧多 |
| 创意写作 / 头脑风暴 | 低 | 多样续写都是"对的"，p 本身就平坦 |

两个放大器要记住：**温度**——温度越高分布越平，p 与 q 的重叠越小，α 随温度单调恶化（5.1 篇 FAQ 的定性结论，实测温升时肉眼可见）；**draft 与 target 的能力差**——开放对话里 draft 不仅在"骰子"上输，还在"知识"上输，α 受损是双重的。Leviathan et al. 与 Chen et al. 两篇奠基论文的 2×~3× 都是在翻译、摘要、结构化任务上测的；换到高温创意生成，同样的系统加速比掉到 1.2×~1.5× 甚至接近 1 是常态。**"投机解码能加速多少"的第一答案永远是：先测你的负载的 α。**

## 2. Batch 边界：验证何时不再免费

### 2.1 用 Roofline 把失效点算出来

5.1 篇的前提——"验证 γ+1 个位置与生成 1 个 token 同价"——只在 memory-bound 区间成立。decode 的每一步，内存侧要搬一遍权重（与 batch 无关），计算侧是 batch × 每 token FLOPs（随 batch 线性涨）。设 H100 SXM（HBM3 3.35 TB/s，BF16 dense 约 989 TFLOPS），7B FP16 模型：

```text
内存时间 = 15.2 GB / 3.35 TB/s ≈ 4.54 ms   （每步固定）
计算时间 = batch × 2×7e9 FLOPs / 989 TFLOPS ≈ batch × 14.2 μs
盈亏平衡 B* = 4.54 ms / 14.2 μs ≈ 321      （batch 超过 B* 转入 compute-bound）
```

开投机（γ=4，每请求验证 5 个位置）后，每步的等效位置数变成 batch×5，计算时间按位置数线性放大。**只要 batch×(γ+1) < B*，验证仍然白拿；超过之后，多出来的每个位置都按 compute-bound 价格收费**：

```text
7B FP16, gamma=4:
  B=  1: 验证 5 个位置    时间惩罚 1.00x   （纯白拿）
  B= 32: 验证 160 个位置  时间惩罚 1.00x
  B= 64: 验证 320 个位置  时间惩罚 1.00x   （踩线）
  B=128: 验证 640 个位置  时间惩罚 2.00x   （验证费翻倍）
```

### 2.2 这意味着什么

- **投机解码是"低并发延迟优化"，不是"高并发吞吐优化"**。B 小（单请求、小批量、PD 分离里的 Decode 实例低载期），它是免费的午餐；B 大，验证把每步耗时拉高，而被拒绝的位置是纯浪费的 FLOPs——吞吐可以**下降**。
- **γ 和树规模要随负载收缩**。B 大时砍 γ（甚至砍到 0 = 关闭投机）比硬顶更优；这正是 EAGLE-2 动态树在系统侧的价值——验证预算可以按负载调节。
- **引擎已经内置了这个认知**：vLLM 提供按 batch size 禁用投机解码的开关（`speculative_config` 中的 `disable_by_batch_size`，超过阈值即退回普通 decode），SGLang/TensorRT-LLM 有等效机制。生产配置里**这个阈值应该用上面的 Roofline 算个初值，再压测定稿**。

## 3. 与量化叠加：为什么 1 + 1 < 2

第 4 章的量化与第 5 章的投机，吃的本来是**同一块肉**——decode 的内存墙时间。叠加时的三层账：

**第一层：收益次可加。** 量化把每步要搬的权重字节砍掉 3/4，投机的"并行验证白拿"空间同比缩水。用 2.1 的方法重算 7B INT4（权重约 4.3 GB）：内存时间降到 1.28 ms，盈亏平衡 B* 从 321 掉到 **91**——γ=4 时 batch 到 16~32 就开始交验证税（B=32 惩罚 1.76×）。量化后的模型"更早进入 compute-bound"，投机可用的负载窗口整体左移。**两个优化都有效，但叠加后的总收益小于各自收益之和，规划容量时别做乘法梦。**

**第二层："无损"参照系漂移。** 拒绝采样保证的是输出分布等于 **target 模型当前的分布**——target 量化后，投机解码精确复现的是"量化 target 的分布"，不是原始 FP16 模型的分布。量化引入的精度变化不会因为叠加投机而消失或放大，两者正交；但排障时要记住顺序：先单独验证量化 target 的质量，再把"输出与量化 target 是否一致"作为投机的验收标准。

**第三层：draft 侧量化的精度风险。** 给 draft 上 INT4/FP8 把 c 再压一截，代价是 q 分布偏移、α 下降。因为收益= f(α, c) 是联合函数，"draft 量化是否划算"没有先验答案，只能实测：INT8 通常几乎不伤 α，INT4 需要数据说话。另一个坑是 kernel：投机验证的非标准 batch 形状（每请求 γ+1 个位置）可能让某些量化 kernel 离开最优路径，端到端数字才是裁决者。

## 4. 与 Continuous Batching 叠加：调度复杂度涨在哪

普通 Continuous Batching 里，每轮迭代每个请求恰好前进 1 个 token——batch 是整齐的。开了投机之后，**每轮每个请求前进 0~γ+1 个不等长的 token**，这份"参差"会顺着调度链路传导：

- **变长步进（ragged batch）**：有的请求这轮 +5、有的 +1，序列长度逐轮错开。attention kernel、position id、采样状态都要按变长处理；现代引擎（vLLM V1、SGLang）已支持，但这是实打实的代码路径复杂度，边缘 case（chunked prefill 与投机交错、前缀缓存命中段的草稿起点）都是 bug 高发区；
- **KV 回滚**：被拒绝位置的 KV 要作废。PagedAttention 下是块表指针回退，本身便宜，但与前缀缓存（被共享的块不能随一个请求的回滚而失效）、抢占（被抢占请求的投机中间态如何恢复）交织后，正确性论证明显变难；
- **调度决策爆炸**：调度器现在要决定"哪些请求参加本轮、各猜多长、验证预算怎么分"，还要在第 2 节的负载边界上动态开关投机——每多一个旋钮，SLO 违规时的归因链就长一截；
- **可观测性负债**：吞吐指标必须拆成"草稿 token / 接受 token / 真实前进 token"三套口径，否则你会把 draft 空转当成绩。

工程结论：**先把 Continuous Batching + 量化跑稳，再开投机；开投机时按队列深度设自动降档**（升载降 γ、超阈值关闭），并把接受率、验证开销、真实前进速率做成仪表盘三件套。

## 5. 其他值得知道的边界

- **长上下文**：上下文变长后，attention 的 FLOPs 与 KV 读取不再可忽略，2.1 的"验证白拿"前提进一步收紧——验证 γ+1 个位置要多 attend γ+1 份长上下文；同时 draft 也要 attend 同样长的上下文（Self-Draft 同样躲不开），c 随上下文变长而恶化。长文本 RAG 上投机收益通常短于短对话。
- **显存**：draft 权重 + 验证期临时 KV + 树验证的扩展 batch，都挤占本可给 KV Pool 的空间——长上下文高并发场景，这笔显存可能换成更多并发更划算（机会成本的算法见[KV Cache 显存账本：形状、容量与生命周期](../ai-infra/concepts/03-kv-cache-ledger/)）。
- **MoE target**：每 token 只激活部分专家，decode 的内存压力小于总参数量暗示的水平，B* 左移，投机窗口收窄；draft 设计也更难（5.2 篇 FAQ）。
- **冷启动与首 token**：投机只加速 decode，TTFT 一分钱不少；短输出请求（分类、路由）连 decode 都没几步，开了也是白开。

## 6. 启用决策清单

```text
□ 负载画像：分任务、分温度实测 α 与 τ（5.5 篇的实验流程）
    - 代码/格式化/翻译类 α 高 → 强候选；开放创意类 α 低 → 谨慎
□ 负载位置：典型 decode batch 是 1~16（延迟敏感）还是 64+（吞吐优先）？
    - 前者：投机是免费午餐；后者：按 Roofline 算 B*，超了就不开或动态关
□ 与量化的组合：target 已 INT4/FP8？
    - 是：B* 大幅左移，重算 γ 与开关阈值；别期望收益相乘
□ 草稿路线：n-gram（代码）→ 同族小模型 → Self-Draft（5.2/5.3 决策树）
□ γ / 树规模：从 4~5 起步，按实测 τ 与验证成本调；保留按负载降档逻辑
□ 引擎机制：确认所用版本支持投机 + 动态禁用 + 指标暴露（以版本文档为准）
□ 可观测：接受率、验证开销、真实前进速率三件套上线前先接好
□ 回退：一键关闭投机的能力常备；灰度 → 对照 → 全量
```

## 7. 常见问题（FAQ）

**Q1：能不能"高吞吐模式关投机、低延迟模式开投机"一键切换？** 可以且推荐：用 `disable_by_batch_size` 类机制让引擎按实际负载自动切换，比人工模式开关更细粒度。注意切换本身有抖动成本（KV 布局变化），阈值两侧留滞回区间。

**Q2：接受率很高但端到端没快多少，哪里漏了？** 按顺序查：draft 的实际单步延迟（c 是不是比参数量比暗示的大得多——小模型 kernel launch 开销）；验证 kernel 在非标准形状下是否低效；γ 是否与 α 匹配（5.1 的最优 γ 扫描）；指标口径（看的是不是含 prefill 的端到端，投机不加速 TTFT）。

**Q3：投机解码对 P99 延迟的影响方向？** 双向：低载时 decode 步时间均匀缩短，P99 改善；高载时验证税 + 变长步进会放大尾部分叉——高负载 SLO 敏感服务必须实测 P99，不能只看中位数。

**Q4：树越大越好吗？** 不是。树规模 = 验证预算：低 batch 大树划算（验证白拿），高 batch 大树是把 FLOPs 往水里扔。EAGLE-2 的动态树之所以是方向，正因为它把"树多大"变成了逐请求、随负载可调的变量。

**Q5：量化 + 投机 + 前缀缓存 + chunked prefill 能全开吗？** 能，但组合每加一层，正确性与性能归因的复杂度就上一个台阶。纪律：一次只上一个变量，每个组合都有独立的回归测试与压测基线。

## 本章小结

1. **α 是任务属性**：代码/格式化/翻译高，开放创意低，温度是单向放大器（越高越伤）；第一步永远是实测自己负载的 α。
2. **验证免费有明确边界**：Roofline 给出 B*（7B FP16 on H100 ≈ 321 个位置/步），batch×(γ+1) 超过 B* 后投机开始倒贴——它是低并发延迟工具，不是高并发吞吐工具。
3. **与量化次可加**：量化把 B* 左移一个量级（7B INT4 的 B*≈91），叠加收益小于各自之和；"无损"的参照系随 target 量化漂移，draft 量化的净收益只能实测。
4. **与 Continuous Batching 的复杂度在变长步进、KV 回滚、调度决策与可观测口径四处传导**；生产纪律是：先跑稳基础栈、按负载自动降档、指标三件套、常备回退。
5. 下一篇收束全章：动手用 vLLM 把"代码生成 vs 开放对话"的接受率差异实测出来，把本篇的边界条件变成你自己机器上的数字。

## 延伸阅读

- **Fast Inference from Transformers via Speculative Decoding**（Leviathan et al., ICML 2023）：论文所有数字的负载坐标系（翻译/摘要、低并发）值得重读，对照本篇第 1 节；
- **vLLM Speculative Decoding 文档与 `disable_by_batch_size` 机制**：动态开关的工程实现参照；
- **本系列**：第 4.3 篇（decode 带宽账与量化收益）、第 4.6 篇（量化选型决策树，与本篇第 3 节互为镜像）、[KV Cache 显存账本：形状、容量与生命周期](../ai-infra/concepts/03-kv-cache-ledger/)（显存机会成本的算法）；
- **Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding**（Chen et al., 2024）：硬件感知的树规模优化，本篇第 2 节思想的进一步深化。

## 参考文献

- Fast Inference from Transformers via Speculative Decoding — https://arxiv.org/abs/2211.17192
- Accelerating Large Language Model Decoding with Speculative Sampling — https://arxiv.org/abs/2302.01318
- Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding — https://arxiv.org/abs/2402.12374
- EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees — https://arxiv.org/abs/2406.16858
- vLLM Speculative Decoding 文档 — https://docs.vllm.ai/en/latest/features/spec_decode.html
- vLLM Optimization and Tuning 文档 — https://docs.vllm.ai/en/latest/configuration/optimization.html
