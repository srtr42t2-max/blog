---
title: "PagedAttention：块表、分配、共享与抢占"
published: 2026-08-12
description: "从逻辑块到物理 KV Block，理解分页管理减少了什么、没有减少什么，以及它如何影响调度。"
tags: ["AI Infra", "LLM 推理", "概念", "PagedAttention", "Block Table", "Prefix Cache", "抢占"]
category: "AI Infra · 系统学习"
draft: true
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 概念 | 进阶 | 45 分钟 | 在线推理指标；KV Cache 显存账本；虚拟内存分页的基本思想 | 2026-08-13 |

传统实现若为每个请求预留最大上下文的连续 KV 空间，会产生大量未使用预留和外部碎片。PagedAttention 借用了分页的抽象：请求看到连续的逻辑 Block，底层物理 KV Block 可以分散存放。

> 30 秒结论：PagedAttention 通过 Block Table 把逻辑 KV Block 映射到非连续物理 Block，按需分配并支持共享，从而减少预留浪费和外部碎片。它没有降低每 token 的理论 KV 大小，也没有消除 Decode 对历史 KV 的读取。

## 一张 Block Table

假设每个 Block 容纳 4 个 token，请求 A 已缓存 10 个 token：

```text
Request A logical blocks
logical:       0        1        2
tokens:      [0..3]   [4..7]   [8..9]
                 |        |        |
block table:     7       21        3
                 |        |        |
physical KV:   block 7  block 21 block 3
```

上层按逻辑 token 顺序理解 KV，Attention Kernel 根据 Block Table 找到物理位置。物理 Block 不需要连续。

这里没有硬件 Page Fault：Block 的分配、映射和回收由推理引擎显式管理。KV Block 也不是 CUDA Thread Block。

## 按需分配与释放

请求被接纳时不必一次性预留最大长度，而是随 computed token 增长分配 Block：

```text
t0: A -> [7]
t1: A -> [7, 21]
t2: A -> [7, 21, 3]
t3: A finished -> blocks 7, 21, 3 returned to free pool
```

这样减少了外部碎片与过度预留，提高了同一 KV Pool 可承载的并发量。但最后一个 Block 未填满时仍有内部浪费。

## Block Size 的权衡

更小的 Block：

- 尾块平均浪费更少；
- Prefix Cache 匹配粒度更细；
- Block Table 更长，元数据和管理操作更多；
- Kernel 访问与调度开销可能增加。

更大的 Block 则相反。不存在脱离模型、Backend 和负载的“最佳 Block Size”。比较时应同时观察可缓存 token 数、吞吐、TTFT/ITL 和 CPU 调度开销。

## Prefix Sharing

如果请求 B 与 A 有完整 Block 对齐的共同前缀，它们可以引用同一个物理 Block：

```text
A table: [7, 21, 3]
B table: [7, 21, 9]
          ^ shared ^
reference count: block 7 = 2, block 21 = 2
```

只有引用计数归零，Block 才能回到 Free Pool。共享后的分叉部分需要独立 Block，必要时采用 Copy-on-Write 思想。

当前 vLLM 的 Automatic Prefix Caching 使用完整 Block 的 Hash 识别可复用前缀。Key 不应只有 token IDs，还需纳入父前缀及会改变 KV 的身份信息，例如 LoRA、Multi-modal 输入或 Cache Salt。具体字段随版本演进，必须对照当前文档与源码。

## 缓存命中减少了什么

Prefix Cache 命中可以跳过共享前缀的重复 Prefill，因此主要改善：

- 有长公共前缀请求的 TTFT；
- 重复计算量；
- 共享前缀的物理 KV 占用。

它通常不会让后续 Decode 自动变快，因为生成新 token 时仍要读取历史 KV 和模型权重。所谓命中率也必须说明按 Request、Token 还是完整 Block 统计。

## KV 不足时的系统行为

Free Block 不足并不只有 OOM 一种结果。引擎可以：

- 暂不接纳新 Prefill，让请求排队；
- 抢占低优先级序列并释放其 KV；
- 之后从保存状态继续，或重新计算被释放的 KV；
- 拒绝或降级请求。

每个选择都会影响指标。排队恶化 TTFT，抢占/重算消耗额外计算并可能制造 ITL 尖峰，拒绝则降低成功率。KV Manager 和 Scheduler 因此不能分开理解。

## PagedAttention 减少与不减少的内容

| 项目 | 是否减少 | 原因 |
|---|---|---|
| 最大长度的提前预留 | 是 | 改为按需分配 |
| 外部碎片 | 显著减少 | 物理 Block 无需连续 |
| 尾块内部碎片 | 未消除 | 最后一个 Block 可能未填满 |
| 理论 KV bytes/token | 否 | K/V 元素数量没有改变 |
| Decode 历史 KV 读取 | 否 | Attention 语义没有改变 |
| 重复前缀 Prefill | 配合 Prefix Cache 可减少 | 复用已计算的完整 Block |

## 自测

- 为什么说 PagedAttention 是软件管理的分页抽象？
- Block 越小为什么不一定越好？
- KV Pool 用尽时，Scheduler 的不同选择分别会伤害哪些指标？

## 参考资料

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://doi.org/10.1145/3600006.3613165)（论文）
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)（官方文档）
- [vLLM KV Cache Manager（核验 commit：8c011da）](https://github.com/vllm-project/vllm/blob/8c011da6d0a45f2c9316ea3c658da754f770a178/vllm/v1/core/kv_cache_manager.py)（源码）
