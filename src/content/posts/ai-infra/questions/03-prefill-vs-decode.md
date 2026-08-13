---
title: "为什么常说 Prefill compute-bound、Decode memory-bound？这总成立吗？"
published: 2026-08-12
description: "从算术强度、矩阵形状和 KV 读取解释经验规律的适用边界。"
tags: ["AI Infra", "LLM 推理", "面试题", "Prefill", "Decode", "Roofline"]
category: "AI Infra · 面试题"
draft: false
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 面试题 | 进阶 | 12 分钟 | Prefill、Decode 与请求时间线、Roofline 基础 | 2026-08-13 |

## 30 秒回答

Prefill 一次处理很多 Prompt token，矩阵乘法规模大、数据复用较高，通常更接近计算上限；Decode 每个序列每轮通常只有一个新 token，需要反复读取模型权重和历史 KV，算术强度低，常接近显存带宽上限。这只是常见工作点，不是定理。Batch、上下文长度、量化、Kernel、硬件和 MoE 路由都可能改变瓶颈，最终应通过 Roofline 和 Profiler 验证。

## 3 分钟展开

先明确测量的内存层级，再估算 `FLOPs / bytes moved`。Decode 的小矩阵和 KV 读取往往让带宽成为上限；增大 batch 可提高计算利用率，但也会增加排队和 KV 压力。短上下文、大 batch 或特殊 Tensor Core Kernel 下，Decode 可能转为 compute-bound。

## 常见追问

- 为什么 FlashAttention 能在不改变数学结果的情况下加速？
- 长 Prompt 对 Prefill 和 Decode 各有什么影响？
- 如何用 Nsight Systems 和 Nsight Compute 区分 GPU 空闲、带宽瓶颈和 Kernel launch 瓶颈？

关联：[Prefill、Decode 与请求时间线](/blog/posts/ai-infra/concepts/02-prefill-decode/)。

## 参考资料

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)（论文）
