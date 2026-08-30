---
title: "模型权重能装进 GPU，为什么在线服务仍然会 OOM？"
published: 2026-08-12
description: "用完整显存账本定位压测时才出现的 OOM，而不是只盯着模型参数量。"
tags: ["AI Infra", "LLM 推理", "面试题", "OOM", "KV Cache", "容量规划"]
category: "AI Infra · 面试题"
draft: true
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 面试题 | 进阶 | 12 分钟 | KV Cache 显存账本、PagedAttention | 2026-08-13 |

## 30 秒回答

权重只是固定成本。在线服务还需要随并发和上下文增长的 KV Cache，以及激活、Attention/GEMM workspace、CUDA Graph、NCCL 通信 buffer、运行时上下文、分配器预留和碎片空间。应先采集显存账本与请求分布，再限制 batched tokens/并发、调整 KV Pool、量化或改变调度，并用压测验证，而不是简单把显存占满。

## 排查顺序

1. 区分加载时 OOM 和运行一段时间后 OOM。
2. 记录每个时刻的权重、KV Pool、reserved/allocated、workspace 和通信 buffer。
3. 固定输入长度、输出长度、并发，逐项放开，找出触发阈值。
4. 检查是否有未释放请求、prefix 引用、重试或 allocator 碎片。
5. 再调整 KV 预算、最大 batched tokens、量化或拒绝策略。

关联：[KV Cache 显存账本](/blog/posts/ai-infra/concepts/03-kv-cache-ledger/)。

## 参考资料

- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)（论文）
