---
title: "用白板解释 PagedAttention：它减少了什么，又没有减少什么？"
published: 2026-08-12
description: "从逻辑块、物理块和 Block Table 讲清分页 KV 管理的收益边界。"
tags: ["AI Infra", "LLM 推理", "面试题", "PagedAttention", "Block Table", "Prefix Cache"]
category: "AI Infra · 面试题"
draft: true
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 面试题 | 进阶 | 14 分钟 | KV Cache 显存账本、虚拟内存分页 | 2026-08-13 |

## 30 秒回答

PagedAttention 把 KV Cache 切成固定大小的逻辑 Block，用 Block Table 映射到不连续的物理 Block；请求按需申请和释放，多个请求还可以共享相同前缀的物理 Block。它减少最大长度提前预留和外部碎片，提高可承载并发，但不改变理论 KV bytes/token，也不减少 Decode 对历史 KV 的读取。它是引擎显式管理的软件分页抽象，不是 OS page fault，也不是 CUDA thread block。

## 追问

- Block Size 越小为什么不一定更好？
- Prefix Cache 命中后为什么 Decode 不一定变快？
- KV Block 不足时 Scheduler 可以怎样处理？每种策略会影响什么指标？

关联：[PagedAttention：块表、分配、共享与抢占](/blog/posts/ai-infra/concepts/04-paged-attention/)。

## 参考资料

- [PagedAttention Paper](https://doi.org/10.1145/3600006.3613165)（论文）
