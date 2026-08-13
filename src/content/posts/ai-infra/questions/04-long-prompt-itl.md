---
title: "一个超长 Prompt 为什么会让其他请求的 ITL 抖动？Chunked Prefill 如何缓解？"
published: 2026-08-12
description: "考察 iteration-level 调度、统一 token budget 与 TTFT/ITL 权衡。"
tags: ["AI Infra", "LLM 推理", "面试题", "Chunked Prefill", "调度", "ITL"]
category: "AI Infra · 面试题"
draft: false
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 面试题 | 进阶 | 11 分钟 | Prefill、Decode 与请求时间线、在线推理指标 | 2026-08-13 |

## 30 秒回答

完整 Prefill 会占用一个很长的 GPU Step；同一 Worker 上等待 Decode 的请求无法及时执行，因此它们的 ITL 出现尖峰。Chunked Prefill 把长 Prompt 按 token budget 切成多个块，让每轮调度在 Prefill 和 Decode 之间交错，从而限制单轮阻塞时间。代价是更多调度轮次和管理开销，长请求自身 TTFT 不一定改善，切块大小应由 ITL SLO 和吞吐基准决定。

## 展开

连续批处理解决的是请求加入和退出的粒度；它并不自动保证一个超长 Prefill 不会独占一轮。统一 token budget 后，Scheduler 可先给 Decode 保留预算，再把剩余预算分给 Prefill。实际系统还要考虑 KV Block 是否足够、抢占和多模态输入。

关联：[Prefill、Decode 与请求时间线](/blog/posts/ai-infra/concepts/02-prefill-decode/)。

## 参考资料

- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)（论文）
