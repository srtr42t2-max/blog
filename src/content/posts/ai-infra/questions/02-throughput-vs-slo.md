---
title: "系统 B 的 output tok/s 更高，但 P99 TTFT 超标，它优于系统 A 吗？"
published: 2026-08-12
description: "用 Goodput 和公平的实验设计回答吞吐与用户体验冲突。"
tags: ["AI Infra", "LLM 推理", "面试题", "Goodput", "尾延迟", "性能比较"]
category: "AI Infra · 面试题"
draft: false
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 面试题 | 基础 | 10 分钟 | 在线推理指标与测量契约 | 2026-08-13 |

## 30 秒回答

不能仅凭 output tok/s 判断。先确认两个系统在相同模型、硬件、精度、输入输出分布、到达过程、并发、预热和计时边界下比较。如果 B 的 P99 TTFT 超过业务 SLO，它的有效吞吐应按 Goodput 计算，可能反而低于 A。接着要拆解是排队、Prefill、网络还是客户端缓冲造成尾延迟，并分别报告吞吐、TTFT、ITL、错误率和资源利用率。

## 3 分钟展开

更高输出吞吐可能来自更激进的 batching，但这会让短请求等待更久，或让长 Prefill 阻塞 Decode。还要警惕封闭负载和 coordinated omission：客户端因为请求未完成而停止发压，恰好把最差的排队情况隐藏了。

合理做法是先选择口径：要计算逐请求 Goodput，可写成 `request TTFT <= 800 ms` 且 `request ITL P99 <= 80 ms`，再在同一统计窗口内计算通过请求数；要评估窗口级 SLO，则报告窗口的 `TTFT P99`、`ITL P99` 是否达标，不能据此直接给每个请求贴上“违反 P99”的标签。若 B 的原始吞吐高但大量请求超过逐请求阈值，或窗口级尾延迟不达标，则不能称为更优的在线系统。

## 追问与评分点

- 能否解释开放负载与固定并发负载的差异？
- 如何定位 P99 TTFT 是排队还是 Prefill 造成的？
- 你会保留哪些逐请求原始字段？

## 关联知识

回到：[先定义再优化：在线推理指标与测量契约](/blog/posts/ai-infra/concepts/01-metrics-contract/)。

## 参考资料

- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)（论文）
