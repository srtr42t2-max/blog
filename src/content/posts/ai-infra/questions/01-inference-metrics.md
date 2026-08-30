---
title: "如何严格定义 TTFT、TPOT、ITL、E2E、Throughput 和 Goodput？"
published: 2026-08-12
description: "这道题考察你是否能先固定测量边界，再讨论推理系统性能。"
tags: ["AI Infra", "LLM 推理", "面试题", "指标", "Benchmark", "SLO"]
category: "AI Infra · 面试题"
draft: true
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 面试题 | 基础 | 12 分钟 | 在线推理指标与测量契约 | 2026-08-13 |

## 30 秒回答

我会先说明计时边界，再给定义。设请求到达、首 token 可见、后续 token 可见和结束的时间分别是 `t_arrival`、`t_first`、`t_i`、`t_done`：`TTFT = t_first - t_arrival`；`ITL_i = t_i - t_(i-1)`；本站 `TPOT = (t_last - t_first)/(N_out-1)`，所以只输出一个 token 时 TPOT 未定义；`E2E = t_done - t_arrival`。吞吐要说明是请求、输入 token、输出 token 还是总 token 每秒。Goodput 则是统计窗口内满足全部 SLO 的完成请求数除以窗口时长。

## 3 分钟展开

平均 TPOT 不能代替 ITL 分布：一个请求中间卡顿 500 ms，平均值可能仍然很好。因此在线体验至少看 TTFT、ITL P50/P95/P99 和 E2E P99。比较两个系统时，还要固定模型版本、硬件、输入/输出长度分布、到达过程、并发、预热、错误处理和客户端边界。

Goodput 的计算口径必须先说清楚。若采用逐请求硬阈值，就判断 `request TTFT <= 800 ms` 且该请求 `ITL P99 <= 80 ms`，超出任一阈值的请求不计入 Goodput；若业务写的是窗口级 `TTFT P99 <= 800 ms`，那是聚合 SLO，只能判断整个窗口是否达标，不能把单个请求标记为“违反 P99”。仅凭 output tok/s 不能判断系统是否更好。

## 追问与评分点

- **基础**：能写出 TTFT、ITL、吞吐的基本定义。
- **合格**：说明计时边界、P99、输出一个 token 时 TPOT 的边界。
- **优秀**：能解释开放/封闭负载、Goodput 和 coordinated omission，并给出可复现 Benchmark 配置。

## 常见误区

- 把 TPOT 写成首 token 到最后 token之间的总时间，忘记除以 `N_out - 1`。
- 把 ITL 和 TPOT 当同一个指标。
- 只报平均值，不报负载和尾延迟。

详细推导见：[先定义再优化：在线推理指标与测量契约](/blog/posts/ai-infra/concepts/01-metrics-contract/)。

## 参考资料

- [NVIDIA GenAI-Perf Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)（官方文档）
