---
title: "从流式响应日志计算 TTFT、ITL 与 Goodput"
published: 2026-08-12
description: "不用 GPU，通过一组带时间戳的请求事件实现指标计算，并识别平均值隐藏的卡顿。"
tags: ["AI Infra", "LLM 推理", "实验", "Benchmark", "日志分析", "SLO"]
category: "AI Infra · 实验"
draft: true
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 实验 | 基础 | 45 分钟 | 在线推理指标与测量契约；Python 基础 | 2026-08-13 |

本实验不需要 GPU。目标是先把测量代码写正确，再把它接到真实推理服务。

## 实验问题

两个流式请求具有相同平均 TPOT，但其中一个偶尔停顿 500 ms。平均值能否反映用户体验？本实验采用逐请求阈值 `TTFT <= 800 ms` 且请求内 `ITL P99 <= 300 ms`；P99 使用 nearest-rank，统计窗口固定为 `[0, 1100 ms)`，无预热。两个请求的 Goodput 是否相同？

## 输入格式

创建一个事件列表。时间统一用单调时钟的毫秒偏移量：

```python
events = [
    {"request_id": "a", "event": "arrival", "t_ms": 0},
    {"request_id": "a", "event": "token", "token_index": 1, "t_ms": 420},
    {"request_id": "a", "event": "token", "token_index": 2, "t_ms": 695},
    {"request_id": "a", "event": "token", "token_index": 3, "t_ms": 970},
    {"request_id": "a", "event": "done", "t_ms": 975},
    {"request_id": "b", "event": "arrival", "t_ms": 100},
    {"request_id": "b", "event": "token", "token_index": 1, "t_ms": 500},
    {"request_id": "b", "event": "token", "token_index": 2, "t_ms": 550},
    {"request_id": "b", "event": "token", "token_index": 3, "t_ms": 1050},
    {"request_id": "b", "event": "done", "t_ms": 1055},
]
```

## 实现要求

为每个请求计算：

```text
TTFT = first_token - arrival
E2E = done - arrival
ITL_i = token_i - token_(i-1)
TPOT = mean(ITL)
```

输出一个 Token 时，TPOT 应记录为 `None`，不能记为零。百分位数需要注明插值方式；本实验固定使用 nearest-rank，数据量很小时同时保留原始样本。

本组数据的预期结果是：A 的 `TTFT=420 ms`、`ITL=[275, 275] ms`、`TPOT=275 ms`；B 的 `TTFT=400 ms`、`ITL=[50, 500] ms`、`TPOT=275 ms`。在上述阈值和窗口下，A 通过、B 不通过，因此 Goodput 为 `1 / 1.1 ≈ 0.909 req/s`。这组结果只用于练习，真实报告还应给出样本量和百分位算法。

然后实现：

```python
def meets_slo(request, ttft_limit_ms=800, request_itl_p99_limit_ms=300):
    ...

goodput = slo_passed_requests / measurement_window_seconds
```

## 需要回答

1. 请求 A 与 B 的平均 TPOT 各是多少？
2. 哪个请求违反 ITL SLO？
3. 若只看总体平均 TPOT，会遗漏什么？
4. 如果服务端或代理每 200 ms 才 flush 一次，客户端观测的 ITL 会怎样变化？

## 完成标准

- 单元测试覆盖零 token、一个 token、乱序事件和请求失败。
- 报告中写清计时边界、统计窗口、失败请求处理。
- 保存逐请求数据，而不只输出聚合均值。

## 延伸

把同一套事件采集接入任意 OpenAI-compatible 流式接口。SSE chunk 是传输事件，不保证与 tokenizer token 一一对应，可能包含零个、一个或多个 token；优先读取服务端 token usage/token id，无法取得时应把指标命名为 chunk latency，而不是 token ITL。客户端可在每次 read/process 时用单调时钟记录，同时采集服务端排队与执行指标，对比两种边界。

## 参考资料

- [NVIDIA GenAI-Perf Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)（官方文档）
