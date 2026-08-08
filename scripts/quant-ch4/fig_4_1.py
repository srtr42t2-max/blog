# 4.1 配图与量化误差实验：FP16 分布 vs INT8 量化分布
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------- 1. 手写对称量化 / 反量化 ----------
def quantize(x, qmax=127):
    """对称均匀量化：float -> int8 + scale"""
    s = np.abs(x).max() / qmax          # scale：一个量化码对应多少浮点单位
    q = np.clip(np.round(x / s), -qmax, qmax)
    return q.astype(np.int8), s

def dequantize(q, s):
    """反量化：int8 -> float"""
    return q.astype(np.float64) * s

# 合成一份"像 LLM 权重"的数据：正态主体 + 少量重尾
w = rng.standard_normal(200_000) * 0.05
w[:500] *= 6.0
w16 = w.astype(np.float16).astype(np.float64)   # 模拟 FP16 存储

q, s = quantize(w16)
w_hat = dequantize(q, s)
err = w_hat - w16

print(f"scale s = {s:.6f}")
print(f"理论单点误差上界 s/2 = {s/2:.6f}")
print(f"实际最大绝对误差      = {np.abs(err).max():.6f}")
print(f"平均绝对误差 (MAE)    = {np.abs(err).mean():.6f}")
print(f"相对误差 (L2 范数比)  = {np.linalg.norm(err) / np.linalg.norm(w16):.6%}")
print(f"INT8 码点利用率       = {len(np.unique(q))} / 255")

# ---------- 2. 绘图 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(w16, bins=200, color="#4C72B0", alpha=0.85)
axes[0].set_title("Original FP16 Weights")
axes[0].set_xlabel("value")
axes[0].set_ylabel("count")

axes[1].hist(w16, bins=200, color="#4C72B0", alpha=0.6, label="FP16")
axes[1].hist(w_hat, bins=200, color="#DD8452", alpha=0.6, label="INT8 dequantized")
axes[1].set_title("After INT8 Symmetric Quantization")
axes[1].set_xlabel("value")
axes[1].legend()

fig.suptitle(f"FP16 vs INT8 (scale={s:.4f}, rel L2 err={np.linalg.norm(err)/np.linalg.norm(w16):.2%})")
fig.tight_layout()
fig.savefig("src/content/posts/quant-41-fp32-to-int4/4.1-fp16-vs-int8-dist.png", dpi=150)
print("saved: src/content/posts/quant-41-fp32-to-int4/4.1-fp16-vs-int8-dist.png")
