# 4.2 配图与 SmoothQuant 等价性/误差实验
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---------- 1. 合成带 outlier channel 的激活 ----------
tokens, channels = 512, 256
X = rng.standard_normal((tokens, channels)) * 0.5
outlier_idx = rng.choice(channels, size=6, replace=False)
X[:, outlier_idx] *= 20.0                      # 6 个 outlier channel，幅值 ~20 倍
W = rng.standard_normal((channels, channels)) * 0.05

def quant_per_tensor(x, qmax=127):
    s = np.abs(x).max() / qmax
    return np.clip(np.round(x / s), -qmax, qmax) * s

# ---------- 2. SmoothQuant 平滑变换 ----------
alpha = 0.5
s = (np.abs(X).max(axis=0) ** alpha) / (np.abs(W).max(axis=1) ** (1 - alpha))
s = np.maximum(s, 1e-5)

X_smooth = X / s          # 激活被压平
W_smooth = W * s[:, None] # 量化难度迁移到权重

# 数学等价性验证
y_ref = X @ W
y_sm  = X_smooth @ W_smooth
print(f"等价性: max |(X/s)@(sW) - X@W| = {np.abs(y_sm - y_ref).max():.2e}")

# 量化误差对比（激活 per-tensor INT8，权重保持浮点，仅看激活侧难度）
Xq  = quant_per_tensor(X)
Xqs = quant_per_tensor(X_smooth)
err_before = np.linalg.norm(Xq - X) / np.linalg.norm(X)
err_after  = np.linalg.norm(Xqs - X_smooth) / np.linalg.norm(X_smooth)
print(f"激活量化相对误差: 平滑前 {err_before:.2%}  ->  平滑后 {err_after:.2%}")

# ---------- 3. 绘图 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
axes[0].imshow(np.abs(X), aspect="auto", cmap="viridis")
axes[0].set_title("Activation |X| before smoothing")
axes[0].set_xlabel("channel")
axes[0].set_ylabel("token")

axes[1].imshow(np.abs(X_smooth), aspect="auto", cmap="viridis")
axes[1].set_title("Activation |X/s| after SmoothQuant")
axes[1].set_xlabel("channel")

fig.suptitle(f"Outlier channels are flattened (quant err {err_before:.1%} -> {err_after:.1%})")
fig.tight_layout()
fig.savefig("src/content/posts/quant-42-smoothquant-w8a8/4.2-smoothquant-outlier.png", dpi=150)
print("saved: src/content/posts/quant-42-smoothquant-w8a8/4.2-smoothquant-outlier.png")
