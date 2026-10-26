import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

TIMES_NEW_ROMAN_PATH = "/usr/share/fonts/truetype/times.ttf"
font_manager.fontManager.addfont(TIMES_NEW_ROMAN_PATH)
times_new_roman = font_manager.FontProperties(
    fname=TIMES_NEW_ROMAN_PATH
).get_name()
plt.rcParams.update({
    "font.family": times_new_roman,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

methods = [
    "IRRA",
    "RDE",
    "UP-Person",
    "UNIReID",
    "ReID5o*",
    "SCOPE",
]

map_values = {
    "IRRA":      [51.42, 68.83, 75.86, 79.15],
    "RDE":       [48.82, 67.52, 75.27, 78.64],
    "UP-Person": [50.56, 69.52, 78.60, 82.84],
    "UNIReID":   [36.98, 60.77, 72.43, 77.95],
    "ReID5o*":    [44.18, 56.43, 65.32, 70.12],
    "SCOPE":     [53.85, 72.55, 80.30, 84.18],
}

avg_map = np.array([np.mean(map_values[m]) for m in methods])

trainable_params_dict = {
    "IRRA":      195.0,
    "RDE":       153.0,
    "UP-Person": 7.4,
    "UNIReID":   150.0,
    "ReID5o*":    167.0,
    "SCOPE":     10.129,
}

total_params_dict = {
    "IRRA":      195.0,
    "RDE":       153.0,
    "UP-Person": 157.0,
    "UNIReID":   150.0,
    "ReID5o*":    167.0,
    "SCOPE":     156.0,
}

trainable_params = np.array([trainable_params_dict[m] for m in methods])
total_params = np.array([total_params_dict[m] for m in methods])

# scatter 的 s 是面积。用 sqrt 压缩大模型气泡，避免视觉上压过 SCOPE。
bubble_size = np.sqrt(trainable_params) * 95
# bubble_size = np.maximum(bubble_size, 180)  # 防止 SCOPE / UP-Person 太小看不见

plt.figure(figsize=(9, 5))

colors = {
    "SCOPE": "#d62728",
    "UP-Person": "#ff9896",
    "ReID5o*": "#00ed18",
    "UNIReID": "#1f77b4",
    "IRRA": "#9467bd",
    "RDE": "#17becf",
}

scope_idx = methods.index("SCOPE")

for i, m in enumerate(methods):
    if m == "SCOPE":
        plt.scatter(
            trainable_params[i],
            avg_map[i],
            s=bubble_size[i] + 380,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=2.4,
            zorder=6,
        )

    plt.scatter(
        trainable_params[i],
        avg_map[i],
        s=bubble_size[i],
        color=colors[m],
        alpha=0.72,
        edgecolors="black",
        linewidths=1.0,
        marker="*" if m == "SCOPE" else "o",
        zorder=5 if m == "SCOPE" else 3,
    )

offsets = {
    "SCOPE": (11.5, 0.32),
    "UP-Person": (6.0, 0.0),
    "IRRA": (-20.0, 2.0),
    "RDE": (-14.0, 2.0),
    "UNIReID": (-14.0, 2.1),
    "ReID5o*": (4.0, 2.0),
}

for i, m in enumerate(methods):
    dx, dy = offsets[m]
    plt.text(
        trainable_params[i] + dx,
        avg_map[i] + dy,
        f"{m}\n{trainable_params[i]:.1f}M / {total_params[i]:.1f}M",
        fontsize=13,
        fontweight="semibold" if m == "SCOPE" else "normal",
        color="#b00020" if m == "SCOPE" else "black",
        ha="left",
        va="center",
        bbox=(
            dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#d62728", alpha=0.98)
            if m == "SCOPE"
            else None
        ),
    )

# 只保留 SCOPE 的性能参考线，不要 Better 箭头
plt.axhline(
    y=avg_map[scope_idx],
    linestyle="--",
    linewidth=0.9,
    color="gray",
    alpha=0.35,
    zorder=1,
)

plt.xlabel("Trainable Parameters (M)", fontsize=15, fontweight="bold")
plt.ylabel("Avg-mAP on ORBench", fontsize=15, fontweight="bold")

plt.grid(True, linestyle="--", alpha=0.28, zorder=0)

plt.xlim(-5, 220)
plt.ylim(57, 74.6)

plt.tight_layout()

save_dir = "/home/yinhao/UNIReID/Figure"
os.makedirs(save_dir, exist_ok=True)

pdf_path = os.path.join(save_dir, "performance_efficiency_orbench_v2.pdf")
png_path = os.path.join(save_dir, "performance_efficiency_orbench_v2.png")

plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved to: {pdf_path}")
print(f"Saved to: {png_path}")

print("Avg-mAP values:")
for m, v in zip(methods, avg_map):
    print(f"{m:10s}: {v:.2f}")
