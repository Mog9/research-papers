import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="#ffffff")
fig.patch.set_facecolor("#ffffff")

fig.suptitle("Full Fine-tuning vs LoRA on SST-2 — BERT Base",
             fontsize=14, fontweight="bold", color="#111111", y=0.97)

ax1 = axes[0]
ax1.set_facecolor("#fafafa")

ax1.bar(["Trainable Params", "Val Accuracy"], [100, 92.7],
        width=0.4, color=["#cccccc", "#cccccc"], edgecolor="#aaaaaa", linewidth=0.8)

ax1.text(0, 102, "109.6M\n(100%)", ha="center", va="bottom",
         fontsize=11, fontweight="bold", color="#333333")
ax1.text(1, 94.7, "92.7%", ha="center", va="bottom",
         fontsize=11, fontweight="bold", color="#333333")

ax1.set_title("Full Fine-tuning BERT", fontsize=13, fontweight="bold", color="#111111", pad=10)
ax1.set_ylim(0, 130)
ax1.set_yticks([])
ax1.spines[:].set_color("#cccccc")
ax1.tick_params(colors="#333333", labelsize=10)

ax2 = axes[1]
ax2.set_facecolor("#fafafa")

ax2.bar(["Trainable Params", "Val Accuracy"], [0.5, 90.48],
        width=0.4, color=["#4a9eff", "#f0a800"], edgecolor="#aaaaaa", linewidth=0.8)

ax2.text(0, 102, "147K\n(0.13%)", ha="center", va="bottom",
         fontsize=11, fontweight="bold", color="#333333")
ax2.text(1, 92.5, "90.48%", ha="center", va="bottom",
         fontsize=11, fontweight="bold", color="#333333")

ax2.set_title("LoRA Fine-tuning BERT", fontsize=13, fontweight="bold", color="#111111", pad=10)
ax2.set_ylim(0, 130)
ax2.set_yticks([])
ax2.spines[:].set_color("#cccccc")
ax2.tick_params(colors="#333333", labelsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("lora_results.png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
print("saved → lora_results.png")
plt.show()