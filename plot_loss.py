import numpy as np
import matplotlib.pyplot as plt

data = np.load("loss_history.npz")
recon = data["recon"]
semantic = data["semantic"]
sparsity = data["sparsity"]

steps = np.arange(len(recon)) * 20

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(steps, recon, color="steelblue", label="Reconstruction (MSE)")
ax.plot(steps, semantic, color="darkorange", label="Semantic (cosine + MSE vs CLIP)")
ax.plot(steps, sparsity, color="seagreen", label="Sparsity (mean spike rate)")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss History")
ax.set_xmargin(0.01)
ax.set_ymargin(0)
ax.legend()

fig.tight_layout()
plt.savefig("loss_history.png", dpi=150)
plt.show()
