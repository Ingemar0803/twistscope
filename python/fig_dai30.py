import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import twistsim as t
G = t.G_first("graphene")
p = t.simulate_pattern("graphene", 30.0, alpha=1.0, qmax_factor=1.3)
I = ndi.gaussian_filter(p.I, 2.0); h = I.shape[0] // 2
lim = np.array([-h, h, -h, h]) * p.dq / G
fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor="white")
ax.imshow(np.log10(I / I.max() + 1e-9), cmap="gray_r", origin="lower", extent=lim, vmin=-6.5, vmax=0)
th = np.linspace(0, 2 * np.pi, 400)
for rr, col, lab in [(1.0, "#2a78d6", "first-order spots, both layers (|G|)"),
                     (np.sqrt(3) - 1, "#eb6834", "(√3 − 1)|G| = 0.732|G|"),
                     (2 * np.sin(np.radians(15)), "#e34948", "2 sin 15° |G| = 0.518|G| (moiré)")]:
    ax.plot(rr * np.cos(th), rr * np.sin(th), "--", color=col, lw=1.2, label=lab)
ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18)
ax.set_xlabel("$q_x$ / |G|"); ax.set_ylabel("$q_y$ / |G|")
ax.set_title("TwistScope: 30° twisted bilayer graphene (α = 1, log scale)", fontsize=10.5)
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
fig.tight_layout(); fig.savefig("report_figs/fig_dai30_sim.png", dpi=160)
