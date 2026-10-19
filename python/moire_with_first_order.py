"""
moire_with_first_order.py -- full diffraction pattern at high counts showing the
first-order Bragg spots of BOTH layers together with the moire satellites
around the zero-order beam (zoom panel on the right).

Moire spots exist in this model only via the phenomenological alpha term in
twistsim.py (see that file); detector saturation is not modelled.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import twistsim as t

OUT = "results_moire"
os.makedirs(OUT, exist_ok=True)
CASES = [("graphene", 5.0), ("graphene", 10.0), ("MoS2", 3.5)]
COUNTS = 1e7
L, DX, SIGMA_W, ALPHA = 900.0, 0.3, 90.0, 0.25


def first_order_vectors(material):
    a = t.MATERIALS[material]["a"]
    amat = np.array([[a, 0.0], [a / 2, a * np.sqrt(3) / 2]])
    b = 2 * np.pi * np.linalg.inv(amat)            # columns = b1, b2
    b1, b2 = b[:, 0], b[:, 1]
    return np.array([b1, b2, b1 + b2, -b1, -b2, -(b1 + b2)])


def rot(theta):
    th = np.radians(theta)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


fig, axes = plt.subplots(len(CASES), 2, figsize=(12, 17), gridspec_kw={"width_ratios": [1.15, 1]})
for i, (mat, th) in enumerate(CASES):
    pat = t.simulate_pattern(mat, th, L=L, dx=DX, sigma_w=SIGMA_W, alpha=ALPHA, qmax_factor=1.45)
    img = t.add_noise(pat, peak_counts=COUNTS, rng=np.random.default_rng(7))
    n = img.shape[0]
    h = n // 2
    G = t.G_first(mat)
    G1 = first_order_vectors(mat)
    G2 = G1 @ rot(th).T
    gm = G2 - G1
    gmag = np.hypot(*gm[0])
    logimg = np.log10(np.clip(img, 1, None))

    # left: full view with first-order spots of both layers + moire points
    ax = axes[i, 0]
    ext = np.array([-h, h, -h, h]) * pat.dq
    ax.imshow(logimg, cmap="gray_r", extent=ext, origin="lower", interpolation="nearest")
    ax.plot(G1[:, 0], G1[:, 1], "o", mfc="none", mec="tab:blue", ms=16, mew=1.4, label="layer 1 first-order")
    ax.plot(G2[:, 0], G2[:, 1], "s", mfc="none", mec="tab:orange", ms=16, mew=1.4, label="layer 2 first-order")
    ax.plot(gm[:, 0], gm[:, 1], "o", mfc="none", mec="tab:red", ms=7, mew=1.2, label="moire (around zero order)")
    ax.set_title(f"{mat}, twist {th} deg, {COUNTS:.0e} counts: full view")
    ax.set_xlabel("qx (1/A)")
    ax.set_ylabel("qy (1/A)")
    ax.legend(loc="upper right", fontsize=8)

    # right: zoom on the zero order
    zoom = int(2.6 * gmag / pat.dq)
    ax = axes[i, 1]
    crop = logimg[h - zoom:h + zoom + 1, h - zoom:h + zoom + 1]
    ax.imshow(crop, cmap="gray_r", extent=np.array([-zoom, zoom, -zoom, zoom]) * pat.dq,
              origin="lower", interpolation="nearest")
    ax.plot(gm[:, 0], gm[:, 1], "o", mfc="none", mec="tab:red", ms=14, mew=1.2)
    ax.set_title(f"zoom on the zero-order beam (moire spacing |g| = {gmag:.3f} 1/A)")
    ax.set_xlabel("qx (1/A)")

fig.suptitle("First-order Bragg spots of both layers and moire satellites (log counts)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_moire_with_first_order.png", dpi=120)
print("saved", OUT)
