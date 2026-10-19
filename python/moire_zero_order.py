"""
moire_zero_order.py -- show the moire satellite spots around the zero-order
(specular) beam by raising the counts far beyond the earlier "low noise" level.

Reminder: in this model the moire spots exist only because of the
phenomenological interlayer double-scattering term (alpha) in twistsim.py.
They sit at the right positions (G2 - G1) by construction of the geometry, but
their strength (~1e-5 of a first-order spot) is set by alpha, not by physics.
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
COUNTS = [1e3, 1e5, 1e7]            # mean counts at the brightest first-order spot
L, DX, SIGMA_W, ALPHA = 900.0, 0.3, 90.0, 0.25


def moire_points(material, theta):
    """Predicted moire reciprocal points g = R(theta) G - G for the six
    first-order reciprocal vectors G (1/A, shape (6,2))."""
    a = t.MATERIALS[material]["a"]
    amat = np.array([[a, 0.0], [a / 2, a * np.sqrt(3) / 2]])      # rows = a1, a2
    b = 2 * np.pi * np.linalg.inv(amat)                           # columns = b1, b2
    b1, b2 = b[:, 0], b[:, 1]
    Gs = np.array([b1, b2, b1 + b2, -b1, -b2, -(b1 + b2)])
    th = np.radians(theta)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Gs @ R.T - Gs


def significance(img, pat, pts):
    """Excess counts of each predicted moire point over the local background
    (median in an annulus around the zero order), in units of Poisson sigma."""
    n = img.shape[0]
    h = n // 2
    Y, X = np.mgrid[:n, :n]
    r = np.hypot(Y - h, X - h) * pat.dq
    gm = np.hypot(*pts[0])
    ann = (r > 0.5 * gm) & (r < 1.5 * gm)
    bkg = np.median(img[ann])
    res = []
    for gx, gy in pts:
        x0, y0 = int(round(h + gx / pat.dq)), int(round(h + gy / pat.dq))
        win = img[y0 - 1:y0 + 2, x0 - 1:x0 + 2]
        pk = win.max()
        res.append((pk - bkg) / np.sqrt(max(pk, 1.0)))
    return bkg, np.array(res)


fig, axes = plt.subplots(len(CASES), len(COUNTS), figsize=(12, 11))
print(f"{'case':>14} | {'counts':>8} | {'bkg(ring median)':>16} | {'moire excess/sigma (min..max of 6)':>34}")
for i, (mat, th) in enumerate(CASES):
    pat = t.simulate_pattern(mat, th, L=L, dx=DX, sigma_w=SIGMA_W, alpha=ALPHA)
    pts = moire_points(mat, th)
    gm = np.hypot(*pts[0])
    n = pat.I.shape[0]
    h = n // 2
    zoom = int(2.6 * gm / pat.dq)                 # show |q| up to ~2.6 g_m
    for j, cnt in enumerate(COUNTS):
        rng = np.random.default_rng(7)
        img = t.add_noise(pat, peak_counts=cnt, rng=rng)
        bkg, sig = significance(img, pat, pts)
        print(f"{mat+' '+str(th)+' deg':>14} | {cnt:8.0e} | {bkg:16.1f} | {sig.min():12.1f} .. {sig.max():.1f}")
        ax = axes[i, j]
        crop = img[h - zoom:h + zoom + 1, h - zoom:h + zoom + 1]
        ext = np.array([-zoom, zoom, -zoom, zoom]) * pat.dq
        ax.imshow(np.log10(np.clip(crop, 1, None)), cmap="gray_r", extent=ext, origin="lower",
                  interpolation="nearest")
        ax.plot(pts[:, 0], pts[:, 1], "o", mfc="none", mec="tab:red", ms=14, mew=1.2)
        ax.set_title(f"{mat} {th} deg, {cnt:.0e} counts", fontsize=10)
        if j == 0:
            ax.set_ylabel("qy (1/A)")
        if i == len(CASES) - 1:
            ax.set_xlabel("qx (1/A)")
fig.suptitle("Zoom on the zero-order beam (log counts). Red circles = predicted moire points g = R(theta)G - G.\n"
             "Moire spots exist here only via the phenomenological alpha term; detector saturation not modelled.",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_moire_zero_order.png", dpi=130)
print("saved", OUT)
