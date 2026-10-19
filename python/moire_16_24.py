"""
moire_16_24.py -- switch on the phenomenological double-scattering term (alpha)
for the 16 and 24 deg twisted bilayer graphene simulations, zoom on the central
beam to show the moire spots, and mark where those spots would fall on the real
SAED images from the Carbon paper.

alpha is NOT a measured physical constant (see twistsim.py): it only sets how
bright the moire spots are. Their POSITIONS, g_m = R(theta) G - G, are pure
geometry and do not depend on alpha.
"""
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import twistsim as t
import sim_vs_real as S

G1 = t.G_first("graphene")
ALPHAS = [0.0, 0.25, 1.0, 3.0]
CASES = [("b", 16.0), ("c", 24.0)]
RED, BLUE, ORANGE, SURF, INK2 = "#e34948", "#2a78d6", "#eb6834", "#fcfcfb", "#52514e"


def first_order(theta_deg=0.0):
    """Six first-order reciprocal vectors of graphene, layer rotated by theta (sim frame: layer 1 at 30 + k*60 deg)."""
    a = 2.46
    amat = np.array([[a, 0.0], [a / 2, a * np.sqrt(3) / 2]])
    b = 2 * np.pi * np.linalg.inv(amat)
    b1, b2 = b[:, 0], b[:, 1]
    G = np.array([b1, b2, b1 + b2, -b1, -b2, -(b1 + b2)])
    th = np.radians(theta_deg)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return G @ Rm.T


def peak_near(I, dq, q, win=3):
    n = I.shape[0]; h = n // 2
    ix, iy = int(round(q[0] / dq + h)), int(round(q[1] / dq + h))
    return I[iy - win:iy + win + 1, ix - win:ix + win + 1].max()


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.6), facecolor=SURF,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.05]})
    for row, (panel, th) in enumerate(CASES):
        Ga, Gb = first_order(0.0), first_order(th)
        gm = Gb - Ga                                        # moire reciprocal vectors
        gm_len = np.hypot(*gm[0])
        period = 4 * np.pi / (np.sqrt(3) * gm_len)      # moire superlattice period = a / (2 sin(theta/2))
        print(f"\n=== {th:g} deg: moire spots at |g_m| = {gm_len:.3f} 1/A = {gm_len/G1:.3f} x first-order; moire period {period:.1f} A ===")
        for col, al in enumerate(ALPHAS):
            pat = t.simulate_pattern("graphene", th, alpha=al, qmax_factor=1.3)
            I = pat.I
            n = I.shape[0]; h = n // 2
            first = max(peak_near(I, pat.dq, g) for g in Ga)
            moire = np.mean([peak_near(I, pat.dq, g) for g in gm])
            bkg = np.median(I)
            ratio = moire / first
            print(f"  alpha = {al:4.2f}: moire / first-order spot intensity = {ratio:.2e}   (moire / background = {moire/bkg:.1e})")
            ax = axes[row, col]
            disp = np.log10(ndi.gaussian_filter(I, 2.5) / ndi.gaussian_filter(I, 2.5).max() + 1e-8)
            lim = np.array([-h, h, -h, h]) * pat.dq
            ax.imshow(disp, cmap="gray", origin="lower", extent=lim, vmin=-7, vmax=0)
            ax.plot(gm[:, 0], gm[:, 1], "o", mfc="none", mec=RED, ms=16, mew=1.6)
            ax.set_xlim(-1.2 * G1, 1.2 * G1); ax.set_ylim(-1.2 * G1, 1.2 * G1)
            ax.set_title(f"{th:g}°, alpha = {al:g}\nmoire/first-order = {ratio:.1e}" if al > 0 else f"{th:g}°, alpha = 0 (single scattering)\nno moire spots", fontsize=10)
            ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
            if col == 0:
                ax.set_ylabel("$q_y$ (1/Å)")
            if row == 1:
                ax.set_xlabel("$q_x$ (1/Å)")

        # ---------- predicted moire positions on the real image ----------
        rs = S.real_spots(panel)
        scale = G1 / rs["r"].mean()                       # 1/A per corrected px
        c0, c1 = np.degrees(rs["cen"])
        if abs(S.wrap60(c1 - c0 - rs["theta"])) < abs(S.wrap60(c1 - c0 + rs["theta"])):
            ref = 0
        else:
            ref = 1
        rot = 30.0 - np.degrees(rs["cen"][ref])           # real -> sim rotation (deg)
        def to_image(q):
            phi = np.degrees(np.arctan2(q[:, 1], q[:, 0])) - rot
            r = np.hypot(q[:, 0], q[:, 1]) / scale
            x = rs["cx"] + r * np.cos(np.radians(phi)) * rs["k"]
            y = rs["cy"] - r * np.sin(np.radians(phi))
            return x, y
        ax = axes[row, 4]
        ax.imshow(rs["crop"])
        fx, fy = to_image(np.vstack([Ga, Gb]))
        ax.plot(fx, fy, "o", mfc="none", mec="#fcfcfb", ms=11, mew=1.2, label="predicted first-order spots")
        mx, my = to_image(gm)
        ax.plot(mx, my, "o", mfc="none", mec=RED, ms=11, mew=2, label="predicted moire spots")
        # how many predicted moire positions land on the beam stop?
        crop = rs["crop"].astype(float)
        hidden = 0
        for x_, y_ in zip(mx, my):
            xi, yi = int(round(x_)), int(round(y_))
            if 0 <= yi < crop.shape[0] and 0 <= xi < crop.shape[1] and crop[yi, xi].max() < 40:
                hidden += 1
        print(f"  real image: {hidden} of 6 predicted moire positions fall on the (black) beam stop; "
              f"moire radius {gm_len/scale:.1f} px vs first-order {rs['r'].mean():.1f} px")
        ax.set_title(f"Real pattern ({panel}): predicted moire positions\n{hidden} of 6 lie on the beam stop", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.suptitle("Moire spots appear around the central beam only when double scattering (alpha) is switched on. "
                 "Red circles = predicted moire positions g = R(θ)G − G (geometry only).", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("real_data/fig_moire_16_24.png", dpi=120)
    print("\nsaved real_data/fig_moire_16_24.png")
