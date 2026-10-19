"""
sim_vs_real.py -- the reverse test: simulate the electron-diffraction patterns of
twisted bilayer graphene at the twist angles reported in the Carbon paper
(16 and 24 deg) and compare them with the real SAED patterns.

Simulation: twistsim.simulate_pattern (kinematic theory, graphene a = 2.46 A),
alpha = 0, i.e. single scattering only (no extra double-scattering spots).
Comparison: the real spots found by real_saed_extract.py (inner ring, image
stretch corrected) are scaled so their ring radius equals the graphene first-
order reciprocal vector, and rotated so that one real layer lines up with
simulated layer 1. The other layer is then a genuine test: the simulation puts
it at +16 or +24 deg, the data put it wherever it really is.
"""
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import twistsim as t
import real_saed_extract as R

SRC = R.SRC
CASES = [("b", 16.0), ("c", 24.0)]
G1 = t.G_first("graphene")                       # 2.949 1/A
INK2, BLUE, ORANGE, SURF = "#52514e", "#2a78d6", "#eb6834", "#fcfcfb"


def real_spots(panel):
    """Return real inner-ring spots (y up, stretch corrected), their layer
    labels, the two layer orientations (rad, mod 60 deg), and the crop."""
    P = R.PANELS[panel]
    img = np.array(Image.open(SRC).convert("RGB"))
    x0, y0, x1, y1 = P["box"]
    crop = img[y0:y1, x0:x1]
    res, stop = R.spot_image(crop, P["dark"])
    pts = R.find_spots(res, stop)
    xy = pts[:, :2]
    (cx, cy), _ = R.centre_from_pairs(xy)
    dx, dy = xy[:, 0] - cx, -(xy[:, 1] - cy)
    k, _ = R.fit_aspect(dx, dy)
    dx = dx / k
    r = np.hypot(dx, dy)
    keep = np.nonzero(r > 15)[0]
    rings = R.split_rings(r[keep])
    cand = [(r[keep[g]].std() / r[keep[g]].mean(), i) for i, g in enumerate(rings) if len(g) >= 5]
    ids = keep[rings[min(cand)[1]]]
    phi = np.arctan2(dy[ids], dx[ids])
    theta, lab, *_ = R.two_groups(phi)
    psi = 6 * phi
    cen = [np.angle(np.mean(np.exp(1j * psi[lab == g]))) / 6 for g in (0, 1)]   # layer orientation mod 60 deg
    return dict(dx=dx[ids], dy=dy[ids], r=r[ids], lab=lab, cen=cen, theta=theta, crop=crop, k=k, cx=cx, cy=cy)


def wrap60(a):
    """Wrap an angle (deg) into [-30, 30)."""
    return (a + 30) % 60 - 30


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), facecolor=SURF,
                             gridspec_kw={"width_ratios": [1, 1.12, 1.12]})
    for row, (panel, th) in enumerate(CASES):
        # ---------- simulate ----------
        pat = t.simulate_pattern("graphene", th, alpha=0.0, qmax_factor=1.95)
        ext = t.extract_twist(pat.I, pat.dq, G1)
        n = pat.I.shape[0]
        h = n // 2
        lim = np.array([-h, h, -h, h]) * pat.dq
        print(f"\n=== {th:g} deg ===")
        print(f"simulated pattern: {n} x {n} px, |q| up to {h*pat.dq:.2f} 1/A; extractor on the simulation -> {ext['theta']:.3f} deg")

        # ---------- real spots, mapped into simulation coordinates ----------
        rs = real_spots(panel)
        scale = G1 / rs["r"].mean()                        # px -> 1/A
        # simulated layer 1 has first-order spots at 30 + k*60 deg; layer 2 at 30 + th + k*60.
        # choose which real group is "layer 1": the one from which the other group sits at +theta (mod 60)
        c0, c1 = np.degrees(rs["cen"])
        if abs(wrap60(c1 - c0 - rs["theta"])) < abs(wrap60(c1 - c0 + rs["theta"])):
            ref, other = 0, 1
        else:
            ref, other = 1, 0
        rot = 30.0 - np.degrees(rs["cen"][ref])
        phi_r = np.degrees(np.arctan2(rs["dy"], rs["dx"])) + rot
        qx = rs["r"] * scale * np.cos(np.radians(phi_r))
        qy = rs["r"] * scale * np.sin(np.radians(phi_r))
        # deviation of each real spot from the nearest simulated spot of its own layer
        pred = {ref: 30.0, other: 30.0 + th}
        dev = np.array([wrap60(p - pred[l]) for p, l in zip(phi_r, rs["lab"])])
        dev_other = dev[rs["lab"] == other]
        print(f"real data: stretch k = {rs['k']:.3f}, twist from data {rs['theta']:.2f} deg, {len(dev)} spots")
        print(f"  layer used for alignment: rms deviation {np.sqrt(np.mean(dev[rs['lab']==ref]**2)):.2f} deg (only scatter; its mean is zero by construction)")
        print(f"  second layer (the test):  mean offset {dev_other.mean():+.2f} deg, rms {np.sqrt(np.mean(dev_other**2)):.2f} deg  vs simulation at {th:g} deg")
        print(f"  ring radius: predicted |G| = {G1:.3f} 1/A for a = 2.46 A (used to set the scale, not tested)")

        # ---------- plot: real ----------
        ax = axes[row, 0]
        ax.imshow(rs["crop"])
        ax.set_title(f"Real SAED, Carbon paper panel ({panel}), reported {th:g}°", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])

        # ---------- plot: simulation (SAED-like rendering) ----------
        ax = axes[row, 1]
        # blur the sharp simulated spots to a visible size (finite beam / detector resolution)
        blurred = ndi.gaussian_filter(pat.I, 4.0)
        disp = np.log10(blurred / blurred.max() + 1e-5)
        # rotate the simulation into the orientation of the real pattern; pick the rotation
        # direction that puts bright spots where the real spots are (self-check of the sign)
        phi_real = np.degrees(np.arctan2(rs["dy"], rs["dx"]))
        qxr = rs["r"] * scale * np.cos(np.radians(phi_real))
        qyr = rs["r"] * scale * np.sin(np.radians(phi_real))
        best = None
        for sgn in (+1, -1):
            img_r = ndi.rotate(disp, sgn * rot, reshape=False, order=1, cval=disp.min())
            ix = np.clip(np.round(qxr / pat.dq + h).astype(int), 0, n - 1)
            iy = np.clip(np.round(qyr / pat.dq + h).astype(int), 0, n - 1)
            hit = img_r[iy, ix].mean() - np.median(img_r)
            if best is None or hit > best[0]:
                best = (hit, sgn, img_r)
        print(f"  display rotation sign {best[1]:+d} (real spots fall on simulated spots: brightness excess {best[0]:.2f} decades)")
        ax.imshow(best[2], cmap="gray", origin="lower", extent=lim, vmin=-4.5, vmax=0)
        ax.set_xlim(-1.85 * G1, 1.85 * G1); ax.set_ylim(-1.85 * G1, 1.85 * G1)
        ax.set_title(f"Simulated {th:g}° twisted bilayer graphene\n(kinematic theory, rotated to match the real pattern)", fontsize=10.5)
        ax.set_xlabel("$q_x$ (1/Å)"); ax.set_ylabel("$q_y$ (1/Å)")

        # ---------- plot: overlay ----------
        ax = axes[row, 2]
        ax.imshow(disp, cmap="gray", origin="lower", extent=lim, vmin=-4.5, vmax=0)
        for l, col, name in ((ref, BLUE, "real layer used for alignment"), (other, ORANGE, "real second layer (test)")):
            m = rs["lab"] == l
            ax.plot(qx[m], qy[m], "o", mfc="none", mec=col, ms=13, mew=2.2, label=name)
        ax.set_xlim(-1.5 * G1, 1.5 * G1); ax.set_ylim(-1.5 * G1, 1.5 * G1)
        ax.set_title(f"Real spots on the simulation: second layer off by {dev_other.mean():+.1f}°", fontsize=10.5)
        ax.set_xlabel("$q_x$ (1/Å)")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
    fig.suptitle("Reverse test: diffraction patterns simulated at the reported twist angles vs the real patterns "
                 "(real images: Carbon 123, 302, 2017)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("real_data/fig_sim_vs_real_16_24.png", dpi=130)
    print("\nsaved real_data/fig_sim_vs_real_16_24.png")
