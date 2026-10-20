"""Compare TwistScope's 30 deg twisted bilayer graphene simulation with the LEED pattern of
Dai et al., PRL 127, 086805 (2021), Fig. 1(d). Only spot POSITIONS are compared
(ratios |g|/|G| and angles); kinematic intensities do not apply to low-energy LEED."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import twistsim as t

# ---------- measured: Dai et al. figure ----------
im = 255 - np.array(Image.open("real_data/dai2021_fig1d_LEED_30deg_tBLG.png").convert("L")).astype(float)
lab, n = ndi.label(im > 200)
blobs = [(np.nonzero(lab == i)) for i in range(1, n + 1)]
big = sorted(blobs, key=lambda b: -len(b[0]))
cy, cx = big[0][0].mean(), big[0][1].mean()           # (00) beam
sm = ndi.gaussian_filter(im, 1.2)
res = sm - ndi.median_filter(sm, size=15)
pk = (res == ndi.maximum_filter(res, size=7)) & (res > 12)
ys, xs = np.nonzero(pk)
r = np.hypot(xs - cx, ys - cy); a = np.degrees(np.arctan2(-(ys - cy), xs - cx))
amp = res[ys, xs]
# bright first-order spots: 6 strongest blobs at r 120-160 (exclude the text labels by area/shape)
first = []
for yy, xx in big[1:]:
    rr = np.hypot(xx.mean() - cx, yy.mean() - cy)
    if 115 < rr < 165 and 25 < len(yy) < 400:
        first.append((rr, np.degrees(np.arctan2(-(yy.mean() - cy), xx.mean() - cx)), len(yy)))
first = sorted(first, key=lambda f: -f[2])[:6]
G = np.mean([f[0] for f in first]); angs1 = np.sort([f[1] % 360 for f in first])
print(f"(00) centre {cx:.1f},{cy:.1f}; bright first-order ring r = {G:.1f} px, sd {np.std([f[0] for f in first]):.1f}; angles {np.round(angs1,1)}")
def ring(lo, hi):
    m = (r / G > lo) & (r / G < hi)
    return r[m] / G, a[m] % 360
for lo, hi, name in [(0.45, 0.60, "inner ring"), (0.66, 0.80, "middle ring")]:
    rr, aa = ring(lo, hi)
    psi = np.angle(np.mean(np.exp(1j * np.radians(aa) * 12)))   # 12-fold orientation
    print(f"{name}: {len(rr)} spots, |g|/|G| = {rr.mean():.3f} +- {rr.std(ddof=1)/np.sqrt(len(rr)):.3f}; 12-fold spacing check: {np.round(np.sort(aa),0)}")
# weak second-layer first-order spots: peaks on the G ring at ~30 deg from bright ones
m = (np.abs(r / G - 1) < 0.07)
off = []
for ang in a[m] % 360:
    d = np.min(np.abs(((ang - angs1) + 180) % 360 - 180))
    if 20 < d < 40: off.append(d)
print(f"weak spots on first-order ring rotated from bright ones by {np.mean(off):.1f} +- {np.std(off, ddof=1)/np.sqrt(len(off)):.1f} deg (n={len(off)})")

# ---------- simulated: TwistScope ----------
Gs = t.G_first("graphene")
for alpha in (0.0, 1.0):
    p = t.simulate_pattern("graphene", 30.0, alpha=alpha, qmax_factor=1.3)
    I = p.I; h = I.shape[0] // 2
    yy, xx = np.mgrid[:I.shape[0], :I.shape[0]]
    q = np.hypot(yy - h, xx - h) * p.dq / Gs
    peak = (I == ndi.maximum_filter(I, size=5)) & (q > 0.2) & (q < 1.2) & (I > I.max() * 1e-7)
    qs = np.round(q[peak], 3); Is = I[peak]
    first_I = Is[np.abs(qs - 1) < 0.02].mean()
    out = {}
    for target in (0.518, 0.732, 1.0):
        sel = np.abs(qs - target) < 0.02
        out[target] = (sel.sum(), qs[sel].mean() if sel.any() else np.nan, Is[sel].mean() / first_I if sel.any() else 0)
    print(f"sim alpha={alpha}: " + "; ".join(f"|g|/|G|~{k}: n={v[0]}, mean {v[1]:.3f}, I/I1={v[2]:.1e}" for k, v in out.items()))
print("theory: 2 sin 15 deg =", round(2*np.sin(np.radians(15)),3), "; sqrt3 - 1 =", round(np.sqrt(3)-1,3))
