"""
real_saed_extract.py -- extract the twist angle from REAL electron-diffraction
(SAED) images of twisted bilayer graphene, using the same idea as twistsim.py:
spots of the two layers lie on the same rings, and their azimuths (taken modulo
60 deg) fall into two groups separated by the twist angle.

Input here: a screenshot of the published figure ("Raman spectra of twisted CVD
bilayer graphene", Carbon 123, 302 (2017); panels (a)-(c)). Because it is a
figure, not raw detector data, the script has to cope with:
  * a beam stop hiding the centre      -> centre found from Friedel pairs (g, -g)
  * a bright diffuse halo              -> large-scale background subtracted
  * coloured hexagons/labels drawn on top by the authors
                                       -> use the darkest RGB channel, so only
                                          white (or, for panel a, black) spots survive
  * low resolution (~300 px per panel) -> limits the precision
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "real_data/carbon2017_fig_saed.png"
# panel boxes (x0, y0, x1, y1) in the 608 x 548 screenshot; 'dark' = white spots on black
PANELS = {
    "a": dict(box=(24, 47, 292, 282), dark=False, reported=None),
    "b": dict(box=(302, 47, 582, 284), dark=True, reported=16.0),
    "c": dict(box=(28, 290, 296, 528), dark=True, reported=24.0),
}
LABEL_BOX = (70, 42)
ASPECT = "--no-aspect" not in sys.argv     # correct horizontal stretch of the figure image        # top-left panel-letter box to ignore (w, h) in px


def spot_image(rgb, dark):
    """Return a blob-filtered image (point-like spots bright, lines weak) and a
    mask of regions to ignore (beam stop and its edge, drawn yellow labels)."""
    rgb = rgb.astype(float)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h, w = R.shape
    if dark:
        sig = rgb.min(axis=2)                        # white spots survive; coloured lines/text vanish
        darkpix = rgb.max(axis=2) < 25
        lab, n = ndi.label(darkpix)
        c = lab[h // 2, w // 2]                      # the beam stop covers the pattern centre
        stop = (lab == c) if c > 0 else np.zeros_like(darkpix)
        stop = ndi.binary_dilation(stop, iterations=12)
    else:
        sig = 255 - rgb.max(axis=2)                  # dark spots on white; blue lines vanish
        stop = np.zeros((h, w), bool)
    # anything the authors drew on top (coloured hexagons, vertex markers, yellow labels)
    drawn = (rgb.max(axis=2) - rgb.min(axis=2)) > 50
    stop |= ndi.binary_dilation(drawn, iterations=4)
    blob = -ndi.gaussian_laplace(sig, sigma=2.0)    # Laplacian-of-Gaussian: strongest for round spots
    return blob, stop


def find_spots(res, stop, nsig=6.0, edge=6, rel=0.25):
    h, w = res.shape
    mx = ndi.maximum_filter(res, size=9)
    vals = res[~stop]
    med = np.median(vals)
    sig = 1.4826 * np.median(np.abs(vals - med))
    cand = (res == mx) & (res > med + nsig * sig) & ~stop
    cand[:edge, :] = cand[-edge:, :] = cand[:, :edge] = cand[:, -edge:] = False
    cand[:LABEL_BOX[1], :LABEL_BOX[0]] = False
    ys, xs = np.nonzero(cand)
    if len(ys) == 0:
        return np.empty((0, 3))
    peak = res[ys, xs]
    keep = peak > rel * np.percentile(peak, 90)       # drop weak line/text fragments
    ys, xs = ys[keep], xs[keep]
    pts = []
    for y0, x0 in zip(ys, xs):
        y1, y2, x1, x2 = max(y0 - 2, 0), min(y0 + 3, h), max(x0 - 2, 0), min(x0 + 3, w)
        p = np.clip(res[y1:y2, x1:x2], 0, None)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        pts.append(((xx * p).sum() / p.sum(), (yy * p).sum() / p.sum(), res[y0, x0]))
    return np.array(pts)


def centre_from_pairs(xy, tol=3.0, min_sep=30):
    """Friedel pairs share their midpoint (the hidden centre). Return the
    midpoint shared by the most pairs, refined as the mean of that cluster."""
    n = len(xy)
    mids = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(xy[i] - xy[j])) > min_sep:
                mids.append((xy[i] + xy[j]) / 2)
    mids = np.array(mids)
    d = np.hypot(mids[:, None, 0] - mids[None, :, 0], mids[:, None, 1] - mids[None, :, 1])
    counts = (d < tol).sum(1)
    best = np.argmax(counts)
    cl = mids[d[best] < tol]
    return cl.mean(axis=0), counts[best]


def split_rings(r, gap=0.12):
    order = np.argsort(r)
    rings, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if (r[b] - r[a]) / r[a] > gap:
            rings.append(cur)
            cur = []
        cur.append(b)
    rings.append(cur)
    return rings


def fit_aspect(dx, dy, rmin=30.0):
    """Find the horizontal stretch k of the image (x_true = x / k) that makes the
    diffraction rings most circular: minimise the relative radial scatter of the
    spots in the most populated ring. k = 1 means no distortion."""
    sel = np.hypot(dx, dy) > rmin
    best = (np.inf, 1.0)
    for k in np.arange(0.70, 1.40001, 0.002):
        r = np.hypot(dx[sel] / k, dy[sel])
        rings = split_rings(r, gap=0.10)
        ring = max(rings, key=len)
        if len(ring) < 5:          # Friedel pairs alone always look circular; need >= 3 non-collinear directions
            continue
        sc = r[ring].std() / r[ring].mean()
        if sc < best[0]:
            best = (sc, k)
    return best[1], best[0]


def two_groups(phi):
    """Cluster azimuths modulo 60 deg into two groups (circular 2-means)."""
    psi = np.angle(np.exp(6j * phi))
    d = np.abs(np.angle(np.exp(1j * (psi[:, None] - psi[None, :]))))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    c = np.array([psi[i], psi[j]])
    for _ in range(50):
        lab = np.argmin(np.abs(np.angle(np.exp(1j * (psi[:, None] - c[None, :])))), axis=1)
        new = np.array([np.angle(np.mean(np.exp(1j * psi[lab == k]))) if np.any(lab == k) else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    theta = np.degrees(abs(np.angle(np.exp(1j * (c[0] - c[1]))))) / 6
    # scatter of each group about its mean (deg), and standard error of theta
    sd, n = [], []
    for k in range(2):
        dev = np.degrees(np.angle(np.exp(1j * (psi[lab == k] - c[k])))) / 6
        sd.append(dev.std(ddof=1) if (lab == k).sum() > 1 else np.nan)
        n.append((lab == k).sum())
    se = np.sqrt(sum((s ** 2) / m for s, m in zip(sd, n) if m > 1 and not np.isnan(s)))
    return theta, lab, sd, n, se


if __name__ == "__main__":
    img = np.array(Image.open(SRC).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="#fcfcfb")
    results = {}
    for ax, (name, P) in zip(axes, PANELS.items()):
        x0, y0, x1, y1 = P["box"]
        crop = img[y0:y1, x0:x1]
        res, stop = spot_image(crop, P["dark"])
        pts = find_spots(res, stop)
        xy = pts[:, :2]
        (cx, cy), npair = centre_from_pairs(xy)
        dx, dy = xy[:, 0] - cx, -(xy[:, 1] - cy)          # y up
        k_raw = 1.0
        if ASPECT:
            k_raw, sc = fit_aspect(dx, dy)
            if np.isfinite(sc):
                print(f"  image horizontal stretch fitted: k = {k_raw:.3f} (ring scatter after correction {100*sc:.1f}%)")
                dx = dx / k_raw
            else:
                print("  image stretch could not be fitted (too few spots); no correction applied")
        r = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx)
        keep = r > 15                                     # drop anything at the centre
        rings = split_rings(r[keep])
        idx_keep = np.nonzero(keep)[0]
        print(f"\n=== panel ({name}) reported {P['reported']} deg ===")
        print(f"spots found {len(xy)}, centre ({cx:.1f}, {cy:.1f}) px from {npair} coincident Friedel-pair midpoints")
        ax.imshow(crop)
        ax.plot(cx, cy, "+", color="#e34948", ms=14, mew=2)
        ring_results = []
        for k, ring in enumerate(rings):
            ids = idx_keep[ring]
            print(f"  ring {k}: {len(ids)} spots, radius {r[ids].mean():.1f} +- {r[ids].std():.1f} px")
            if len(ids) < 4:
                ax.plot(xy[ids, 0], xy[ids, 1], "x", color="#aaaaaa", ms=7)
                continue
            Rbar = abs(np.mean(np.exp(6j * phi[ids])))
            if Rbar > 0.95:                                   # all spots share one 60-deg orientation
                print(f"    one orientation only (6-fold order parameter {Rbar:.3f}): single layer, or twist < ~5 deg")
                ax.plot(xy[ids, 0], xy[ids, 1], "x", color="#aaaaaa", ms=7)
                continue
            th, lab, sd, n, se = two_groups(phi[ids])
            ring_results.append((th, se, len(ids), n))
            print(f"    twist = {th:.2f} deg  (+- {se:.2f} deg std. error; groups {n[0]} and {n[1]} spots, scatter {sd[0]:.2f}/{sd[1]:.2f} deg)")
            ax.plot(xy[ids, 0], xy[ids, 1], "x", color="#aaaaaa", ms=7)
        # measurement ring: >= 5 spots with the smallest relative radial scatter
        cand = [(r[idx_keep[ring]].std() / r[idx_keep[ring]].mean(), k) for k, ring in enumerate(rings) if len(ring) >= 5]
        best = min(cand)[1] if cand else None
        chosen = None
        if best is not None:
            ids = idx_keep[rings[best]]
            Rbar = abs(np.mean(np.exp(6j * phi[ids])))
            if Rbar > 0.95:
                chosen = ("single", Rbar, len(ids))
                ax.plot(xy[ids, 0], xy[ids, 1], "o", mfc="none", mec="#2a78d6", ms=11, mew=2)
            else:
                th, lab, sd, n, se = two_groups(phi[ids])
                chosen = ("twist", th, se, len(ids), n)
                for g, col in ((0, "#2a78d6"), (1, "#eb6834")):
                    m = lab == g
                    ax.plot(xy[ids][m, 0], xy[ids][m, 1], "o", mfc="none", mec=col, ms=11, mew=2)
        results[name] = chosen
        if chosen is None:
            ax.set_title(f"({name}) no clean ring found", fontsize=11)
        elif chosen[0] == "single":
            print(f"  -> measurement ring {best}: one orientation (order parameter {chosen[1]:.3f}) = single layer")
            ax.set_title(f"({name}) one set of spots: single layer", fontsize=11)
        else:
            _, th, se, nn, n = chosen
            print(f"  -> measurement ring {best}: twist {th:.2f} +- {se:.2f} deg from {nn} spots ({n[0]} + {n[1]})   reported {P['reported']}")
            ax.set_title(f"({name}) extracted {th:.1f} ± {se:.1f}°   (paper: {P['reported']:.0f}°)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Twist angle extracted by my algorithm from published SAED patterns (Carbon 123, 302, 2017). "
                 "Circles: spots used, coloured by layer; grey x: rejected detections; + = centre from Friedel pairs. Image stretch corrected.", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("real_data/fig_real_saed_extraction.png" if ASPECT else "real_data/fig_real_saed_extraction_raw.png", dpi=140)
    print("\nsaved real_data/fig_real_saed_extraction.png")
