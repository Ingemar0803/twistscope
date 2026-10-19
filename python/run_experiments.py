"""
run_experiments.py -- how reliably can the twist angle be extracted from
simulated electron diffraction of twisted bilayer graphene and MoS2?

Experiments
  1. error vs true twist angle, at several noise levels (both materials)
  2. effect of uniaxial strain on layer 2 (fixed true angle)
  3. effect of pattern mis-centering (beam tilt / misalignment)
Outputs go to ./results (CSV + PNG).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import twistsim as t

OUT = "results"
os.makedirs(OUT, exist_ok=True)

ANGLES = [1.0, 1.5, 2.0, 3.0, 5.0, 7.34, 10.0, 13.17, 15.0, 21.79, 25.0, 27.8, 30.0]
NOISE = {"low noise (1000 counts)": 1000, "medium (100)": 100, "high noise (20)": 20}
N_SEEDS = 8
MATERIALS = ["graphene", "MoS2"]

rows = []
patterns = {}

# ---------------------------------------------------------------- experiment 1
for mat in MATERIALS:
    G = t.G_first(mat)
    for th in ANGLES:
        pat = t.simulate_pattern(mat, th)
        patterns[(mat, th)] = pat
        clean = t.extract_twist(pat.I, pat.dq, G)["theta"]
        rows.append(dict(exp="clean", material=mat, true=th, noise="none", seed=-1,
                         est=clean, n_spots=12))
        for label, counts in NOISE.items():
            for s in range(N_SEEDS):
                rng = np.random.default_rng(1000 * s + int(th * 10))
                img = t.add_noise(pat, peak_counts=counts, rng=rng)
                r = t.extract_twist(img, pat.dq, G)
                rows.append(dict(exp="noise", material=mat, true=th, noise=label, seed=s,
                                 est=r["theta"], n_spots=r["n_spots"]))
    print("experiment 1 done:", mat, flush=True)

# ---------------------------------------------------------------- experiment 2
STRAINS = [0.0, 0.002, 0.005, 0.01, 0.02]
TH_STRAIN = 10.0
for mat in MATERIALS:
    G = t.G_first(mat)
    for e in STRAINS:
        pat = t.simulate_pattern(mat, TH_STRAIN, strain2=(e, 0.0))
        for s in range(N_SEEDS):
            rng = np.random.default_rng(s)
            img = t.add_noise(pat, peak_counts=1000, rng=rng)
            r = t.extract_twist(img, pat.dq, G)
            rows.append(dict(exp="strain", material=mat, true=TH_STRAIN, noise=f"strain={e}",
                             seed=s, est=r["theta"], n_spots=r["n_spots"]))
print("experiment 2 done", flush=True)

# ---------------------------------------------------------------- experiment 3
SHIFTS = [0, 1, 2, 4, 8]
for mat in MATERIALS:
    G = t.G_first(mat)
    pat = patterns[(mat, TH_STRAIN)]
    for sh in SHIFTS:
        for s in range(N_SEEDS):
            rng = np.random.default_rng(s)
            img = t.add_noise(pat, peak_counts=1000, center_shift_px=(sh * 0.6, sh * 0.8), rng=rng)
            r = t.extract_twist(img, pat.dq, G)
            rows.append(dict(exp="shift", material=mat, true=TH_STRAIN, noise=f"shift={sh}px",
                             seed=s, est=r["theta"], n_spots=r["n_spots"]))
print("experiment 3 done", flush=True)

# ---------------------------------------------------------------- save table
import csv
with open(f"{OUT}/all_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def sel(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


# ---------------------------------------------------------------- summary
print("\n=== Summary: mean absolute error (deg) vs true angle ===")
for mat in MATERIALS:
    print(f"\n{mat}")
    print(f"{'true':>6} | {'clean':>7} | " + " | ".join(f"{k.split('(')[0].strip():>10}" for k in NOISE))
    for th in ANGLES:
        c = sel(exp="clean", material=mat, true=th)[0]["est"]
        line = f"{th:6.2f} | {abs(c - th):7.3f} | "
        cells = []
        for k in NOISE:
            e = np.array([r["est"] for r in sel(exp="noise", material=mat, true=th, noise=k)])
            cells.append(f"{np.nanmean(np.abs(e - th)):10.3f}")
        print(line + " | ".join(cells))

print("\n=== strain on layer 2 at 10 deg (mean est +- std) ===")
for mat in MATERIALS:
    for e in STRAINS:
        v = np.array([r["est"] for r in sel(exp="strain", material=mat, noise=f"strain={e}")])
        print(f"{mat:9s} strain {e*100:4.1f}% -> {np.nanmean(v):7.3f} +- {np.nanstd(v):.3f}")

print("\n=== mis-centering at 10 deg ===")
for mat in MATERIALS:
    for sh in SHIFTS:
        v = np.array([r["est"] for r in sel(exp="shift", material=mat, noise=f"shift={sh}px")])
        print(f"{mat:9s} shift {sh}px -> {np.nanmean(v):7.3f} +- {np.nanstd(v):.3f}")

# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, (mat, th) in zip(axes.ravel(), [("graphene", 10.0), ("graphene", 30.0),
                                          ("MoS2", 10.0), ("MoS2", 30.0)]):
    pat = patterns[(mat, th)]
    rng = np.random.default_rng(3)
    img = t.add_noise(pat, peak_counts=300, rng=rng)
    G = t.G_first(mat)
    r = t.extract_twist(img, pat.dq, G)
    n = img.shape[0]
    ext = np.array([-n / 2, n / 2, -n / 2, n / 2]) * pat.dq
    ax.imshow(np.log10(np.clip(img, 1, None) + 1), cmap="gray_r", extent=ext, origin="lower")
    cx, cy = r["centre"]
    sp = r["spots"]
    lab = r.get("labels", np.zeros(len(sp), int))
    for k, c in enumerate(["tab:red", "tab:blue"]):
        m = lab == k
        ax.plot((sp[m, 0] + cx - n / 2) * pat.dq, (sp[m, 1] + cy - n / 2) * pat.dq,
                "o", mfc="none", mec=c, ms=9)
    ax.set_title(f"{mat}, true {th:.0f} deg, extracted {r['theta']:.2f} deg")
    ax.set_xlabel("qx (1/A)"); ax.set_ylabel("qy (1/A)")
fig.suptitle("Simulated diffraction (log scale); circles = detected first-order spots, coloured by layer")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_patterns.png", dpi=130)

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
cols = ["#1b9e77", "#d95f02", "#7570b3"]
for ax, mat in zip(axes, MATERIALS):
    for (k, _), c in zip(NOISE.items(), cols):
        mae = []
        for th in ANGLES:
            e = np.array([r["est"] for r in sel(exp="noise", material=mat, true=th, noise=k)])
            mae.append(np.nanmean(np.abs(e - th)))
        ax.plot(ANGLES, np.maximum(mae, 1e-3), "o-", color=c, label=k)
    ax.set_yscale("log")
    ax.set_xlabel("true twist angle (deg)")
    ax.set_title(mat)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mean absolute error (deg)")
axes[0].legend()
fig.suptitle("Twist-angle extraction error from simulated electron diffraction")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_error_vs_angle.png", dpi=130)

# ---------------------------------------------------------------- figure 3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for mat, c in zip(MATERIALS, cols):
    m = [np.nanmean([r["est"] for r in sel(exp="strain", material=mat, noise=f"strain={e}")]) for e in STRAINS]
    axes[0].plot(np.array(STRAINS) * 100, m, "o-", color=c, label=mat)
    m2 = [np.nanmean([r["est"] for r in sel(exp="shift", material=mat, noise=f"shift={s}px")]) for s in SHIFTS]
    axes[1].plot(SHIFTS, m2, "o-", color=c, label=mat)
axes[0].axhline(TH_STRAIN, color="k", ls="--", lw=1)
axes[0].set_xlabel("uniaxial strain on layer 2 (%)"); axes[0].set_ylabel("extracted angle (deg)")
axes[0].set_title("Strain bias at true 10 deg"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].axhline(TH_STRAIN, color="k", ls="--", lw=1)
axes[1].set_xlabel("pattern shift (units of 1 px step)"); axes[1].set_title("Mis-centering at true 10 deg")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_strain_shift.png", dpi=130)
print("\nsaved figures and CSV in", OUT)
