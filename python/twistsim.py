"""
twistsim.py -- kinematic electron-diffraction simulator for twisted bilayers
of hexagonal 2D materials (graphene, MoS2), plus a twist-angle extractor.

Model
-----
Each layer is projected along the beam (normal incidence), so a layer is a 2D
set of point scatterers. The diffraction amplitude is

    A(q) = FT[V1 + V2] + alpha * FT[V1 * V2]

* V1, V2 : projected potential of each layer (atoms smeared by a Gaussian
           form factor, multiplied by a Gaussian aperture of width sigma_w).
* The linear term is ordinary (single) scattering: two rotated hexagonal sets
  of spots.
* The product term is a *phenomenological* stand-in for interlayer (double)
  scattering: a product in real space is a convolution in q, so it creates
  spots at sums of the two layers' reciprocal vectors (the weak 12-fold
  "fractional order" spots seen at 30 deg in Dai et al.). alpha is a free
  strength, not a fitted physical constant.

This is NOT a dynamical (multiple-scattering) LEED calculation.

Parameter provenance is recorded in PAPER below. The graphene lattice
constant (2.46 A) is a standard literature value, NOT taken from the three
papers.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------
# Constants and data taken from the papers
# ----------------------------------------------------------------------------
H2_OVER_2M_EV_A2 = 150.4   # h^2/(2 m_e) in eV*A^2  ->  lambda[A] = 12.264/sqrt(E[eV])

PAPER = {
    # Huang et al., Nano Lett. 16, 1435 (2016)  -- twisted bilayer MoS2
    "MoS2": {
        "a_A": 3.13,                              # in-plane lattice constant
        "interlayer_Mo_Mo_A_mismatched": 6.36,    # DFT, 20-40 deg
        "commensurate_angles_deg": [7.3, 13.2, 21.8, 27.8, 32.2, 38.2, 46.8, 52.7],
        "supercell_A": {21.8: 8.29, 27.8: 11.29, 13.2: 13.65, 7.3: 24.41},
        "raman_LF_2H_cm1": {"shear": 22.9, "breathing": 38.1},
        "raman_LF_variation_cm1": {"shear_max": 8.0, "breathing_max_near_0_60": 6.7},
        "raman_breathing_20_40_cm1": 37.0,
        "shear_DFT_20_40_cm1": 2.0,
        "force_const_Kb_1e19_N_m3": (6.26, 9.52),
        "force_const_Ks_1e19_N_m3": (2.30, 4.51),
    },
    # Dai et al., Phys. Rev. Lett. 127, 086805 (2021) -- 30 deg tBLG, LEEM/LEED
    "graphene_tBLG": {
        "d_AB_A": 3.34,
        "dE_AB_eV": 2.6,
        "dE_30deg_eV": 2.3,
        "d_ratio_paper": 1.06,
        "d_increase_paper_A": 0.2,
        "lambda_3eV_paper_A": 7.0,
        "LEED_energy_eV": 42.0,
        "LEED_features": "12 first-order spots + 12 weaker fractional-order spots",
        "raman_2D_enhancement": 4.0,
        "fano_G": {"SLG": (0.33, 35), "AB-BLG": (0.13, 21), "30deg-tBLG": (0.20, 26)},  # (q, Gamma cm-1)
    },
    # Novoselov et al., Science 306, 666 (2004): transport / field effect only.
    # No twisted-bilayer structural data -> not used by this simulation.
}

MATERIALS = {
    # basis = (fractional coords in (a1,a2), scattering weight)
    # weight ~ atomic number; MoS2 projected S sites coincide (top+bottom S -> 2*16)
    "graphene": dict(a=2.46, basis=[((0.0, 0.0), 6.0), ((1/3, 1/3), 6.0)]),
    "MoS2":     dict(a=PAPER["MoS2"]["a_A"], basis=[((0.0, 0.0), 42.0), ((1/3, 1/3), 32.0)]),
}


def G_first(material):
    """Magnitude of the first-order reciprocal vector (1/A, with 2*pi)."""
    a = MATERIALS[material]["a"]
    return 4 * np.pi / (np.sqrt(3) * a)


# ----------------------------------------------------------------------------
# Small physics helpers used to cross-check against the papers
# ----------------------------------------------------------------------------
def electron_wavelength(E_eV):
    return np.sqrt(H2_OVER_2M_EV_A2 / np.asarray(E_eV, float))


def spacing_from_resonance(dE_ref, dE_new, d_ref):
    """Dai et al. Eq. (1): d ~ dE^(-1/2)."""
    return d_ref * np.sqrt(dE_ref / dE_new)


def commensurate_angles(max_index=6):
    """Commensurate twist angles of a hexagonal bilayer from coprime (m,n):
    cos(th) = (m^2+4mn+n^2) / (2(m^2+mn+n^2)).  Returns sorted unique angles
    in degrees on [0,30] with the supercell size factor sqrt(m^2+mn+n^2)."""
    out = {}
    for m in range(1, max_index + 1):
        for n in range(m + 1, max_index + 2):
            if np.gcd(m, n) != 1:
                continue
            c = (m*m + 4*m*n + n*n) / (2 * (m*m + m*n + n*n))
            th = np.degrees(np.arccos(c))
            th = min(th, 60 - th)
            out[round(th, 2)] = np.sqrt(m*m + m*n + n*n)
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------
# Building the bilayer
# ----------------------------------------------------------------------------
def layer_atoms(material, theta_deg=0.0, radius=250.0, strain=(0.0, 0.0)):
    """Projected atom positions (A) and weights for one layer, rotated by
    theta about the origin. `strain` = (exx, eyy) applied before rotation."""
    m = MATERIALS[material]
    a = m["a"]
    a1 = np.array([a, 0.0])
    a2 = np.array([a / 2, a * np.sqrt(3) / 2])
    n = int(2 * radius / a) + 3
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    i, j = i.ravel(), j.ravel()
    pos, wts = [], []
    for (fx, fy), w in m["basis"]:
        p = (i + fx)[:, None] * a1 + (j + fy)[:, None] * a2
        pos.append(p)
        wts.append(np.full(len(p), w))
    pos = np.vstack(pos)
    wts = np.concatenate(wts)
    pos = pos * np.array([1 + strain[0], 1 + strain[1]])
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pos = pos @ R.T
    keep = np.hypot(pos[:, 0], pos[:, 1]) < radius
    return pos[keep], wts[keep]


def rasterize(pos, wts, N, dx):
    """Bilinear splat of point scatterers onto an N x N grid (float32)."""
    x = pos[:, 0] / dx + N / 2
    y = pos[:, 1] / dx + N / 2
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - ix, y - iy
    g = np.zeros(N * N, np.float64)
    for di, dj, w in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
        xi, yj = ix + di, iy + dj
        ok = (xi >= 0) & (xi < N) & (yj >= 0) & (yj < N)
        g += np.bincount(yj[ok] * N + xi[ok], weights=(wts * w)[ok], minlength=N * N)
    return g.reshape(N, N)


@dataclass
class Pattern:
    I: np.ndarray        # intensity, centre = specular beam at index N//2
    dq: float            # reciprocal-space pixel size (1/A)
    material: str
    theta_true: float


def simulate_pattern(material, theta_deg, L=600.0, dx=0.25, sigma_w=70.0,
                     sigma_atom=0.35, alpha=0.25, strain2=(0.0, 0.0), qmax_factor=2.3):
    """Kinematic diffraction pattern of a bilayer twisted by theta_deg.
    L: box size (A), dx: real-space pixel (A), sigma_w: Gaussian aperture (A).
    Returns a Pattern cropped to |q| < qmax_factor * G."""
    N = int(round(L / dx))
    c = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(c, c)
    W = np.exp(-(X**2 + Y**2) / (2 * sigma_w**2))
    R = 3.2 * sigma_w

    p1, w1 = layer_atoms(material, 0.0, R)
    p2, w2 = layer_atoms(material, theta_deg, R, strain=strain2)
    g1 = rasterize(p1, w1, N, dx) * W
    g2 = rasterize(p2, w2, N, dx) * W

    q1d = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    QX, QY = np.meshgrid(q1d, q1d)
    ff = np.exp(-(QX**2 + QY**2) * sigma_atom**2 / 2)   # Gaussian form factor
    F1 = np.fft.fft2(g1) * ff
    F2 = np.fft.fft2(g2) * ff
    amp = F1 + F2
    if alpha > 0:
        V1 = np.fft.ifft2(F1).real
        V2 = np.fft.ifft2(F2).real
        amp = amp + alpha * np.fft.fft2(V1 * V2) / V1.max()
    I = np.fft.fftshift(np.abs(amp) ** 2)
    dq = 2 * np.pi / (N * dx)

    half = int(qmax_factor * G_first(material) / dq)
    cN = N // 2
    I = I[cN - half:cN + half + 1, cN - half:cN + half + 1].astype(np.float64)
    return Pattern(I=I, dq=dq, material=material, theta_true=theta_deg)


# ----------------------------------------------------------------------------
# Detector / noise model
# ----------------------------------------------------------------------------
def add_noise(pat, peak_counts=500.0, background=2.0, read_sigma=1.0,
              center_shift_px=(0.0, 0.0), rng=None):
    """Return a noisy detector image.
    peak_counts : mean counts in the brightest first-order spot
    background  : mean counts/pixel of a flat background
    read_sigma  : Gaussian read noise (counts)
    center_shift_px : rigid shift of the pattern (beam tilt / misalignment)."""
    rng = np.random.default_rng() if rng is None else rng
    I = pat.I
    if any(center_shift_px):
        I = ndi.shift(I, center_shift_px, order=1, mode="constant")
    n = I.shape[0]
    h = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - h, xx - h) * pat.dq
    G = G_first(pat.material)
    ring = (r > 0.8 * G) & (r < 1.2 * G)
    scale = peak_counts / I[ring].max()
    expected = I * scale + background
    img = rng.poisson(expected).astype(float) + rng.normal(0, read_sigma, I.shape)
    return img


# ----------------------------------------------------------------------------
# Twist-angle extraction
# ----------------------------------------------------------------------------
def _circ_mean(psi):
    return np.angle(np.mean(np.exp(1j * psi)))


def _two_means_circle(psi):
    """2-means clustering of angles on a circle. Returns centres, labels."""
    d = np.abs(np.angle(np.exp(1j * (psi[:, None] - psi[None, :]))))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    centres = np.array([psi[i], psi[j]])
    for _ in range(50):
        dist = np.abs(np.angle(np.exp(1j * (psi[:, None] - centres[None, :]))))
        lab = np.argmin(dist, axis=1)
        new = np.array([_circ_mean(psi[lab == k]) if np.any(lab == k) else centres[k]
                        for k in range(2)])
        if np.allclose(new, centres):
            break
        centres = new
    return centres, lab


def find_spots(img, dq, G, ring_tol=0.10, smooth=1.5, nsig=5.0, max_spots=12):
    """Locate first-order diffraction spots. Returns (xy_rel_to_centre, centre_xy,
    smoothed image). Positions in pixels, sub-pixel by local centroid."""
    n = img.shape[0]
    sm = ndi.gaussian_filter(img, smooth)

    # centre = centroid of the specular beam near the image centre
    h = n // 2
    w = 8
    sub = sm[h - w:h + w + 1, h - w:h + w + 1]
    sy, sx = np.unravel_index(np.argmax(sub), sub.shape)
    cy0, cx0 = h - w + sy, h - w + sx
    box = img[cy0 - 3:cy0 + 4, cx0 - 3:cx0 + 4]
    box = np.clip(box - np.median(img), 0, None)
    yy, xx = np.mgrid[cy0 - 3:cy0 + 4, cx0 - 3:cx0 + 4]
    cy, cx = (yy * box).sum() / box.sum(), (xx * box).sum() / box.sum()

    Y, X = np.mgrid[:n, :n]
    r = np.hypot(Y - cy, X - cx) * dq
    ann = np.abs(r - G) < ring_tol * G
    vals = sm[ann]
    bkg = np.median(vals)
    sig = 1.4826 * np.median(np.abs(vals - bkg)) + 1e-12
    mx = ndi.maximum_filter(sm, footprint=np.ones((7, 7)))
    cand = ann & (sm == mx) & (sm > bkg + nsig * sig)
    ys, xs = np.nonzero(cand)
    if len(ys) == 0:
        return np.empty((0, 2)), (cx, cy), sm
    order = np.argsort(sm[ys, xs])[::-1][:max_spots]
    ys, xs = ys[order], xs[order]

    pts = []
    for y0, x0 in zip(ys, xs):
        wdw = 3
        sl = (slice(y0 - wdw, y0 + wdw + 1), slice(x0 - wdw, x0 + wdw + 1))
        patch = np.clip(sm[sl] - bkg, 0, None)
        if patch.sum() <= 0 or patch.shape != (2 * wdw + 1, 2 * wdw + 1):
            continue
        yy, xx = np.mgrid[sl]
        pts.append(((xx * patch).sum() / patch.sum() - cx,
                    (yy * patch).sum() / patch.sum() - cy))
    return np.array(pts), (cx, cy), sm


def extract_twist(img, dq, G, **kw):
    """Estimate the twist angle (deg, folded to [0, 30]) from a diffraction
    image of a hexagonal bilayer. Returns dict with theta, n_spots, and the
    spot positions. theta = NaN when fewer than 6 spots are found.

    Method: spots of both layers share |G|; the azimuth of each spot, taken
    modulo 60 deg, falls in two clusters separated by the twist angle."""
    pts, centre, sm = find_spots(img, dq, G, **kw)
    out = dict(theta=np.nan, n_spots=len(pts), spots=pts, centre=centre)
    if len(pts) < 6:
        return out
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    psi = 6 * phi                       # period 60 deg -> period 2*pi
    centres, lab = _two_means_circle(psi)
    dpsi = abs(np.angle(np.exp(1j * (centres[0] - centres[1]))))
    out["theta"] = np.degrees(dpsi) / 6.0
    out["labels"] = lab
    return out
