import json, subprocess, hashlib, os, re
V = lambda s: r"\vec{" + s + "}"
G = r"|\vec{G}|"
# key text in the doc -> list of segments (str = plain text, dict = math)
M = lambda tex, display=False: {"tex": tex, "display": display}
R = [
 ("a₁ = a(1, 0)", [M(r"\vec{a}_1 = a\,(1,\,0)")]),
 ("a₂ = a(1/2, √3/2)", [M(r"\vec{a}_2 = a\,(1/2,\ \sqrt{3}/2)")]),
 ("where a is the lattice constant", ["where ", M("a"), " is the lattice constant"]),
 ("r′ = R(θ)r", [M(r"\vec{r}\,' = R(\theta)\,\vec{r}")]),
 ("Ψ(k) = Σⱼ fⱼ(k) exp(−i k · rⱼ), I(k) = |Ψ(k)|²", [M(r"\Psi(\vec{k}) = \sum_j f_j(\vec{k})\, e^{-i\vec{k}\cdot\vec{r}_j}, \qquad I(\vec{k}) = |\Psi(\vec{k})|^2", True)]),
 ("fⱼ(k) is its", [M(r"f_j(\vec{k})"), " is its"]),
 ("G = m b₁ + n b₂", [M(r"\vec{G} = m\,\vec{b}_1 + n\,\vec{b}_2")]),
 ("where m and n are integers", ["where ", M("m"), " and ", M("n"), " are integers"]),
 ("|b₁| = |b₂| = 4π/(√3 a)", [M(r"|\vec{b}_1| = |\vec{b}_2| = 4\pi/(\sqrt{3}\,a)")]),
 ("at radius |G|", ["at radius ", M(G)]),
 ("top layer is rotated by θ", ["top layer is rotated by ", M(r"\theta")]),
 ("g = R(θ)G − G, |g| = 2|G| sin(θ/2)", [M(r"\vec{g} = R(\theta)\,\vec{G} - \vec{G}, \qquad |\vec{g}| = 2\,|\vec{G}|\,\sin(\theta/2)", True)]),
 ("α·F[V₁V₂]", [M(r"\alpha\,\mathcal{F}[V_1V_2]")]),
 ("where V₁ and V₂ are", ["where ", M("V_1"), " and ", M("V_2"), " are"]),
 ("F denotes", [M(r"\mathcal{F}"), " denotes"]),
 ("and α sets", ["and ", M(r"\alpha"), " sets"]),
 ("defined as “alpha.”", ["defined as ", M(r"\alpha"), "."]),
 ("other by θ.", ["other by ", M(r"\theta"), "."]),
 ("as θ doubles", ["as ", M(r"\theta"), " doubles"]),
 ("in agreement with |g| = 2|G| sin(θ/2)", ["in agreement with ", M(r"|\vec{g}| = 2\,|\vec{G}|\,\sin(\theta/2)")]),
 ("strength α was", ["strength ", M(r"\alpha"), " was"]),
 ("Even at α = 3", ["Even at ", M(r"\alpha = 3")]),
 ("(α = 0, 0.25, 1, 3)", ["(", M(r"\alpha = 0,\ 0.25,\ 1,\ 3"), ")"]),
 ("2|G| sin 15° = 0.518|G|", [M(r"2\,|\vec{G}|\sin 15^\circ = 0.518\,|\vec{G}|")]),
 ("(√3 − 1)|G| = 0.732|G|", [M(r"(\sqrt{3}-1)\,|\vec{G}| = 0.732\,|\vec{G}|")]),
 ("0.520 ± 0.006|G|", [M(r"(0.520 \pm 0.006)\,|\vec{G}|")]),
 ("0.728 ± 0.003|G|", [M(r"(0.728 \pm 0.003)\,|\vec{G}|")]),
 ("(α = 1,", ["(", M(r"\alpha = 1"), ","]),
 ("at |G|, 0.732|G| and 0.518|G|", ["at ", M(G), ", ", M(r"0.732\,|\vec{G}|"), " and ", M(r"0.518\,|\vec{G}|")]),
 ("1 × |G|", [M(r"1 \times |\vec{G}|")]),
 ("2|G| sin 15° = 0.518 |G|", [M(r"2\,|\vec{G}|\sin 15^\circ = 0.518\,|\vec{G}|")]),
 ("0.520 ± 0.006 |G|", [M(r"(0.520 \pm 0.006)\,|\vec{G}|")]),
 ("(√3 − 1)|G| = 0.732 |G|", [M(r"(\sqrt{3}-1)\,|\vec{G}| = 0.732\,|\vec{G}|")]),
 ("0.728 ± 0.003 |G|", [M(r"(0.728 \pm 0.003)\,|\vec{G}|")]),
 ("(|g| = 2|G| sin(θ/2))", ["(", M(r"|\vec{g}| = 2\,|\vec{G}|\,\sin(\theta/2)"), ")"]),
 ("2·G·sin(θ/2) apart, where G is", [M(r"2\,|\vec{G}|\,\sin(\theta/2)"), " apart, where ", M(r"|\vec{G}|"), " is"]),
 ("brightness α is", ["brightness ", M(r"\alpha"), " is"]),
]
PRE = r"""\documentclass[12pt,border=0.4pt]{standalone}
\usepackage{amsmath,amssymb}\usepackage{mathptmx}
\begin{document}%s\end{document}"""
out, done = [], {}
os.makedirs("png", exist_ok=True)
for key, segs in R:
    js = []
    for s in segs:
        if isinstance(s, str): js.append(s); continue
        tex = (r"$\displaystyle " if s["display"] else "$") + s["tex"] + "$"
        h = hashlib.md5(tex.encode()).hexdigest()[:10]
        if h not in done:
            open(f"{h}.tex", "w").write(PRE % tex)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{h}.tex"], capture_output=True, check=True)
            info = subprocess.run(["pdfinfo", f"{h}.pdf"], capture_output=True, text=True).stdout
            w_pt, h_pt = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
            subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile", f"{h}.pdf", f"png/{h}"], check=True)
            scale = 96 / 72 * (1.15 if s["display"] else 1.0)
            done[h] = dict(file=f"{h}.png", w=round(w_pt * scale, 1), h=round(h_pt * scale, 1), tex=s["tex"])
        js.append(dict(img=h, **{k: done[h][k] for k in ("w", "h")}))
    out.append(dict(key=key, segs=js))
json.dump(out, open("math_map.json", "w"), ensure_ascii=False)
print(len(done), "images;", sum(1 for _ in R), "replacements")
for h, d in done.items(): print(h, d["w"], d["h"], d["tex"])
