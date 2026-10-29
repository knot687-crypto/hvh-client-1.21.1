"""Derivations and checks behind every constant and claim in right_stick.gpc.

    python3 derive.py            verify everything (about a minute)
    python3 derive.py --search   also re-run the mask annealing (slow)

Sections
  1  Poisson-disk seed (r = 4 on the 8x8 torus)
  2  8x8 blue-noise mask: 2-D spectrum, progressive spacing
  3  voxel rays: golden DDA, period, Sturmian steps, 1-D spectra
  4  Bellman equation: minimum-variance gain  (+ Clairaut, curse of dimensionality)
  5  HJB: continuous-time counterpart
  6  Smith predictor == error feedback; Bode integral; PSD of the output error
  7  Cramer-Rao / Fisher information of the whole-percent channel
  8  ADMM: l1-constrained noise-shaping design; rejected window estimator
  9  cross ratio timing test
 10  Kolmogorov randomness: compressibility of the threshold sequences
 11  sensor fusion accuracy (simulated Zen)
 12  fixed-point ranges

Needs numpy.  Pure-integer parts are checked against tools/reference.py,
the same model the GPC test-suite compares the script with bit by bit.
"""
import bz2
import itertools
import lzma
import math
import os
import random
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gpcsim import Parser  # noqa: E402
from zen_model import BUILTIN_CONSTS  # noqa: E402
import reference as ref  # noqa: E402

GPC = os.path.join(os.path.dirname(__file__), "..", "right_stick.gpc")
P = Parser(open(GPC).read(), BUILTIN_CONSTS).parse()
MASK = np.array(P.consts["RS_MASK"])
D = P.defines
N8 = 8
PHI = (1 + 5 ** 0.5) / 2
SEED = [(0, 0), (4, 0), (2, 4), (6, 4)]          # (x, y)


def head(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def tdist2(a, b):
    dx = abs(a[0] - b[0]); dx = min(dx, N8 - dx)
    dy = abs(a[1] - b[1]); dy = min(dy, N8 - dy)
    return dx * dx + dy * dy


def hand_motion(n, seed):
    """1-D right-stick trajectory in percent at 10 ms: minimum-jerk flicks (Flash &
    Hogan), holds and pursuit with 8-12 Hz physiological tremor, spring releases."""
    rng = np.random.default_rng(seed)
    T = 0.01
    x = []
    pos = 0.0

    def trem(m):
        t = np.arange(m) * T
        a = rng.uniform(0.05, 0.35)
        f1, f2 = rng.uniform(8, 12, 2)
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        return a * (0.7 * np.sin(2 * np.pi * f1 * t + p1) + 0.3 * np.sin(2 * np.pi * f2 * t + p2))

    while len(x) < n:
        kind = rng.choice(4, p=[0.35, 0.25, 0.25, 0.15])
        if kind == 0:                                   # flick
            target = float(np.clip(rng.normal(0, 55), -100, 100))
            m = max(2, int(rng.uniform(0.08, 0.40) / T))
            s = np.arange(1, m + 1) / m
            seg = pos + (target - pos) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5) + 0.5 * trem(m)
            pos = target
        elif kind == 1:                                 # hold
            m = int(rng.uniform(0.05, 0.8) / T)
            seg = pos + trem(m)
        elif kind == 2:                                 # pursuit
            m = int(rng.uniform(0.3, 2.0) / T)
            v, p, sv = 0.0, pos, rng.uniform(5, 40)
            seg = np.zeros(m)
            for i in range(m):
                v += -v / 0.2 * T + sv * np.sqrt(2 * T / 0.2) * rng.normal()
                p = float(np.clip(p + v * T, -100, 100))
                seg[i] = p
            seg = seg + trem(m)
            pos = p
        else:                                           # release, then rest
            m = int(rng.uniform(0.1, 1.0) / T)
            seg = np.r_[pos * np.exp(-np.arange(1, 5) * T / 0.012), np.zeros(m)]
            pos = 0.0
        x.extend(np.clip(seg, -100, 100).tolist())
    return np.array(x[:n])


CELLS = [(x, y) for y in range(N8) for x in range(N8)]
D2 = np.array([[tdist2(a, b) for b in CELLS] for a in CELLS], dtype=float)


# ============================================================== 1 Poisson-disk
def gaussian_energy(pts, sigma=1.5):
    e = 0.0
    for a, b in itertools.combinations(pts, 2):
        e += 2 * math.exp(-tdist2(a, b) / (2 * sigma * sigma))
    return e


def poisson_disk():
    head("1  Poisson-disk sampling, radius 4, 8x8 torus")
    r2 = 16
    best = []

    def grow(cur, start):
        nonlocal best
        if len(cur) > len(best[0]) if best else True:
            best = [list(cur)]
        elif len(cur) == len(best[0]):
            best.append(list(cur))
        for i in range(start, 64):
            c = CELLS[i]
            if all(tdist2(c, p) >= r2 for p in cur):
                grow(cur + [c], i + 1)

    grow([(0, 0)], 1)
    size = len(best[0])
    bound = math.floor(math.pi / (2 * math.sqrt(3)) * 64 / (math.pi * 2 ** 2))
    print(f"  maximal sets containing (0,0): size {size}; hexagonal packing bound "
          f"floor(0.9069 * 64 / (pi 2^2)) = {bound}")
    assert size == bound == 4
    uniq = sorted({tuple(sorted(s)) for s in best}, key=lambda s: (round(gaussian_energy(s), 9),
                                                                   sorted(y * 8 + x for x, y in s)))
    for s in uniq[:3]:
        md = min(math.sqrt(tdist2(a, b)) for a, b in itertools.combinations(s, 2))
        print(f"    {s}  Gaussian energy {gaussian_energy(s):.6f}  min distance {md:.3f}")
    assert tuple(sorted(SEED)) == uniq[0], "seed must be the minimum-energy maximal set"
    print(f"  chosen seed (ties broken by cell index): {SEED}")
    ranks = [int(MASK[y, x]) for x, y in SEED]
    print(f"  mask ranks at the seed: {ranks}")
    assert ranks == [0, 1, 2, 3]


# ============================================================== 2 blue-noise mask
def gf_energy(flat, sig_i=2.1, sig_s=1.0):
    spat = np.exp(-D2 / sig_i ** 2)
    np.fill_diagonal(spat, 0.0)
    v = flat / 63.0
    dv = np.sqrt(np.abs(v[:, None] - v[None, :]))
    return float((spat * np.exp(-dv / sig_s ** 2)).sum())


def radial_psd(M):
    fr = np.fft.fftfreq(N8) * N8
    FX, FY = np.meshgrid(fr, fr)
    RAD = np.sqrt(FX ** 2 + FY ** 2)
    acc = np.zeros((N8, N8))
    for k in range(1, 64):
        B = (M < k).astype(float)
        B -= B.mean()
        acc += np.abs(np.fft.fft2(B)) ** 2
    acc /= 63
    return {round(float(r), 3): float(acc[np.abs(RAD - r) < 1e-6].mean()) for r in np.unique(RAD) if r > 0}


def mask_checks():
    head("2  8x8 blue-noise mask")
    flat = MASK.flatten()
    assert sorted(flat.tolist()) == list(range(64))
    for k, r2 in ((4, 16), (8, 4), (16, 2)):
        idx = np.where(flat < k)[0]
        sub = D2[np.ix_(idx, idx)] + np.eye(k) * 99
        print(f"  ranks 0..{k - 1:2d}: min toroidal distance {math.sqrt(sub.min()):.3f} (required {math.sqrt(r2):.3f})")
        assert sub.min() >= r2
    rp = radial_psd(MASK)
    rng = np.random.default_rng(0)
    white = [radial_psd(rng.permutation(64).reshape(8, 8)) for _ in range(300)]
    wm = {r: float(np.mean([w[r] for w in white])) for r in rp}
    print("  radially averaged power of the 63 threshold patterns (white = random permutation):")
    for r in sorted(rp):
        print(f"    |f| = {r:5.3f}   mask {rp[r]:7.2f}   white {wm[r]:6.2f}   ratio {rp[r] / wm[r]:5.2f}")
    low = [rp[r] / wm[r] for r in rp if r < 2]
    assert max(low) < 0.35, "lowest spatial frequencies must be suppressed (blue noise)"
    ge = gf_energy(flat)
    gw = np.mean([gf_energy(rng.permutation(64)) for _ in range(200)])
    print(f"  Georgiev-Fajardo energy: mask {ge:.2f}, random permutations {gw:.2f}")


# ============================================================== 3 voxel rays
def ray_path(n, cx, cy, d, second):
    dtx, dty = D["RS_DTX"], D["RS_DTY"]
    out = np.empty(n, dtype=int)
    steps = []
    for k in range(n):
        if d < 0:
            if second:
                cy = (cy + 1) % 8
            else:
                cx = (cx + 1) % 8
            d += dtx
            steps.append(0)
        else:
            if second:
                cx = (cx + 7) % 8
            else:
                cy = (cy + 1) % 8
            d -= dty
            steps.append(1)
        out[k] = cy * 8 + cx
    return out, steps


def lowband(seq, seg=256, fmax=1 / 16):
    s = (seq - seq.mean()) / seq.std()
    win = np.hanning(seg)
    X = np.fft.rfft(s[: len(s) // seg * seg].reshape(-1, seg) * win, axis=1)
    Pw = (np.abs(X) ** 2).mean(axis=0) / (win ** 2).sum()
    f = np.fft.rfftfreq(seg)
    return float(Pw[f < fmax].mean()), Pw, f


def xcorr_max(a, b, L=32):
    a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    n = len(a)
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * n)) * np.fft.rfft(b, 2 * n), 2 * n) / n
    return float(np.abs(np.r_[cc[:L + 1], cc[-L:]]).max())


def voxel_rays():
    head("3  voxel rays (Amanatides-Woo DDA, slope 1/phi) through the mask")
    dtx, dty = D["RS_DTX"], D["RS_DTY"]
    print(f"  tDeltaY/tDeltaX = {dty}/{dtx} = {dty / dtx:.8f}   phi = {PHI:.8f}   "
          f"(phi * 2^16 = {PHI * 65536:.3f}, rounded)")
    assert dty == round(PHI * 65536)
    g = math.gcd(dtx, dtx + dty)
    period_d = (dtx + dty) // g             # d is a rotation on Z/(dtx + dty)
    nx, ny = dty // g, dtx // g             # x- and y-steps per d-period
    order = math.lcm(8 // math.gcd(8, nx % 8), 8 // math.gcd(8, ny % 8))
    full = period_d * order
    print(f"  d-state period {period_d} steps ({nx} x-steps, {ny} y-steps); the cell shifts by "
          f"({nx % 8}, {ny % 8}) mod 8 per period,")
    print(f"  so the full (cell, d) state repeats after {full} steps = {full * 0.01 / 3600:.2f} h at 10 ms;"
          " each stroke restarts it anyway")
    # Sturmian check: a cutting sequence of a line is balanced - every window of
    # length L holds floor(L a) or ceil(L a) y-steps (a = 1/phi^2 here)
    _, st = ray_path(200000, 0, 0, -1, False)
    st = np.array(st)
    cs = np.r_[0, np.cumsum(st)]
    worst = 0
    for L in range(1, 400):
        counts = cs[L:] - cs[:-L]
        worst = max(worst, int(counts.max() - counts.min()))
    frac_y = st.mean()
    print(f"  fraction of y-steps {frac_y:.6f} vs 1/phi^2 = {1 / PHI ** 2:.6f}; every window of length")
    print(f"  1..399 differs by at most {worst} y-step between positions: a balanced (Sturmian)")
    print("  word, i.e. the golden cutting sequence")
    assert worst == 1
    # equidistribution over one d-period
    cells, _ = ray_path(period_d, 0, 0, -1, False)
    counts = np.bincount(cells, minlength=64)
    print(f"  cell visits over one d-period: min {counts.min()}, max {counts.max()} (ideal {period_d / 64:.1f})")
    # 1-D spectra and cross-correlation, random starts as used per stroke
    rnd = random.Random(2026)
    lbx, lby, xc = [], [], []
    flat = MASK.flatten().astype(float)
    wl = []
    for _ in range(16):
        px, _ = ray_path(4096, rnd.randrange(8), rnd.randrange(8), rnd.randrange(-dty, dtx), False)
        py, _ = ray_path(4096, rnd.randrange(8), rnd.randrange(8), rnd.randrange(-dty, dtx), True)
        lbx.append(lowband(flat[px])[0]); lby.append(lowband(flat[py])[0])
        xc.append(xcorr_max(flat[px], flat[py]))
        wl.append(lowband(np.array([rnd.random() for _ in range(4096)]))[0])
    print(f"  low-band (f < 1/16) power of the thresholds, white noise = 1:")
    print(f"    X ray {np.mean(lbx):.4f}   Y ray {np.mean(lby):.4f}   iid sequence {np.mean(wl):.4f}")
    print(f"  max |cross-correlation| X vs Y, lags +-32: {np.mean(xc):.4f} "
          f"(independent sequences give about 2.9/sqrt(4096) = {2.9 / 64:.4f})")
    assert np.mean(lbx) < 0.1 and np.mean(lby) < 0.1 and np.mean(xc) < 3 / 64
    taus = sorted(ref.threshold(r) for r in range(64))
    print(f"  thresholds (100 r + 82) / 64: {taus[0]} .. {taus[-1]}, 64 distinct values")


# ============================================================== 4 Bellman (+ Clairaut, curse)
def bellman():
    head("4  Bellman equation: minimum-variance error feedback")
    print("  state  theta = integrated output error before a sample; control c added to the target")
    print("  theta' = theta + c + q   (q: quantisation error of this sample, zero mean, var s2)")
    print("  V(theta) + J = min_c E[ theta'^2 + mu c^2 + V(theta') ]      (average cost)")
    for mu in (0.0, 0.1, 1.0):
        p = 0.0
        for _ in range(2000):                  # relative value iteration on V = p theta^2
            s = 1 + p
            p = s * mu / (s + mu) if mu > 0 else 0.0
        k = (1 + p) / (1 + p + mu)
        closed = (math.sqrt(1 + 4 * mu) - 1) / 2
        print(f"    mu = {mu:3.1f}: p = {p:.6f} (closed form {closed:.6f}), gain k = {k:.6f}")
        assert abs(p - closed) < 1e-9
    print("  mu = 0 (only the camera-angle error counts): p = 0, k = 1 -> c = -theta (deadbeat).")
    print("  The script's w = v - smith_m is exactly c = -theta.")

    print("\n  Clairaut / curse of dimensionality: both axes together")
    print("  V(theta) = theta' P theta over theta in R^2, cost |theta'|^2 + mu |c|^2, P full 2x2")
    mu = 0.3
    Pm = np.array([[0.7, 0.4], [0.1, 0.2]])   # deliberately asymmetric, coupled start
    for _ in range(500):
        S = np.eye(2) + Pm
        S = 0.5 * (S + S.T)                     # the quadratic form only sees the symmetric part
        K = np.linalg.solve(S + mu * np.eye(2), S)
        A = np.eye(2) - K
        Pm = A.T @ S @ A + mu * K.T @ K
    print(f"    fixed point P = [[{Pm[0, 0]:.6f}, {Pm[0, 1]:.2e}], [{Pm[1, 0]:.2e}, {Pm[1, 1]:.6f}]]")
    print("    symmetric (Hessian of V: mixed partials commute, Clairaut/Schwarz) and the cross")
    print("    term vanishes: the optimal policy factorises per axis, so the script keeps one")
    print("    scalar error state per axis instead of a joint table.")
    assert abs(Pm[0, 1]) < 1e-12 and abs(Pm[1, 0]) < 1e-12
    grid = 199 ** 2 * 8 ** 4 * (D["RS_DTX"] + D["RS_DTY"]) ** 2
    print(f"    a tabulated DP over the joint state (two error states x two ray states) would need")
    print(f"    199^2 x 8^4 x 171575^2 = {grid:.2e} cells: the curse the factorisation avoids.")


# ============================================================== 5 HJB
def hjb():
    head("5  HJB: continuous-time counterpart")
    print("  d theta = c dt + s dW,  cost E int (theta^2 + mu c^2) dt")
    print("  J = min_c [theta^2 + mu c^2 + V'(theta) c] + s2/2 V''(theta),  V = p theta^2")
    print("  -> c* = -p theta / mu,  p = sqrt(mu),  J = sqrt(mu) s2")
    for mu in (1.0, 0.01, 1e-4, 1e-6):
        p = math.sqrt(mu)
        print(f"    mu = {mu:7.0e}: feedback gain p/mu = {p / mu:10.1f}, average cost/s2 = {p:.4f}")
    print("  With no delay the gain grows without bound and the integrated error -> 0.")
    print("  Sampling plus the one-cycle delay caps it: the Bellman optimum (section 4) is gain 1")
    print("  with residual cost s2, i.e. the error of a single sample.  That residual is what the")
    print("  Bode integral below says cannot be removed, only moved in frequency.")


# ============================================================== 6 Smith / Bode / PSD
def smith_bode_psd():
    head("6  Smith predictor = error feedback;  Bode integral;  PSD")
    rnd = random.Random(5)
    flat = MASK.flatten()
    # Smith-predictor form vs direct error-feedback recursion
    for trial in range(50):
        m_s = 0          # smith mismatch
        e_prev = 0       # error feedback state
        v0 = rnd.randint(-9900, 9900)
        for n in range(400):
            v = v0 + rnd.randint(-250, 250)
            tau = ref.threshold(int(flat[rnd.randrange(64)]))
            fl = v // 100
            # Smith: model output w is transmitted ideally; plant output y measured next cycle
            w = v - m_s
            y1 = fl if v % 100 == 0 else (fl + 1 if w - fl * 100 >= tau else fl)
            if v % 100:
                m_s = y1 * 100 - w
            # error feedback: subtract the previous quantisation error
            u = v - e_prev
            y2 = fl if v % 100 == 0 else (fl + 1 if u - fl * 100 >= tau else fl)
            if v % 100:
                e_prev = y2 * 100 - u
            assert y1 == y2
    print("  Smith form (model mismatch fed back through z^-1) and error-feedback form give")
    print("  identical outputs on 20000 random samples.  Linearised loop: y = v + (1 - z^-1) e.")

    w = np.linspace(1e-7, math.pi, 2_000_001)
    lnS = np.log(np.abs(1 - np.exp(-1j * w)))
    integral = np.trapezoid(lnS, w)
    print(f"  Bode sensitivity integral  int_0^pi ln|1 - e^-jw| dw = {integral:.2e}  (exact value 0)")
    assert abs(integral) < 1e-4
    print("  |S| < 1 for w < pi/3 (below fs/6), |S| > 1 above, max |S| = 2 at w = pi: the area of")
    print("  log-attenuation equals the area of log-amplification (waterbed effect).")

    # measured PSD of the output error of the real quantiser on a slow input
    model = ref.Reference(MASK.tolist())
    model.m = [0, 0]
    n = 1 << 16
    errs = []
    for k in range(n):
        v = int(4000 + 2500 * math.sin(2 * math.pi * k / 7919.0) + 37 * math.sin(k / 13.0))
        model.advance()
        y = model.quantise(0, v, model.tau[0])
        errs.append(y * 100 - v)
    e = np.array(errs, float) / 100
    seg = 1024
    win = np.hanning(seg)
    X = np.fft.rfft(e[: n // seg * seg].reshape(-1, seg) * win, axis=1)
    Pw = (np.abs(X) ** 2).mean(axis=0) / (win ** 2).sum()
    f = np.fft.rfftfreq(seg)
    sel = (f > 2 / seg) & (f < 0.02)
    slope = np.polyfit(np.log10(f[sel]), 10 * np.log10(Pw[sel]), 1)[0]
    print(f"  measured output-error PSD: low-frequency slope {slope:.1f} dB/decade (first-order")
    print("  shaping of an exactly white error gives +20; here the error is mildly signal dependent)")
    print(f"    power at f = 1/{seg}: {Pw[1]:.2e}   at f = 1/4: {Pw[seg // 4]:.2e}   at f = 1/2: {Pw[-1]:.2e}")
    print(f"  integrated error |sum(y - v)| over {n} samples never exceeded "
          f"{np.abs(np.cumsum(e)).max():.2f} percent-samples (bound: < 1)")
    assert 14 < slope < 26 and np.abs(np.cumsum(e)).max() < 1


# ============================================================== 7 Cramer-Rao
def fisher_quantised(x, sigma):
    """Fisher information about x from z = round(x + n), n ~ N(0, sigma^2)"""
    if sigma == 0:
        return 0.0
    from math import erf, exp, pi, sqrt
    I = 0.0
    for k in range(int(math.floor(x)) - 8, int(math.floor(x)) + 9):
        a, b = (k - 0.5 - x) / sigma, (k + 0.5 - x) / sigma
        Pk = 0.5 * (erf(b / sqrt(2)) - erf(a / sqrt(2)))
        dP = -(exp(-b * b / 2) - exp(-a * a / 2)) / (sigma * sqrt(2 * pi))
        if Pk > 1e-300:
            I += dP * dP / Pk
    return I


def cramer_rao():
    head("7  Cramer-Rao: what a whole-percent reading can tell about the true position")
    print("  z = round(x + n), n ~ N(0, s^2) (s: hand tremor / sensor noise, in percent).")
    print("  The 0.01 % channel measures x + n directly: Fisher information 1/s^2.")
    print("  efficiency = I_quantised / I_fine, averaged over the cell; CRB(N) = 1/(N I).")
    for s in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        xs = np.linspace(0.0, 1.0, 101)
        I = np.mean([fisher_quantised(12 + x, s) for x in xs])
        eff = I * s * s if s > 0 else 0.0
        crb = (1 / math.sqrt(100 * I)) if I > 0 else float("inf")
        print(f"    s = {s:4.2f} %: I = {I:10.4f}/%^2   efficiency {eff:5.3f}   "
              f"CRB std after 100 samples {crb:7.4f} %")
    print("  s = 0 (a still thumb): the likelihood is flat on the cell, I = 0, no unbiased")
    print("  estimator exists below 1 %.  This is why the script fuses the polar channel instead")
    print("  of estimating sub-percent values from the integer channel over time.")
    print("  The quantisation errors are bounded, which breaks the CRB regularity conditions;")
    print("  the script exploits that directly: a single sample can refute a rounding")
    print("  hypothesis outright (section 11), with no statistics to tune.")


# ============================================================== 8 ADMM
def project_l1(v, r=1.0):
    if np.abs(v).sum() <= r:
        return v
    u = np.sort(np.abs(v))[::-1]
    css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(u) + 1) > (css - r))[0][-1]
    theta = (css[k] - r) / (k + 1.0)
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0)


def admm_ntf(K=4, rho=1.0, iters=3000):
    """min sum_j g_j^2, g_j = 1 + h_1 + .. + h_j (j < K)  [camera-angle error variance / s2]
       s.t. 1 + sum h = 0 (no DC error) and ||h||_1 <= 1 (no overload)"""
    L = np.tril(np.ones((K - 1, K)))[:, :]            # g_j for j = 1..K-1 uses h_1..h_j
    L = np.array([[1.0 if i <= j else 0.0 for i in range(K)] for j in range(K - 1)])
    H = 2 * (L.T @ L)                                  # objective 1 + |1 + L h|^2 -> Hessian
    lin = 2 * (L.T @ np.ones(K - 1))
    lam, V = np.linalg.eigh(H)                         # symmetric (Clairaut): one eigensolve
    a = np.ones(K)                                     # equality a'h = -1
    z = np.zeros(K); u = np.zeros(K)
    for _ in range(iters):
        # x-update: min 1/2 h'Hh + lin'h + rho/2|h - z + u|^2  s.t. a'h = -1 (KKT)
        Minv = V @ np.diag(1 / (lam + rho)) @ V.T
        q = -lin + rho * (z - u)
        hx = Minv @ q
        nu = (a @ hx + 1) / (a @ Minv @ a)
        h = hx - nu * (Minv @ a)
        z = project_l1(h + u)
        u = u + h - z
    return z


def admm_section():
    head("8  ADMM")
    h = admm_ntf()
    print(f"  (a) noise-shaping design QP, order 4: h* = {np.round(h, 6)}")
    print("      NTF(z) = 1 + sum h_k z^-k = 1 - z^-1: first-order error feedback is optimal for")
    print("      the integrated (camera-angle) error, matching the Bellman result.")
    assert np.allclose(h, [-1, 0, 0, 0], atol=1e-4)

    print("  (b) rejected design: sliding-window MAP estimate of the true stick from the")
    print("      whole-percent channel alone (min sum of squared second differences, subject to")
    print("      every sample lying in its cell), solved by ADMM, 50 iterations per sample.")
    W, eps, rh = 8, 1 / 16, 1.0
    Dm = np.diff(np.eye(W), 2, axis=0)
    Minv = np.linalg.inv(Dm.T @ Dm + (eps + rh) * np.eye(W))

    def admm_map(zq):
        z = np.zeros(W); c = np.zeros(W); u = np.zeros(W)
        est = []
        for zz in zq:
            z = np.r_[z[1:], zz]
            lo, hi = z - 0.5, z + 0.5
            c = np.clip(np.r_[c[1:], 2 * c[-1] - c[-2]], lo, hi)
            u = np.r_[u[1:], 0.0]
            for _ in range(50):
                xx = Minv @ (eps * z + rh * (c - u))
                c = np.clip(xx + u, lo, hi)
                u = u + xx - c
            est.append(c[-1])
        return np.array(est)

    def windowed(e, w):
        c = np.cumsum(np.r_[0.0, e])
        m = (c[w:] - c[:-w]) / w
        return float(np.sqrt((m ** 2).mean()))

    cases = [("smooth ramp, no tremor", 10 + 0.3 * (np.arange(6000) % 200)),
             ("hand model: flicks, holds, pursuit, 8-12 Hz tremor", hand_motion(6000, 4))]
    print(f"      {'':52s} {'RMSE':>14s} {'1 s window':>16s}")
    for name, x in cases:
        zq = np.round(x)
        est = admm_map(zq)
        r = lambda e: float(np.sqrt((e[50:] ** 2).mean()))
        print(f"      {name:52s} raw {r(zq - x):.3f} ADMM {r(est - x):.3f}   "
              f"raw {windowed(zq - x, 100):.4f} ADMM {windowed(est - x, 100):.4f}")
    print("      Lower per-sample error on a clean ramp, worse on realistic motion by both measures:")
    print("      tremor already dithers the integer reading and the smoothness prior removes real")
    print("      motion.  Not in the script.")


# ============================================================== 9 cross ratio
def cross_ratio_section():
    head("9  cross ratio timing test")
    cr = lambda t: ((t[2] - t[0]) * (t[3] - t[1])) / ((t[2] - t[1]) * (t[3] - t[0]))
    for period in (1, 5, 10, 16.67, 33.3):
        t = [0, period, 2 * period, 3 * period]
        assert abs(cr(t) - 4 / 3) < 1e-12
    mob = lambda x: (2 * x + 1) / (0.01 * x + 3)
    t = [0, 10, 20, 30]
    print(f"  CR of evenly spaced samples = 4/3 for every period; under a projective map of time")
    print(f"  it stays {cr([mob(v) for v in t]):.6f}.  get_rtime() is whole ms, so a triple of intervals")
    print("  passes when CR = 4/3 is reachable with each interval off by < 1 ms:")
    for trip in ((10, 10, 10), (5, 5, 5), (10, 11, 10), (10, 12, 10), (10, 13, 10), (10, 20, 10), (16, 17, 16)):
        print(f"    intervals {trip}: {'uniform' if ref.uniform_timing(*trip) else 'NOT uniform'}")


# ============================================================== 10 Kolmogorov
def kolmogorov():
    head("10 Kolmogorov randomness")
    n = 1 << 16
    flat = MASK.flatten()
    dtx, dty = D["RS_DTX"], D["RS_DTY"]
    det, _ = ray_path(n, 0, 0, -1, False)
    seq_det = bytes(int(flat[c]) for c in det)
    rnd = random.Random(7)
    parts = []
    restarts = 0
    while sum(len(p) for p in parts) < n:
        cells, _ = ray_path(rnd.randint(20, 120), rnd.randrange(8), rnd.randrange(8), rnd.randrange(-dty, dtx), False)
        parts.append(bytes(int(flat[c]) for c in cells))
        restarts += 1
    seq_rs = b"".join(parts)[:n]
    seq_iid = bytes(rnd.randrange(64) for _ in range(n))
    print("  compressed size (bytes) of 65536 thresholds; smaller = more compressible:")
    print(f"    {'':38s} {'zlib':>8s} {'bz2':>8s} {'lzma':>8s}")
    for name, s in (("one golden ray, no restarts", seq_det), ("restarted per stroke (script)", seq_rs),
                    ("i.i.d. uniform ranks", seq_iid)):
        sizes = [len(zlib.compress(s, 9)), len(bz2.compress(s, 9)), len(lzma.compress(s))]
        print(f"    {name:38s} " + " ".join(f"{v:8d}" for v in sizes))
    bits = restarts * math.log2(64 * (dtx + dty))
    print(f"  information injected by {restarts} restarts at log2(64 x {dtx + dty}) = "
          f"{math.log2(64 * (dtx + dty)):.1f} bits each: {bits / 8:.0f} bytes")
    print("  Kolmogorov complexity is uncomputable; compressed size is only an upper bound.")
    print("  A deterministic program cannot raise it beyond its own length plus its input, so the")
    print("  script's sequence is exactly as random as the entropy it harvests from the stick")
    print("  readings.  The ray restarts break the long-range regularity that the compressors")
    print("  exploit in the single deterministic ray.")


# ============================================================== 11 fusion accuracy
def fusion_accuracy():
    head("11 sensor fusion on a simulated Zen (assumed conventions; tests cover the others)")
    from zen_model import ZenModel
    rnd = random.Random(21)
    out = {"integer": [], "fused": []}
    for trial in range(3):
        model = ref.Reference(MASK.tolist())
        dev = ZenModel()
        x = y = 0.0
        xs = []
        k = 0
        while len(xs) < 6000:
            tx, ty = rnd.gauss(0, 55), rnd.gauss(0, 22)
            steps = rnd.randint(10, 80)
            x0, y0 = x, y
            a = rnd.uniform(0.05, 0.35)
            for s in range(1, steps + 1):
                uu = s / steps
                hh = 10 * uu ** 3 - 15 * uu ** 4 + 6 * uu ** 5
                k += 1
                x = max(-100, min(100, x0 + (tx - x0) * hh + a * math.sin(0.63 * k)))
                y = max(-100, min(100, y0 + (ty - y0) * hh + a * math.sin(0.59 * k)))
                xs.append((x, y))
        errs_i, errs_f = [], []
        for (x, y) in xs:
            dev.set_stick(x, y)
            zx, zy = dev.ival("x"), dev.ival("y")
            ox, oy = model.step(zx, zy, dev.polar(0), dev.polar(1), 10)
            errs_i.append((zx - x, zy - y))
            errs_f.append((ox - x, oy - y))
        out["integer"].append(np.array(errs_i))
        out["fused"].append(np.array(errs_f))

    def wr(e, w):
        c = np.cumsum(np.r_[0.0, e])
        m = (c[w:] - c[:-w]) / w
        return np.sqrt((m ** 2).mean())

    print("  RMS of the windowed-mean error (camera-velocity error) in percent:")
    print(f"    {'':10s} {'X 250ms':>9s} {'X 1s':>8s} {'Y 250ms':>9s} {'Y 1s':>8s}")
    for k, arrs in out.items():
        vals = np.mean([[wr(a[:, 0], 25), wr(a[:, 0], 100), wr(a[:, 1], 25), wr(a[:, 1], 100)] for a in arrs], axis=0)
        print(f"    {k:10s} " + " ".join(f"{v:9.4f}" for v in vals))


# ============================================================== 12 fixed point
def fixed_point():
    head("12 fixed-point ranges (all must stay inside +-2^31)")
    lim = 2 ** 31 - 1
    items = {
        "pr * 10000 + 16383": 32767 * 10000 + 16383,
        "f_rho * cos (Q14)": 10000 * 16384,
        "slab bound * 16384 (worst, wrong conventions)": (100 * 99 + 100 + 4 + 10000) * 16384,
        "f_t * tangent": (10000 * 286 // 16384 + 5) * 16384,
        "Park-Miller A * lo": 16807 * 127772,
        "Park-Miller R * hi": 2836 * (2 ** 31 // 127773),
        "harvest key pr*361+pa": 32767 * 361 + 359,
        "harvest key 2": 201 * 203 + 201 + 1000 * 41209,
        "cross-ratio F terms (intervals clamped to 1000 ms)": 3 * 2002 * 2002,
        "ray_d range": max(D["RS_DTX"], D["RS_DTY"]),
    }
    for k, v in items.items():
        print(f"  {k:48s} {v:>14,d}  ({v / lim * 100:5.2f} % of 2^31)")
        assert v < lim


# ============================================================== mask search
class SplitMix64:
    """portable PRNG for the annealer (independent of numpy versions)"""
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def unit(self):
        return (self.next() >> 11) / float(1 << 53)


SEARCH_LAMBDA, SEARCH_MU, SEARCH_ITERS = 400.0, 400.0, 60000
SEARCH_SEEDS = range(1, 17)


class MaskSearch:
    """Annealing objective  GF(M) + lambda (LB_x + LB_y) + mu XC
       GF  Georgiev-Fajardo blue-noise energy (sigma_i 2.1, sigma_s 1)
       LB  power below f = 1/16 of the thresholds read along each ray
       XC  RMS cross-correlation (lags +-16) between the two rays
    subject to ranks 0..3 = Poisson seed (r = 4), ranks 0..7 >= 2 apart,
    ranks 0..15 >= sqrt 2 apart (progressive Poisson-disk spacing)."""

    def __init__(self):
        dtx, dty = D["RS_DTX"], D["RS_DTY"]

        def px(n, cx, cy, d):
            out = []
            for _ in range(n):
                if d < 0:
                    cx = (cx + 1) & 7; d += dtx
                else:
                    cy = (cy + 1) & 7; d -= dty
                out.append(cy * 8 + cx)
            return np.array(out)

        def py(n, cx, cy, d):
            out = []
            for _ in range(n):
                if d < 0:
                    cy = (cy + 1) & 7; d += dtx
                else:
                    cx = (cx - 1) & 7; d -= dty
                out.append(cy * 8 + cx)
            return np.array(out)

        self.PX = [px(4096, 0, 0, -1000), px(4096, 5, 2, 70000)]
        self.PY = [py(4096, 3, 5, 40000), py(4096, 6, 1, -50000)]
        spat = np.exp(-D2 / 2.1 ** 2)
        np.fill_diagonal(spat, 0.0)
        self.spat = spat
        self.win = np.hanning(256)
        self.low = np.fft.rfftfreq(256) < 1 / 16

    def gf(self, flat):
        v = flat / 63.0
        dv = np.sqrt(np.abs(v[:, None] - v[None, :]))
        return float((self.spat * np.exp(-dv / 1.0 ** 2)).sum())

    def lowband(self, s):
        s = (s - s.mean()) / s.std()
        X = np.fft.rfft(s.reshape(-1, 256) * self.win, axis=1)
        Pw = (np.abs(X) ** 2).mean(axis=0) / (self.win ** 2).sum()
        return float(Pw[self.low].mean())

    @staticmethod
    def xc_rms(a, b, L=16):
        a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
        n = len(a)
        fa = np.fft.rfft(a, 2 * n); fb = np.fft.rfft(b, 2 * n)
        cc = np.fft.irfft(np.conj(fa) * fb, 2 * n) / n
        lags = np.r_[cc[:L + 1], cc[-L:]]
        return float(np.sqrt((lags ** 2).mean()))

    def parts(self, flat):
        v = flat.astype(float)
        lbx = np.mean([self.lowband(v[p]) for p in self.PX])
        lby = np.mean([self.lowband(v[p]) for p in self.PY])
        xr = np.mean([self.xc_rms(v[a], v[b]) for a, b in zip(self.PX, self.PY)])
        return self.gf(flat), lbx, lby, xr

    def objective(self, flat):
        g, a, b, x = self.parts(flat)
        return g + SEARCH_LAMBDA * (a + b) + SEARCH_MU * x

    @staticmethod
    def spacing_ok(flat):
        for k, r2 in ((8, 4.0), (16, 2.0)):
            idx = np.where(flat < k)[0]
            sub = D2[np.ix_(idx, idx)] + np.eye(len(idx)) * 99
            if sub.min() < r2 - 1e-9:
                return False
        return True

    def anneal(self, seed, iters=SEARCH_ITERS):
        rng = SplitMix64(seed)
        fixed = [y * 8 + x for (x, y) in SEED]
        free = [i for i in range(64) if i not in fixed]
        flat = -np.ones(64, dtype=int)
        for r, i in enumerate(fixed):
            flat[i] = r
        for r in range(4, 16):                       # greedy farthest-point start
            placed = np.where(flat >= 0)[0]
            cand = [i for i in free if flat[i] < 0]
            d = [D2[i, placed].min() for i in cand]
            best = max(d)
            ties = [c for c, dd in zip(cand, d) if dd == best]
            flat[ties[rng.below(len(ties))]] = r
        rest = [i for i in free if flat[i] < 0]
        vals = list(range(16, 64))
        for k in range(len(vals) - 1, 0, -1):
            j = rng.below(k + 1)
            vals[k], vals[j] = vals[j], vals[k]
        for i, r in zip(rest, vals):
            flat[i] = r
        cur = self.objective(flat)
        best, bestf = cur, flat.copy()
        for t in range(iters):
            T = 0.5 * (1 - t / iters) + 1e-4
            a = free[rng.below(len(free))]
            b = free[rng.below(len(free))]
            if a == b:
                continue
            flat[a], flat[b] = flat[b], flat[a]
            if not self.spacing_ok(flat):
                flat[a], flat[b] = flat[b], flat[a]
                continue
            new = self.objective(flat)
            if new <= cur or rng.unit() < math.exp(-(new - cur) / T):
                cur = new
                if cur < best:
                    best, bestf = cur, flat.copy()
            else:
                flat[a], flat[b] = flat[b], flat[a]
        return bestf.reshape(8, 8)


def selection_score(M):
    """score used to choose among annealed masks: random ray starts, as in use"""
    dtx, dty = D["RS_DTX"], D["RS_DTY"]
    rng = SplitMix64(2026)
    flat = M.flatten().astype(float)
    lbx, lby, xc = [], [], []
    for _ in range(16):
        p, _ = ray_path(4096, rng.below(8), rng.below(8), rng.below(dtx + dty) - dty, False)
        q, _ = ray_path(4096, rng.below(8), rng.below(8), rng.below(dtx + dty) - dty, True)
        lbx.append(lowband(flat[p])[0]); lby.append(lowband(flat[q])[0])
        xc.append(xcorr_max(flat[p], flat[q]))
    return float(np.mean(lbx)), float(np.mean(lby)), float(np.mean(xc))


def _anneal_seed(seed):
    return seed, MaskSearch().anneal(seed)


def mask_search():
    head("mask search: re-run the annealing for all seeds and apply the selection rule")
    print(f"  lambda {SEARCH_LAMBDA}, mu {SEARCH_MU}, {SEARCH_ITERS} iterations, seeds "
          f"{SEARCH_SEEDS.start}..{SEARCH_SEEDS.stop - 1} (takes a while)")
    from multiprocessing import Pool
    with Pool() as pool:
        results = pool.map(_anneal_seed, list(SEARCH_SEEDS))
    rows = []
    for seed, M in results:
        lbx, lby, xc = selection_score(M)
        rows.append((xc > 3 / 64, max(lbx, lby), gf_energy(M.flatten()), seed, M))
        print(f"    seed {seed:2d}: LB_x {lbx:.4f} LB_y {lby:.4f} XC {xc:.4f} GF {rows[-1][2]:.3f}")
    rows.sort(key=lambda r: r[:3])
    best = rows[0]
    print(f"  rule: XC <= 3/sqrt(4096), then smallest worse-axis LB, then lowest GF -> seed {best[3]}")
    same = np.array_equal(best[4], MASK)
    print(f"  identical to RS_MASK in right_stick.gpc: {same}")
    if not same:
        print("  (different floating-point libraries can steer the annealer elsewhere; the")
        print("   committed mask is what the checks above verify)")
        print(best[4])


if __name__ == "__main__":
    poisson_disk()
    mask_checks()
    voxel_rays()
    bellman()
    hjb()
    smith_bode_psd()
    cramer_rao()
    admm_section()
    cross_ratio_section()
    kolmogorov()
    fusion_accuracy()
    fixed_point()
    if "--search" in sys.argv:
        mask_search()
    print("\nall checks passed")
