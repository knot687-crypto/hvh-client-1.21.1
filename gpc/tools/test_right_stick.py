"""Tests that run the real right_stick.gpc in the gpcsim interpreter.

    python3 test_right_stick.py          (or: python3 -m pytest test_right_stick.py)

Covers: static checks, bit-exact agreement with the independent reference,
the quantiser invariants (exhaustively, via the reference), rest/saturation,
behaviour under wrong assumptions about the Zen, timing, and overflow.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gpcsim import Interp, GPCError  # noqa: E402
from zen_model import ZenModel, BUILTIN_CONSTS as C  # noqa: E402
from reference import Reference, threshold  # noqa: E402

SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "right_stick.gpc")
SRC = open(SRC_PATH).read()
RX, RY = C["PS4_RX"], C["PS4_RY"]


def boot(**zen):
    dev = ZenModel(**zen)
    vm = Interp(SRC, C, dev)
    vm.run_init()
    return vm, dev


def mask_of(vm):
    return vm.p.consts["RS_MASK"]


def trajectory(n, seed, amp=60.0):
    """Smooth random 2-D stick motion with holds, releases and tremor."""
    rnd = random.Random(seed)
    x = y = 0.0
    tx = ty = 0.0
    out = []
    k = 0
    while len(out) < n:
        mode = rnd.random()
        if mode < 0.15:                                   # release + rest
            for _ in range(rnd.randint(5, 40)):
                out.append((0.0, 0.0))
            x = y = 0.0
            continue
        tx = max(-100.0, min(100.0, rnd.gauss(0, amp)))
        ty = max(-100.0, min(100.0, rnd.gauss(0, amp * 0.5)))
        steps = rnd.randint(8, 60)
        x0, y0 = x, y
        f1, f2 = rnd.uniform(8, 12), rnd.uniform(8, 12)
        a = rnd.uniform(0.05, 0.4)
        for s in range(1, steps + 1):
            u = s / steps
            h = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5     # minimum jerk
            k += 1
            x = x0 + (tx - x0) * h + a * math.sin(2 * math.pi * f1 * k * 0.01)
            y = y0 + (ty - y0) * h + a * math.sin(2 * math.pi * f2 * k * 0.01 + 1)
            out.append((max(-100.0, min(100.0, x)), max(-100.0, min(100.0, y))))
    return out[:n]


# ------------------------------------------------------------------ static
def test_parses_and_static_checks():
    vm, _ = boot()
    p = vm.p
    assert set(p.consts) == {"RS_MASK", "RS_COS"}
    mask = p.consts["RS_MASK"]
    assert sorted(v for row in mask for v in row) == list(range(64))
    for rank, (x, y) in enumerate([(0, 0), (4, 0), (2, 4), (6, 4)]):
        assert mask[y][x] == rank, "ranks 0..3 must be the r = 4 Poisson-disk seed"
    cos = p.consts["RS_COS"]
    assert cos == [round(16384 * math.cos(math.radians(d))) for d in range(91)]
    assert p.defines["RS_DTY"] == round(65536 * (1 + 5 ** 0.5) / 2)
    assert p.defines["RS_SIN1"] == math.ceil(16384 * math.sin(math.radians(1)))
    assert len(p.globals) + sum(len(v) - 1 for v in p.globals.values() if isinstance(v, list)) < 1020


def test_static_checker_catches_mistakes():
    bad = {
        "local variable": "main { int a; }",
        "undeclared": "main { a = 1; }",
        "section order": "main { } define X = 1;",
        "recursion": "int a; main { a = f(1); } function f(p) { return f(p); }",
        "mixed returns": "int a; main { a = f(1); } function f(p) { if (p) { return; } return 1; }",
        "shadowing": "int a; main { a = f(1); } function f(a) { return a; }",
    }
    for name, src in bad.items():
        try:
            Interp(src, C, ZenModel())
        except GPCError:
            continue
        raise AssertionError(f"checker missed: {name}")


# ------------------------------------------------------------------ equivalence
def run_both(samples, dts=None, **zen):
    vm, dev = boot(**zen)
    ref = Reference(mask_of(vm))
    outs = []
    for n, (x, y) in enumerate(samples):
        dt = dts[n] if dts else 10
        dev.set_stick(x, y, dt)
        vm.run_main()
        zx, zy = dev.ival("x"), dev.ival("y")
        R, A = dev.polar(0), dev.polar(1)
        rx, ry = ref.step(zx, zy, R, A, dt)
        gx, gy = dev.out[RX], dev.out[RY]
        assert (gx, gy) == (rx, ry), f"sample {n}: gpc {gx, gy} != reference {rx, ry}"
        assert vm.g["smith_m"] == ref.m, f"sample {n}: smith state differs"
        assert vm.g["rng"] == ref.rng, f"sample {n}: entropy pool differs"
        assert vm.g["pol_state"] == ref.state
        assert vm.g["hyp_alive"] == ref.alive, f"sample {n}: hypothesis sets differ"
        outs.append((x, y, zx, zy, gx, gy, vm.g["vx"], vm.g["vy"], vm.g["pol_state"]))
    return vm, outs


def test_bit_exact_against_reference():
    for seed in (1, 2, 3):
        run_both(trajectory(3000, seed))


# ------------------------------------------------------------------ quantiser
def test_quantiser_invariants_exhaustive():
    """two-level output and |smith_m| < 100 for every reachable case"""
    ref = Reference([[r + 8 * c for r in range(8)] for c in range(8)])
    taus = sorted({threshold(r) for r in range(64)})
    assert taus[0] >= 1 and taus[-1] <= 99 and len(taus) == 64
    for m in range(-99, 100):
        for fr in range(100):
            for tau in taus:
                for base in (-3700, 4200):
                    v = base + fr
                    ref.m = [m, 0]
                    y = ref.quantise(0, v, tau)
                    assert y in (v // 100, v // 100 + 1)
                    assert -100 < ref.m[0] < 100
                    if fr == 0:
                        assert y == v // 100 and ref.m[0] == m


def test_quantiser_gpc_matches_reference_function():
    vm, _ = boot()
    ref = Reference(mask_of(vm))
    rnd = random.Random(7)
    for _ in range(4000):
        m = rnd.randint(-99, 99)
        v = rnd.randint(-10000, 10000)
        tau = threshold(rnd.randint(0, 63))
        vm.g["smith_m"][0] = m
        ref.m[0] = m
        assert vm.call("rs_quantise", [0, v, tau], 0) == ref.quantise(0, v, tau)
        assert vm.g["smith_m"][0] == ref.m[0]


def test_mean_tracking_bound():
    """sum of (output - target) telescopes to the mismatch state, so it stays within one step"""
    ref = Reference([[r + 8 * c for r in range(8)] for c in range(8)])
    rnd = random.Random(3)
    for trial in range(200):
        ref.m = [0, 0]
        v0 = rnd.randint(-9900, 9900)
        acc = 0
        for n in range(500):
            v = v0 + int(300 * math.sin(n * rnd.uniform(0.001, 0.05)))
            tau = threshold(rnd.randint(0, 63))
            acc += ref.quantise(0, v, tau) * 100 - v
            assert abs(acc) < 100


# ------------------------------------------------------------------ behaviour
def test_rest_is_exact_zero_and_clears_state():
    vm, dev = boot()
    for x, y in trajectory(400, 5):
        dev.set_stick(x, y)
        vm.run_main()
    for _ in range(3):
        dev.set_stick(0.2, -0.3)            # inside the zero cell
        vm.run_main()
        assert dev.out[RX] == 0 and dev.out[RY] == 0
        assert vm.g["smith_m"] == [0, 0]


def test_output_bounds():
    """|out - true stick| < 1.6 for every convention; with the Zen's most
    likely conventions (both rounded) and once identified, the output is
    always within 1 point of the raw whole-percent reading"""
    for angle, integer in [("round", "round"), ("trunc", "trunc"), ("round", "floor")]:
        _, outs = run_both(trajectory(2500, 11), rounding=integer, angle_mode=angle)
        for (x, y, zx, zy, gx, gy, vx, vy, st) in outs:
            if zx == 0 and zy == 0:
                assert gx == 0 and gy == 0
                continue
            assert abs(gx - x) < 1.6 and abs(gy - y) < 1.6, (x, y, gx, gy)
    vm, dev = boot()
    for n, (x, y) in enumerate(trajectory(4000, 17)):
        dev.set_stick(x, y)
        vm.run_main()
        if sum(vm.g["hyp_alive"]) == 1:
            assert abs(dev.out[RX] - dev.ival("x")) <= 1 and abs(dev.out[RY] - dev.ival("y")) <= 1


CONVENTIONS = [(a, c) for a in ("round", "trunc") for c in ("round", "trunc", "floor")]


def test_conventions_identified_and_never_wrongly_refuted():
    """bounded-error identification: the true convention pair always survives;
    with enough motion it is the only survivor"""
    for k, (angle, integer) in enumerate(CONVENTIONS):
        for seed in (12, 13):
            vm, _ = run_both(trajectory(2500, seed), rounding=integer, angle_mode=angle)
            assert vm.g["pol_state"] == 1, f"polar channel rejected under {angle}/{integer}"
            assert vm.g["hyp_alive"][k] == 1, f"true hypothesis {angle}/{integer} refuted"
            assert sum(vm.g["hyp_alive"]) == 1, f"{angle}/{integer}: still open {vm.g['hyp_alive']}"


def test_wrong_polar_conventions_fall_back_to_exact_passthrough():
    for zen in ({"angle_mode": "mirror"}, {"angle_mode": "offset90"},
                {"radius_scale": 0.7}, {"radius_scale": 1.3}):
        vm, outs = run_both(trajectory(1500, 13), **zen)
        assert vm.g["pol_state"] == 2, f"bad convention {zen} not detected"
        # once rejected, the output is exactly the integer reading
        first_bad = next(i for i, o in enumerate(outs) if o[8] == 2)
        assert first_bad < 200, f"rejection took {first_bad} samples under {zen}"
        for (x, y, zx, zy, gx, gy, vx, vy, st) in outs[first_bad + 1:]:
            assert (gx, gy) == (zx, zy)


def test_fusion_improves_low_frequency_error():
    """camera-angle (integrated) error over 1 s windows: fused vs integer"""
    samples = trajectory(6000, 21)
    _, outs = run_both(samples)

    def win_rms(err, w=100):
        c = [0.0]
        for e in err:
            c.append(c[-1] + e)
        means = [(c[i + w] - c[i]) / w for i in range(len(err) - w)]
        return math.sqrt(sum(m * m for m in means) / len(means))

    raw = win_rms([o[2] - o[0] for o in outs])
    ours = win_rms([o[4] - o[0] for o in outs])
    print(f"    1 s windowed error: integer {raw:.4f} %, fused {ours:.4f} %")
    assert ours < 0.5 * raw, (ours, raw)


def test_saturation_and_corners():
    vm, dev = boot()
    ref = Reference(mask_of(vm))
    pts = [(100, 0), (-100, 0), (0, 100), (0, -100), (100, 100), (-100, -100), (71, 71), (99.6, 0), (-99.4, 3)]
    for x, y in pts * 20:
        dev.set_stick(x, y)
        vm.run_main()
        rx, ry = ref.step(dev.ival("x"), dev.ival("y"), dev.polar(0), dev.polar(1), 10)
        assert (dev.out[RX], dev.out[RY]) == (rx, ry)
        if abs(dev.ival("x")) == 100:
            assert dev.out[RX] == dev.ival("x")


def test_fast_flips_and_overflow_safety():
    rnd = random.Random(99)
    samples = [(rnd.choice([-100, 100, 0, rnd.uniform(-100, 100)]), rnd.uniform(-100, 100)) for _ in range(3000)]
    run_both(samples)                    # any 32-bit overflow raises inside gpcsim


# ------------------------------------------------------------------ timing / cross ratio
def test_cross_ratio_timing_test():
    vm, _ = boot()
    f = lambda a, b, c: vm.call("rs_uniform_timing", [a, b, c], 0)
    for h in (1, 2, 5, 10, 16, 20, 40):
        assert f(h, h, h) == 1, h                    # evenly spaced at any period
    assert f(10, 11, 10) == 1 and f(10, 12, 10) == 1  # explained by 1 ms quantisation
    assert f(10, 13, 10) == 0 and f(10, 20, 10) == 0 and f(10, 10, 25) == 0


def test_extreme_rtime_values_are_safe():
    samples = trajectory(300, 33)
    dts = [10] * 300
    for k, v in ((5, 0), (50, 10 ** 6), (51, 2 ** 31 - 1), (120, 999), (200, 1001)):
        dts[k] = v
    run_both(samples, dts)               # no overflow, still bit-exact


def test_timing_glitch_resets_integrated_state():
    samples = trajectory(800, 31)
    dts = [10] * 800
    dts[400] = 25
    vm, dev = boot()
    for n, (x, y) in enumerate(samples):
        dev.set_stick(x, y, dts[n])
        vm.run_main()
        if n in (400, 401, 402):
            assert vm.g["t_uniform"] == 0
    run_both(samples, dts)


# ------------------------------------------------------------------ voxel rays / entropy
def test_rays_and_entropy_pool_stay_in_range():
    vm, dev = boot()
    for x, y in trajectory(5000, 41):
        dev.set_stick(x, y)
        vm.run_main()
        for k in range(2):
            assert 0 <= vm.g["ray_cx"][k] < 8 and 0 <= vm.g["ray_cy"][k] < 8
            assert -106039 <= vm.g["ray_d"][k] < 65536
        assert 1 <= vm.g["rng"] <= 2 ** 31 - 2
        if dev.ival("x") or dev.ival("y"):
            assert 1 <= vm.g["tau_x"] <= 99 and 1 <= vm.g["tau_y"] <= 99


def test_vm_cost():
    vm, dev = boot()
    worst = 0
    for x, y in trajectory(3000, 51):
        dev.set_stick(x, y)
        worst = max(worst, vm.run_main())
    print(f"    interpreted operations per cycle: worst {worst}")
    assert worst < 2000


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
