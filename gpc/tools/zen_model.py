"""Simulated Cronus Zen input/output for running right_stick.gpc in gpcsim.

The model encodes the documented behaviour plus explicit, switchable
assumptions about what the documentation leaves open, so tests can check
that the script stays safe when an assumption is wrong:

* integer channel get_ival(PS4_RX/RY): percent, `rounding` in
  {"round", "trunc", "floor"}
* polar channel get_ipolar(POLAR_RS, ...): radius 0..32767 with 32767 at
  full deflection (`radius_scale` multiplies it), angle in whole degrees,
  X = r cos a, Y = -r sin a (the set_polar2 example in the 32-bit update
  notes); `angle_mode` in {"round", "trunc", "mirror", "offset90"}
* get_rtime(): whole milliseconds
"""
import math

BUILTIN_CONSTS = {
    "TRUE": 1, "FALSE": 0,
    "PS4_RX": 9, "PS4_RY": 10, "PS4_LX": 11, "PS4_LY": 12,
    "POLAR_RS": 2, "POLAR_LS": 4, "POLAR_RADIUS": 0, "POLAR_ANGLE": 1,
    "TRACE_1": 101, "TRACE_2": 102, "TRACE_3": 103, "TRACE_4": 104, "TRACE_5": 105, "TRACE_6": 106,
}
TRACE_RANGES = {101: 32, 102: 32, 103: 32, 104: 16, 105: 16, 106: 16}


def _q(v, mode):
    if mode == "round":
        return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))
    if mode == "trunc":
        return int(v)
    if mode == "floor":
        return int(math.floor(v))
    raise ValueError(mode)


class ZenModel:
    def __init__(self, rounding="round", radius_scale=1.0, angle_mode="round"):
        self.rounding = rounding
        self.radius_scale = radius_scale
        self.angle_mode = angle_mode
        self.x = self.y = 0.0          # true stick, percent
        self.dt = 10
        self.out = {}
        self.calls = 0

    def set_stick(self, x, y, dt=10):
        self.x, self.y, self.dt = float(x), float(y), dt
        self.out = {}

    # what the Zen would report
    def ival(self, axis):
        v = self.x if axis == "x" else self.y
        return max(-100, min(100, _q(v, self.rounding)))

    def polar(self, part):
        r = math.hypot(self.x, self.y) / 100.0
        if part == 0:
            return max(0, min(32767, int(round(32767 * min(r, 1.0) * self.radius_scale))))
        if r == 0:
            return 0
        a = math.degrees(math.atan2(-self.y, self.x))
        if self.angle_mode == "mirror":
            a = -a
        elif self.angle_mode == "offset90":
            a += 90
        a %= 360.0
        if self.angle_mode == "trunc":
            return int(a) % 360
        return int(math.floor(a + 0.5)) % 360

    # GPC built-ins used by the script
    def builtin(self, name, args, line):
        self.calls += 1
        if name == "get_ival":
            if args[0] == BUILTIN_CONSTS["PS4_RX"]:
                return self.ival("x")
            if args[0] == BUILTIN_CONSTS["PS4_RY"]:
                return self.ival("y")
            return 0
        if name == "get_ipolar":
            if args[0] != BUILTIN_CONSTS["POLAR_RS"]:
                raise AssertionError("only the right stick is read")
            return self.polar(args[1])
        if name == "get_rtime":
            return self.dt
        if name == "set_val":
            ident, v = args
            if ident in (BUILTIN_CONSTS["PS4_RX"], BUILTIN_CONSTS["PS4_RY"]):
                if not -100 <= v <= 100:
                    raise AssertionError(f"line {line}: stick output {v} out of range")
            elif ident in TRACE_RANGES:
                bits = TRACE_RANGES[ident]
                if not -(1 << (bits - 1)) <= v < (1 << (bits - 1)):
                    raise AssertionError(f"line {line}: trace {ident} value {v} exceeds {bits} bits")
            else:
                raise AssertionError(f"line {line}: unexpected set_val target {ident}")
            self.out[ident] = v
            return 0
        if name == "abs":
            return abs(args[0])
        raise AssertionError(f"line {line}: built-in '{name}' not modelled")
