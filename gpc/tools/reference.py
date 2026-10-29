"""Independent reference implementation of right_stick.gpc.

Written directly from the algorithm description, not translated line by
line, so that agreement with the interpreted script is a real cross-check.
All arithmetic is exact integer arithmetic.
"""
import math

Q = 100
POL_FULL = 32767
SIN05_Q14 = math.ceil(16384 * math.sin(math.radians(0.5)))    # 143
SIN1_Q14 = math.ceil(16384 * math.sin(math.radians(1)))       # 286
SLACK = 4
DTX, DTY = 65536, round(65536 * (1 + 5 ** 0.5) / 2)          # 65536, 106039
PM_M, PM_A = 2 ** 31 - 1, 16807
COS = [round(16384 * math.cos(math.radians(d))) for d in range(91)]

UNKNOWN, VALID, INVALID = 0, 1, 2


def cos_q14(deg):
    deg %= 360
    if deg <= 90:
        return COS[deg]
    if deg <= 180:
        return -COS[180 - deg]
    if deg <= 270:
        return -COS[deg - 180]
    return COS[360 - deg]


def round_div(a, b):
    """round half away from zero, b > 0"""
    q = (abs(a) + b // 2) // b
    return q if a >= 0 else -q


def ceil_div(a, b):
    return -((-a) // b)


def threshold(rank):
    return (rank * Q + 82) // 64


def uniform_timing(a, b, c):
    def f(a, b, c):
        return 3 * (a + b) * (b + c) - 4 * b * (a + b + c)
    return int(f(a - 1, b + 1, c - 1) <= 0 <= f(a + 1, b - 1, c + 1))


class Reference:
    def __init__(self, mask):
        self.mask = [list(r) for r in mask]
        self.m = [0, 0]
        self.rng = 1043618065
        self.ray = [[0, 0, 0], [0, 0, 0]]
        self.reseed()
        self.dts = [10, 10, 10]
        self.state = UNKNOWN
        self.alive = [1] * 6
        self.resting = True
        self.v = (0, 0)
        self.tau = (0, 0)
        self.uniform = 1

    # Park-Miller minimal standard, exact
    def pm(self):
        self.rng = (self.rng * PM_A) % PM_M

    def mix(self, k):
        self.rng = (self.rng + k) % PM_M or 1
        self.pm()

    def reseed(self):
        for r in self.ray:
            self.pm()
            r[0] = self.rng % 8
            r[1] = (self.rng // 8) % 8
            r[2] = (self.rng // 64) % (DTX + DTY) - DTY

    def advance(self):
        cx, cy, d = self.ray[0]
        if d < 0:
            cx, d = (cx + 1) % 8, d + DTX
        else:
            cy, d = (cy + 1) % 8, d - DTY
        self.ray[0] = [cx, cy, d]
        cx, cy, d = self.ray[1]
        if d < 0:
            cy, d = (cy + 1) % 8, d + DTX
        else:
            cx, d = (cx - 1) % 8, d - DTY
        self.ray[1] = [cx, cy, d]
        self.tau = tuple(threshold(self.mask[r[1]][r[0]]) for r in self.ray)

    @staticmethod
    def cell(z, conv):
        """offset interval of the true value relative to 100 z, in 0.01 %"""
        if conv == 0:
            return -50, 50                      # rounded
        if conv == 1:                           # truncated toward zero
            return (0, 100) if z > 0 else ((-100, 0) if z < 0 else (-100, 100))
        return 0, 100                           # floored

    def fuse(self, zx, zy, R, A):
        """returns (status, x, y) with status 1 fused, 0 n/a, -1 inconsistent"""
        if R >= POL_FULL or abs(zx) >= 100 or abs(zy) >= 100:
            return 0, None, None
        rho = (R * 10000 + 16383) // 32767
        c, s = cos_q14(A), cos_q14(A + 270)
        px, py = round_div(rho * c, 16384), -round_div(rho * s, 16384)
        tx, ty = -s, -c
        u05 = (rho * SIN05_Q14 + 16383) // 16384 + SLACK
        u1 = (rho * SIN1_Q14 + 16383) // 16384 + SLACK
        hull = None
        for h in range(6):
            if not self.alive[h]:
                continue
            lo, hi = (-u05, u05) if h < 3 else (-SLACK, u1)
            for p, t, z in ((px, tx, zx), (py, ty, zy)):
                clo, chi = self.cell(z, h % 3)
                bl = z * Q + clo - SLACK - p
                bh = z * Q + chi + SLACK - p
                if t == 0:
                    if bl > 0 or bh < 0:
                        lo, hi = 1, 0
                elif t > 0:
                    lo = max(lo, ceil_div(bl * 16384, t))
                    hi = min(hi, (bh * 16384) // t)
                else:
                    lo = max(lo, ceil_div(-bh * 16384, -t))
                    hi = min(hi, (-bl * 16384) // -t)
            if lo > hi:
                self.alive[h] = 0
            else:
                hull = (lo, hi) if hull is None else (min(hull[0], lo), max(hull[1], hi))
        if hull is None:
            return -1, None, None
        t = (hull[0] + hull[1]) // 2
        return 1, px + round_div(t * tx, 16384), py + round_div(t * ty, 16384)

    def quantise(self, ax, v, tau):
        fl = v // Q
        fr = v - fl * Q
        if fr == 0:
            return fl
        w = v - self.m[ax]
        y = fl + 1 if w - fl * Q >= tau else fl
        self.m[ax] = y * Q - w
        return y

    def step(self, zx, zy, R, A, dt):
        self.dts = [self.dts[1], self.dts[2], min(1000, max(1, dt))]
        self.uniform = uniform_timing(*self.dts)
        if zx == 0 and zy == 0:
            self.m = [0, 0]
            self.resting = True
            self.v = (0, 0)
            return 0, 0
        if self.resting:
            self.reseed()
            self.resting = False
        self.mix(R * 361 + A)
        self.mix((zx + 101) * 203 + (zy + 101) + self.dts[2] * 41209)
        vx, vy = zx * Q, zy * Q
        if self.state != INVALID:
            st, fx, fy = self.fuse(zx, zy, R, A)
            decisive = abs(zx) + abs(zy) >= 3
            if st == -1 and decisive:
                self.state = INVALID
            if st == 1 and self.state == UNKNOWN and decisive:
                self.state = VALID
            if st == 1 and self.state == VALID:
                vx, vy = fx, fy
        vx = max(-10000, min(10000, vx))
        vy = max(-10000, min(10000, vy))
        self.v = (vx, vy)
        self.advance()
        if not self.uniform:
            self.m = [0, 0]
        return self.quantise(0, vx, self.tau[0]), self.quantise(1, vy, self.tau[1])
