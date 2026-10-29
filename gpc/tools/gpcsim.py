"""Reference interpreter for the subset of Cronus Zen GPC used by right_stick.gpc.

It exists so the tests run the actual .gpc text, not a re-implementation.
Semantics follow the Zen Studio Live language reference:

* one runtime type: signed 32-bit int; globals only (function parameters are
  the only locals); runtime arrays `int a[N];`; typed const arrays (1-D/2-D)
* division truncates toward zero, `%` takes the sign of the dividend
* sections in the documented order: define, const, int, init, main, combo,
  function

Deliberately stricter than the device, to catch bugs early:

* any 32-bit overflow raises (the device would wrap silently)
* array indexes are bounds-checked
* undeclared names, assignments to constants, arity mismatches, recursion,
  parameters shadowing globals and mixed valued/bare returns are errors
"""
import re

INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1


class GPCError(Exception):
    pass


# --------------------------------------------------------------------------- lexer
_TOKEN = re.compile(r"""
    (?P<ws>\s+|//[^\n]*|/\*.*?\*/)
  | (?P<num>0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)
  | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><<=|>>=|&&=|\|\|=|\^\^=|==|!=|<=|>=|&&|\|\||\^\^|<<|>>|\+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^=
          |[-+*/%<>=!~&|^?:;,(){}\[\]])
""", re.S | re.X)

KEYWORDS = {"define", "const", "int", "init", "main", "combo", "function", "if", "else", "while",
            "do", "for", "break", "continue", "return", "data", "remap", "unmap"}


def tokenize(src):
    pos, out, line = 0, [], 1
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m:
            raise GPCError(f"line {line}: unexpected character {src[pos]!r}")
        kind = m.lastgroup
        text = m.group(kind)
        if kind == "num":
            if text[:2] in ("0x", "0X"):
                val = int(text, 16)
            elif text[:2] in ("0b", "0B"):
                val = int(text[2:], 2)
            else:
                val = int(text)
            out.append(("num", val, line))
        elif kind == "id":
            out.append(("kw" if text in KEYWORDS else "id", text, line))
        elif kind == "op":
            out.append(("op", text, line))
        line += text.count("\n")
        pos = m.end()
    out.append(("eof", None, line))
    return out


# --------------------------------------------------------------------------- parser
BIN_LEVELS = [
    ["||"], ["^^"], ["&&"], ["|"], ["^"], ["&"], ["==", "!="], [">", ">=", "<", "<="],
    ["<<", ">>"], ["+", "-"], ["*", "/", "%"],
]
ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
CONST_TYPES = {"int8": (-128, 127), "uint8": (0, 255), "int16": (-32768, 32767),
               "uint16": (0, 65535), "int32": (INT_MIN, INT_MAX)}
SECTION_ORDER = ["define", "const", "int", "init", "main", "combo", "function"]


class Parser:
    def __init__(self, src, builtin_consts):
        self.t = tokenize(src)
        self.i = 0
        self.builtin_consts = builtin_consts
        self.defines, self.consts, self.globals, self.functions = {}, {}, {}, {}
        self.init, self.main = [], []
        self.section = 0

    # token helpers
    def peek(self, k=0):
        return self.t[self.i + k]

    def take(self, kind=None, val=None):
        tok = self.t[self.i]
        if (kind and tok[0] != kind) or (val is not None and tok[1] != val):
            raise GPCError(f"line {tok[2]}: expected {val or kind}, got {tok[1]!r}")
        self.i += 1
        return tok

    def accept(self, kind, val=None):
        tok = self.t[self.i]
        if tok[0] == kind and (val is None or tok[1] == val):
            self.i += 1
            return tok
        return None

    def order(self, name, line):
        idx = SECTION_ORDER.index(name)
        if idx < self.section:
            raise GPCError(f"line {line}: '{name}' section after '{SECTION_ORDER[self.section]}'")
        self.section = idx

    def declared(self, name):
        return name in self.defines or name in self.consts or name in self.globals

    def new_name(self, name, line):
        if self.declared(name) or name in self.builtin_consts or name in BUILTIN_FUNCS:
            raise GPCError(f"line {line}: name '{name}' already in use")

    # program
    def parse(self):
        while self.peek()[0] != "eof":
            kind, val, line = self.peek()
            if kind != "kw":
                raise GPCError(f"line {line}: unexpected {val!r} at top level")
            if val == "define":
                self.order("define", line); self.i += 1
                name = self.take("id")[1]; self.new_name(name, line)
                self.take("op", "=")
                self.defines[name] = self.const_eval(self.expr())
                self.take("op", ";")
            elif val == "const":
                self.order("const", line); self.i += 1
                self.const_array(line)
            elif val == "int":
                self.order("int", line); self.i += 1
                while True:
                    name = self.take("id")[1]; self.new_name(name, line)
                    size, init = None, 0
                    if self.accept("op", "["):
                        size = self.const_eval(self.expr()); self.take("op", "]")
                        if size < 1:
                            raise GPCError(f"line {line}: array '{name}' size {size}")
                    if self.accept("op", "="):
                        if size is not None:
                            raise GPCError(f"line {line}: runtime arrays cannot have initialisers")
                        init = self.const_eval(self.expr())
                    self.globals[name] = [0] * size if size is not None else init
                    if not self.accept("op", ","):
                        break
                self.take("op", ";")
            elif val in ("init", "main"):
                self.order(val, line); self.i += 1
                (self.init if val == "init" else self.main).extend(self.block())
            elif val == "function":
                self.order("function", line); self.i += 1
                name = self.take("id")[1]
                if name in self.functions or name in BUILTIN_FUNCS:
                    raise GPCError(f"line {line}: function name '{name}' already in use")
                self.take("op", "(")
                params = []
                if not self.accept("op", ")"):
                    while True:
                        p = self.take("id")[1]
                        if self.declared(p):
                            raise GPCError(f"line {line}: parameter '{p}' shadows a global (GPC3009)")
                        params.append(p)
                        if self.accept("op", ")"):
                            break
                        self.take("op", ",")
                self.functions[name] = (params, None, line)
                body = self.block()
                self.functions[name] = (params, body, line)
            else:
                raise GPCError(f"line {line}: unsupported top-level '{val}'")
        return self

    def const_array(self, line):
        typ = self.take("id")[1]
        if typ not in CONST_TYPES:
            raise GPCError(f"line {line}: unknown const type '{typ}'")
        name = self.take("id")[1]; self.new_name(name, line)
        self.take("op", "["); self.take("op", "]")
        two_d = False
        if self.accept("op", "["):
            self.take("op", "]"); two_d = True
        self.take("op", "=")
        lo, hi = CONST_TYPES[typ]

        def row():
            self.take("op", "{")
            vals = []
            while True:
                v = self.const_eval(self.expr())
                if not lo <= v <= hi:
                    raise GPCError(f"line {line}: {v} does not fit {typ}")
                vals.append(v)
                if self.accept("op", "}"):
                    return vals
                self.take("op", ",")

        if two_d:
            self.take("op", "{")
            rows = []
            while True:
                rows.append(row())
                if self.accept("op", "}"):
                    break
                self.take("op", ",")
            if len({len(r) for r in rows}) != 1:
                raise GPCError(f"line {line}: rows of '{name}' differ in length (GPC4018)")
            self.consts[name] = rows
        else:
            self.consts[name] = row()
        self.take("op", ";")

    def const_eval(self, node):
        if node[0] == "num":
            return node[1]
        if node[0] == "name":
            if node[1] in self.defines:
                return self.defines[node[1]]
            if node[1] in self.builtin_consts:
                return self.builtin_consts[node[1]]
            raise GPCError(f"'{node[1]}' is not a compile-time constant")
        if node[0] == "unary" and node[1] == "-":
            return -self.const_eval(node[2])
        if node[0] == "bin":
            a, b = self.const_eval(node[2]), self.const_eval(node[3])
            return Interp.binop(node[1], a, b)
        raise GPCError("unsupported constant expression")

    # statements
    def block(self):
        self.take("op", "{")
        out = []
        while not self.accept("op", "}"):
            out.append(self.stmt())
        return out

    def stmt(self):
        kind, val, line = self.peek()
        if kind == "op" and val == "{":
            return ("block", self.block(), line)
        if kind == "kw":
            if val == "int":
                raise GPCError(f"line {line}: scoped variable declaration (GPC4003)")
            if val == "if":
                self.i += 1; self.take("op", "(")
                c = self.expr(); self.take("op", ")")
                then = self.stmt()
                other = self.stmt() if self.accept("kw", "else") else None
                return ("if", c, then, other, line)
            if val == "while":
                self.i += 1; self.take("op", "(")
                c = self.expr(); self.take("op", ")")
                return ("while", c, self.stmt(), line)
            if val == "do":
                self.i += 1
                body = self.stmt()
                self.take("kw", "while"); self.take("op", "(")
                c = self.expr(); self.take("op", ")"); self.take("op", ";")
                return ("do", body, c, line)
            if val == "for":
                self.i += 1; self.take("op", "(")
                a = None if self.peek()[1] == ";" else self.expr()
                self.take("op", ";")
                c = None if self.peek()[1] == ";" else self.expr()
                self.take("op", ";")
                s = None if self.peek()[1] == ")" else self.expr()
                self.take("op", ")")
                return ("for", a, c, s, self.stmt(), line)
            if val in ("break", "continue"):
                self.i += 1; self.take("op", ";")
                return (val, line)
            if val == "return":
                self.i += 1
                if self.accept("op", ";"):
                    return ("return", None, line)
                e = self.expr(); self.take("op", ";")
                return ("return", e, line)
        if kind == "op" and val == ";":
            self.i += 1
            return ("block", [], line)
        e = self.expr(); self.take("op", ";")
        return ("expr", e, line)

    # expressions
    def expr(self):
        return self.assignment()

    def assignment(self):
        left = self.ternary()
        tok = self.peek()
        if tok[0] == "op" and tok[1] in ASSIGN_OPS:
            self.i += 1
            if left[0] not in ("name", "index"):
                raise GPCError(f"line {tok[2]}: invalid assignment target")
            return ("assign", tok[1], left, self.assignment())
        return left

    def ternary(self):
        c = self.binary(0)
        if self.accept("op", "?"):
            a = self.assignment(); self.take("op", ":")
            b = self.assignment()
            return ("tern", c, a, b)
        return c

    def binary(self, level):
        if level == len(BIN_LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while True:
            tok = self.peek()
            if tok[0] == "op" and tok[1] in BIN_LEVELS[level]:
                self.i += 1
                left = ("bin", tok[1], left, self.binary(level + 1))
            else:
                return left

    def unary(self):
        tok = self.peek()
        if tok[0] == "op" and tok[1] in ("!", "-", "~"):
            self.i += 1
            return ("unary", tok[1], self.unary())
        if tok[0] == "op" and tok[1] in ("++", "--"):
            self.i += 1
            return ("preinc", tok[1], self.unary())
        return self.postfix()

    def postfix(self):
        node = self.primary()
        while True:
            tok = self.peek()
            if tok[0] == "op" and tok[1] == "[":
                self.i += 1
                idx = self.expr(); self.take("op", "]")
                node = ("index", node, idx) if node[0] != "index" else ("index2", node, idx)
            elif tok[0] == "op" and tok[1] in ("++", "--"):
                self.i += 1
                node = ("postinc", tok[1], node)
            else:
                return node

    def primary(self):
        kind, val, line = self.take()
        if kind == "num":
            return ("num", val)
        if kind == "id":
            if self.accept("op", "("):
                args = []
                if not self.accept("op", ")"):
                    while True:
                        args.append(self.expr())
                        if self.accept("op", ")"):
                            break
                        self.take("op", ",")
                return ("call", val, args, line)
            return ("name", val, line)
        if kind == "op" and val == "(":
            e = self.expr(); self.take("op", ")")
            return e
        raise GPCError(f"line {line}: unexpected {val!r}")


# --------------------------------------------------------------------------- interpreter
class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


BUILTIN_FUNCS = {"get_ival", "get_val", "get_lval", "set_val", "get_rtime", "get_ipolar",
                 "get_polar", "abs", "min", "max", "clamp", "isqrt", "random", "inv", "pow"}


class Interp:
    def __init__(self, src, builtin_consts, device):
        self.p = Parser(src, builtin_consts).parse()
        self.device = device
        self.g = {k: (list(v) if isinstance(v, list) else v) for k, v in self.p.globals.items()}
        self.ops = 0
        self.stack = []
        self.check_static()

    # ---- static checks over all code
    def check_static(self):
        p = self.p
        calls = {}

        def walk(node, params, fname):
            if isinstance(node, list):
                for n in node:
                    walk(n, params, fname)
                return
            if not isinstance(node, tuple):
                return
            tag = node[0]
            if tag == "name":
                n = node[1]
                if not (n in params or p.declared(n) or n in p.builtin_consts):
                    raise GPCError(f"line {node[2]}: undeclared identifier '{n}'")
            if tag == "assign" or tag in ("preinc", "postinc"):
                tgt = node[2]
                base = tgt
                while base[0] in ("index", "index2"):
                    base = base[1]
                n = base[1]
                if n in p.defines or n in p.consts or n in p.builtin_consts:
                    raise GPCError(f"cannot assign to constant '{n}'")
            if tag == "call":
                name, args, line = node[1], node[2], node[3]
                if name in p.functions:
                    if len(args) != len(p.functions[name][0]):
                        raise GPCError(f"line {line}: '{name}' expects {len(p.functions[name][0])} args")
                    calls.setdefault(fname, set()).add(name)
                elif name not in BUILTIN_FUNCS:
                    raise GPCError(f"line {line}: unknown function '{name}'")
            for child in node[1:]:
                walk(child, params, fname)

        walk(p.init, (), "<init>")
        walk(p.main, (), "<main>")
        for name, (params, body, line) in p.functions.items():
            walk(body, set(params), name)
            rets = []

            def find_returns(n):
                if isinstance(n, list):
                    for c in n:
                        find_returns(c)
                elif isinstance(n, tuple):
                    if n[0] == "return":
                        rets.append(n[1] is not None)
                    for c in n[1:]:
                        find_returns(c)

            find_returns(body)
            if rets and len(set(rets)) > 1:
                raise GPCError(f"function '{name}' mixes valued and bare returns (GPC5125)")
        # recursion check (direct or indirect)
        def reach(f, seen):
            for g in calls.get(f, ()):
                if g == seen[0] or (g not in seen and reach(g, seen + [g])):
                    return True
            return False
        for f in p.functions:
            if reach(f, [f]):
                raise GPCError(f"recursion through '{f}'")

    # ---- arithmetic
    @staticmethod
    def chk(v):
        if v < INT_MIN or v > INT_MAX:
            raise GPCError(f"32-bit overflow: {v}")
        return v

    @staticmethod
    def binop(op, a, b):
        if op == "+": return Interp.chk(a + b)
        if op == "-": return Interp.chk(a - b)
        if op == "*": return Interp.chk(a * b)
        if op in ("/", "%"):
            if b == 0:
                raise GPCError("division by zero")
            q = abs(a) // abs(b)
            q = q if (a >= 0) == (b >= 0) else -q
            return Interp.chk(q) if op == "/" else Interp.chk(a - q * b)
        if op == "==": return int(a == b)
        if op == "!=": return int(a != b)
        if op == "<": return int(a < b)
        if op == "<=": return int(a <= b)
        if op == ">": return int(a > b)
        if op == ">=": return int(a >= b)
        if op == "&": return a & b
        if op == "|": return a | b
        if op == "^": return a ^ b
        if op == "<<":
            if not 0 <= b <= 31: raise GPCError("shift out of range")
            return Interp.chk(a << b)
        if op == ">>":
            if not 0 <= b <= 31: raise GPCError("shift out of range")
            return a >> b
        raise GPCError(f"operator {op}")

    # ---- variables
    def frame(self):
        return self.stack[-1] if self.stack else None

    def load(self, name):
        f = self.frame()
        if f is not None and name in f:
            return f[name]
        if name in self.g:
            v = self.g[name]
            if isinstance(v, list):
                raise GPCError(f"array '{name}' used as a scalar")
            return v
        if name in self.p.defines:
            return self.p.defines[name]
        if name in self.p.builtin_consts:
            return self.p.builtin_consts[name]
        raise GPCError(f"'{name}' is not a scalar value")

    def store(self, target, value):
        self.chk(value)
        if target[0] == "name":
            f = self.frame()
            if f is not None and target[1] in f:
                f[target[1]] = value
            elif target[1] in self.g and not isinstance(self.g[target[1]], list):
                self.g[target[1]] = value
            else:
                raise GPCError(f"cannot assign '{target[1]}'")
        elif target[0] == "index":
            arr = self.array(target[1][1])
            i = self.ev(target[2])
            if not 0 <= i < len(arr):
                raise GPCError(f"index {i} out of range for '{target[1][1]}'")
            if target[1][1] not in self.g:
                raise GPCError(f"cannot assign to const array '{target[1][1]}'")
            arr[i] = value
        else:
            raise GPCError("unsupported assignment target")

    def array(self, name):
        if name in self.g and isinstance(self.g[name], list):
            return self.g[name]
        if name in self.p.consts:
            return self.p.consts[name]
        raise GPCError(f"'{name}' is not an array")

    # ---- expressions
    def ev(self, n):
        self.ops += 1
        tag = n[0]
        if tag == "num":
            return n[1]
        if tag == "name":
            return self.load(n[1])
        if tag == "index":
            arr = self.array(n[1][1])
            i = self.ev(n[2])
            if not 0 <= i < len(arr):
                raise GPCError(f"index {i} out of range for '{n[1][1]}'")
            v = arr[i]
            if isinstance(v, list):
                raise GPCError(f"2-D array '{n[1][1]}' needs two indexes")
            return v
        if tag == "index2":
            arr = self.array(n[1][1][1])
            i, j = self.ev(n[1][2]), self.ev(n[2])
            if not (0 <= i < len(arr) and isinstance(arr[i], list) and 0 <= j < len(arr[i])):
                raise GPCError(f"index [{i}][{j}] out of range for '{n[1][1][1]}'")
            return arr[i][j]
        if tag == "unary":
            v = self.ev(n[2])
            if n[1] == "-": return self.chk(-v)
            if n[1] == "!": return int(not v)
            return ~v
        if tag == "bin":
            op = n[1]
            if op == "&&":
                return int(bool(self.ev(n[2])) and bool(self.ev(n[3])))
            if op == "||":
                return int(bool(self.ev(n[2])) or bool(self.ev(n[3])))
            if op == "^^":
                return int(bool(self.ev(n[2])) != bool(self.ev(n[3])))
            return self.binop(op, self.ev(n[2]), self.ev(n[3]))
        if tag == "tern":
            return self.ev(n[2]) if self.ev(n[1]) else self.ev(n[3])
        if tag == "assign":
            op, tgt = n[1], n[2]
            v = self.ev(n[3])
            if op != "=":
                v = self.binop(op[:-1], self.ev(tgt), v)
            self.store(tgt, v)
            return v
        if tag in ("preinc", "postinc"):
            old = self.ev(n[2])
            new = self.chk(old + (1 if n[1] == "++" else -1))
            self.store(n[2], new)
            return new if tag == "preinc" else old
        if tag == "call":
            return self.call(n[1], [self.ev(a) for a in n[2]], n[3])
        raise GPCError(f"unknown node {tag}")

    def call(self, name, args, line):
        if name in self.p.functions:
            params, body, _ = self.p.functions[name]
            if len(self.stack) > 32:
                raise GPCError("call depth")
            self.stack.append(dict(zip(params, args)))
            try:
                self.run(body)
                ret = 0
            except _Return as r:
                ret = r.value if r.value is not None else 0
            finally:
                self.stack.pop()
            return ret
        return self.device.builtin(name, args, line)

    # ---- statements
    def run(self, stmts):
        for s in stmts:
            self.exec(s)

    def exec(self, s):
        self.ops += 1
        tag = s[0]
        if tag == "expr":
            self.ev(s[1])
        elif tag == "block":
            self.run(s[1])
        elif tag == "if":
            if self.ev(s[1]):
                self.exec(s[2])
            elif s[3] is not None:
                self.exec(s[3])
        elif tag == "while":
            guard = 0
            while self.ev(s[1]):
                guard += 1
                if guard > 100000:
                    raise GPCError("runaway loop")
                try:
                    self.exec(s[2])
                except _Break:
                    break
                except _Continue:
                    continue
        elif tag == "do":
            guard = 0
            while True:
                guard += 1
                if guard > 100000:
                    raise GPCError("runaway loop")
                try:
                    self.exec(s[1])
                except _Break:
                    break
                except _Continue:
                    pass
                if not self.ev(s[2]):
                    break
        elif tag == "for":
            if s[1] is not None:
                self.ev(s[1])
            guard = 0
            while s[2] is None or self.ev(s[2]):
                guard += 1
                if guard > 100000:
                    raise GPCError("runaway loop")
                try:
                    self.exec(s[4])
                except _Break:
                    break
                except _Continue:
                    pass
                if s[3] is not None:
                    self.ev(s[3])
        elif tag == "break":
            raise _Break()
        elif tag == "continue":
            raise _Continue()
        elif tag == "return":
            raise _Return(None if s[1] is None else self.ev(s[1]))
        else:
            raise GPCError(f"unknown statement {tag}")

    # ---- entry points
    def run_init(self):
        self.run(self.p.init)

    def run_main(self):
        self.ops = 0
        self.run(self.p.main)
        return self.ops
