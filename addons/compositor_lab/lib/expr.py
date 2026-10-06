# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Safe per-pixel expression compiler (numpy + GLSL) for the Expression node.

An expression is a single Python-syntax expression, parsed with ``ast`` and checked against a
whitelist. Nothing is ever passed to ``eval`` / ``exec``: the checked tree is lowered to a small
typed IR which is then turned into (a) a tree of numpy closures and (b) a GLSL statement block of
SSA temporaries. Both backends use float32 and the same definitions.

Types: ``f`` (scalar), ``v2``, ``v3``, ``v4`` (vectors). Scalars broadcast against vectors.

Variables
    a b c d          the four colour inputs (v4, premultiplied scene-linear RGBA)
    x y              pixel centre coordinates, ``texel + 0.5`` (y = 0 is the bottom row)
    u v              normalised coordinates ``x / w``, ``y / h``
    w h              image size in pixels
    t frame          context time in seconds / frame number
    pi tau           constants

Operators:  ``+ - * / % **``, unary ``-`` ``+`` ``not``, comparisons (scalars only, giving 1.0 or
0.0, chains allowed), ``and`` / ``or`` (scalars, 1.0 or 0.0; non-zero is true), ``a if c else b``.

Functions (componentwise on vectors unless noted)
    sin cos tan asin acos atan sqrt abs floor ceil fract exp log sign
    atan2(y, x) min(a, b) max(a, b) pow(x, y) step(edge, x)
    clamp(x, lo, hi) mix(a, b, t) smoothstep(e0, e1, x)
    length(v) -> f   dot(a, b) -> f   normalize(v)   luma(v3|v4) -> f
    noise(x, y[, z]) -> f in [0, 1] (Perlin noise of the Lab noise library)
    vec(...): vec(f) splats to v3; vec(f, f) v2; vec(f, f, f) v3; vec(f, f, f, f) v4; any mix of
    scalars and vectors whose component counts add up to 2, 3 or 4.
Swizzles: ``.r .g .b .a .rgb .rg .bgr .xyz ...`` (1 to 4 components, rgba or xyzw letters).

Definitions that differ from the raw maths so that both backends agree:
    sqrt(x) = sqrt(max(x, 0))     log(x) = log(max(x, 1e-30))     asin/acos clamp x to [-1, 1]
    a % b = a - b * floor(a / b)  mix(a, b, t) = a + (b - a) * t  normalize(0) = 0
    a ** n with a literal integer n (|n| <= 16) is repeated multiplication (negative bases work);
    any other power is pow(max(a, 0), b).
    Division by zero and out-of-range arguments are otherwise undefined (inf / nan).

Output: a scalar becomes grey (r = g = b), alpha 1; ``v3`` has alpha 1; ``v4`` is used as is.
``ExprError`` (a ValueError) explains what was rejected.
"""

import ast

import numpy as np

F32 = np.float32

MAX_LENGTH = 2000
MAX_NODES = 1500
MAX_DEPTH = 60

DIMS = {"f": 1, "v2": 2, "v3": 3, "v4": 4}
GLSL_TYPES = {"f": "float", "v2": "vec2", "v3": "vec3", "v4": "vec4"}
_BY_DIM = {1: "f", 2: "v2", 3: "v3", 4: "v4"}

COLOR_VARS = ("a", "b", "c", "d")
SCALAR_VARS = ("x", "y", "u", "v", "w", "h", "t", "frame")
CONSTANTS = {"pi": float(F32(np.pi)), "tau": float(F32(2.0 * np.pi))}

_SWIZZLE_SETS = ("rgba", "xyzw")


class ExprError(ValueError):
    """The expression is not valid or uses something outside the allowed subset."""


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------

class N:
    __slots__ = ("kind", "ty", "args", "val")

    def __init__(self, kind, ty, args=(), val=None):
        self.kind = kind
        self.ty = ty
        self.args = tuple(args)
        self.val = val


def _promote(node, ty):
    """Scalar node -> vector type `ty` (no-op if already that type)."""
    if node.ty == ty:
        return node
    if node.ty != "f":
        raise ExprError("cannot combine %s and %s" % (_tname(node.ty), _tname(ty)))
    return N("promote", ty, (node,))


def _tname(ty):
    return {"f": "a scalar", "v2": "a vec2", "v3": "a vec3", "v4": "a vec4"}[ty]


def _bcast(*nodes):
    """Common type of nodes (scalars broadcast to the vector type); returns promoted nodes."""
    ty = "f"
    for n in nodes:
        if n.ty != "f":
            if ty != "f" and ty != n.ty:
                raise ExprError("cannot combine %s with %s (component counts differ)" % (
                    _tname(ty), _tname(n.ty)))
            ty = n.ty
    return [_promote(n, ty) for n in nodes], ty


# Function table: name -> (min_args, max_args)
_UNARY_MAP = ("sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "abs", "floor", "ceil",
              "fract", "exp", "log", "sign")
_FUNCS = {n: (1, 1) for n in _UNARY_MAP}
_FUNCS.update({
    "atan2": (2, 2), "min": (2, 2), "max": (2, 2), "pow": (2, 2), "step": (2, 2),
    "clamp": (3, 3), "mix": (3, 3), "smoothstep": (3, 3),
    "length": (1, 1), "dot": (2, 2), "normalize": (1, 1), "luma": (1, 1),
    "noise": (2, 3), "vec": (1, 4),
})

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%",
           ast.Pow: "**"}
_CMPOPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==",
           ast.NotEq: "!="}


class _Builder:
    def __init__(self):
        self.count = 0
        self.vars = {}
        self.uses_noise = False

    def tick(self):
        self.count += 1
        if self.count > MAX_NODES:
            raise ExprError("expression is too large (more than %d nodes)" % MAX_NODES)

    def var(self, name):
        node = self.vars.get(name)
        if node is None:
            node = N("var", "v4" if name in COLOR_VARS else "f", val=name)
            self.vars[name] = node
        return node

    def build(self, node, depth=0):
        self.tick()
        if depth > MAX_DEPTH:
            raise ExprError("expression is nested too deeply")
        m = getattr(self, "v_" + type(node).__name__, None)
        if m is None:
            raise ExprError("unsupported syntax: %s" % type(node).__name__)
        return m(node, depth + 1)

    # -- leaves -----------------------------------------------------------
    def v_Constant(self, node, d):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ExprError("only numbers are allowed as literals (got %s)" % type(v).__name__)
        try:
            f = float(v)
        except OverflowError:
            raise ExprError("number out of range") from None
        if not np.isfinite(F32(f)):
            raise ExprError("number out of range: %s" % v)
        return N("const", "f", val=float(F32(f)))

    def v_Name(self, node, d):
        name = node.id
        if name in COLOR_VARS or name in SCALAR_VARS:
            return self.var(name)
        if name in CONSTANTS:
            return N("const", "f", val=CONSTANTS[name])
        if name in _FUNCS:
            raise ExprError("'%s' is a function: call it as %s(...)" % (name, name))
        raise ExprError("unknown name '%s'" % name)

    # -- operators --------------------------------------------------------
    def v_BinOp(self, node, d):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExprError("unsupported operator: %s" % type(node.op).__name__)
        a = self.build(node.left, d)
        b = self.build(node.right, d)
        if op == "**":
            return self._pow(a, b)
        (a, b), ty = _bcast(a, b)
        return N("bin", ty, (a, b), op)

    def _pow(self, a, b):
        if b.kind == "const" and float(b.val).is_integer() and abs(b.val) <= 16:
            n = int(b.val)
            if n == 0:
                return _promote(N("const", "f", val=1.0), a.ty)
            return N("ipow", a.ty, (a,), n)
        (a, b), ty = _bcast(a, b)
        return N("call", ty, (a, b), "pow")

    def v_UnaryOp(self, node, d):
        x = self.build(node.operand, d)
        if isinstance(node.op, ast.UAdd):
            return x
        if isinstance(node.op, ast.USub):
            if x.kind == "const":
                return N("const", "f", val=-x.val)
            return N("neg", x.ty, (x,))
        if isinstance(node.op, ast.Not):
            self._scalar(x, "not")
            return N("not", "f", (x,))
        raise ExprError("unsupported operator: %s" % type(node.op).__name__)

    def _scalar(self, x, what):
        if x.ty != "f":
            raise ExprError("'%s' needs scalar operands, got %s" % (what, _tname(x.ty)))

    def v_BoolOp(self, node, d):
        op = "and" if isinstance(node.op, ast.And) else "or"
        vals = [self.build(v, d) for v in node.values]
        for v in vals:
            self._scalar(v, op)
        out = vals[0]
        for v in vals[1:]:
            out = N("logic", "f", (out, v), op)
        if len(vals) == 1:
            out = N("logic", "f", (out, out), op)
        return out

    def v_Compare(self, node, d):
        left = self.build(node.left, d)
        out = None
        for op, right_node in zip(node.ops, node.comparators):
            sym = _CMPOPS.get(type(op))
            if sym is None:
                raise ExprError("unsupported comparison: %s" % type(op).__name__)
            right = self.build(right_node, d)
            self._scalar(left, sym)
            self._scalar(right, sym)
            c = N("cmp", "f", (left, right), sym)
            out = c if out is None else N("logic", "f", (out, c), "and")
            left = right
        return out

    def v_IfExp(self, node, d):
        c = self.build(node.test, d)
        self._scalar(c, "if")
        a = self.build(node.body, d)
        b = self.build(node.orelse, d)
        (a, b), ty = _bcast(a, b)
        return N("ifexp", ty, (c, a, b))

    # -- attribute / call -------------------------------------------------
    def v_Attribute(self, node, d):
        attr = node.attr
        sets = [s for s in _SWIZZLE_SETS if attr and all(ch in s for ch in attr)]
        if not sets or len(attr) > 4:
            raise ExprError("attribute access is only allowed for swizzles like .r .rgb .xy "
                            "(got '.%s')" % attr)
        x = self.build(node.value, d)
        if x.ty == "f":
            raise ExprError("cannot swizzle a scalar (.%s)" % attr)
        idx = tuple(sets[0].index(ch) for ch in attr)
        if max(idx) >= DIMS[x.ty]:
            raise ExprError("swizzle .%s is out of range for %s" % (attr, _tname(x.ty)))
        return N("swizzle", _BY_DIM[len(idx)], (x,), idx)

    def v_Call(self, node, d):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            what = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
            raise ExprError("function not allowed: %s" % what)
        if node.keywords:
            raise ExprError("keyword arguments are not supported")
        name = node.func.id
        lo, hi = _FUNCS[name]
        n = len(node.args)
        if not lo <= n <= hi:
            raise ExprError("%s() takes %s argument(s), got %d" % (
                name, str(lo) if lo == hi else "%d to %d" % (lo, hi), n))
        args = [self.build(a, d) for a in node.args]
        return self.call(name, args)

    def call(self, name, args):
        if name in _UNARY_MAP:
            return N("call", args[0].ty, (args[0],), name)
        if name in ("atan2", "min", "max", "pow"):
            args, ty = _bcast(*args)
            return N("call", ty, args, name)
        if name == "step":
            args, ty = _bcast(*args)
            return N("call", ty, args, name)
        if name == "clamp":
            x = args[0]
            lo = _promote(args[1], x.ty) if args[1].ty == "f" else args[1]
            hi = _promote(args[2], x.ty) if args[2].ty == "f" else args[2]
            if lo.ty != x.ty or hi.ty != x.ty:
                raise ExprError("clamp(): bounds must be scalars or the same type as the value")
            return N("call", x.ty, (x, lo, hi), name)
        if name == "mix":
            (a, b), ty = _bcast(args[0], args[1])
            t = args[2]
            if t.ty != "f" and t.ty != ty:
                raise ExprError("mix(): t must be a scalar or the same type as a and b")
            t = _promote(t, ty) if t.ty == "f" else t
            return N("call", ty, (a, b, t), name)
        if name == "smoothstep":
            args, ty = _bcast(*args)
            return N("call", ty, args, name)
        if name == "length":
            return N("call", "f", (args[0],), name)
        if name == "dot":
            if args[0].ty != args[1].ty:
                raise ExprError("dot(): both arguments must have the same type")
            return N("call", "f", args, name)
        if name == "normalize":
            return N("call", args[0].ty, (args[0],), name)
        if name == "luma":
            if args[0].ty not in ("v3", "v4"):
                raise ExprError("luma() needs a vec3 or vec4")
            return N("call", "f", (args[0],), name)
        if name == "noise":
            for a in args:
                self._scalar(a, "noise")
            if len(args) == 2:
                args = args + [N("const", "f", val=0.0)]
            self.uses_noise = True
            return N("call", "f", args, name)
        if name == "vec":
            total = sum(DIMS[a.ty] for a in args)
            if len(args) == 1 and args[0].ty == "f":
                return N("vec", "v3", (args[0],) * 3)
            if total not in (2, 3, 4):
                raise ExprError("vec(): needs 2 to 4 components in total (got %d)" % total)
            return N("vec", _BY_DIM[total], args)
        raise ExprError("function not allowed: %s" % name)


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------

def _f(v):
    return F32(v)


def _bool_f(cond):
    return np.where(cond, F32(1.0), F32(0.0)).astype(F32)


def _sat3(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def _np_ipow(x, n):
    k = abs(n)
    r = x
    for _ in range(k - 1):
        r = r * x
    return r if n > 0 else F32(1.0) / r


def _np_call(name):
    one, zero = F32(1.0), F32(0.0)
    table = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": lambda x: np.arcsin(_sat3(x, -one, one)),
        "acos": lambda x: np.arccos(_sat3(x, -one, one)),
        "atan": np.arctan,
        "sqrt": lambda x: np.sqrt(np.maximum(x, zero)),
        "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
        "fract": lambda x: x - np.floor(x),
        "exp": np.exp,
        "log": lambda x: np.log(np.maximum(x, F32(1e-30))),
        "sign": np.sign,
        "atan2": np.arctan2, "min": np.minimum, "max": np.maximum,
        "pow": lambda a, b: np.power(np.maximum(a, zero), b),
        "step": lambda e, x: np.where(x < e, zero, one).astype(F32),
        "clamp": _sat3,
        "mix": lambda a, b, t: a + (b - a) * t,
    }
    return table[name]


def _np_smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), F32(0.0), F32(1.0))
    return t * t * (F32(3.0) - F32(2.0) * t)


_LUMA = (F32(0.2126), F32(0.7152), F32(0.0722))


def _make_np(node):
    """IR node -> closure(env) -> numpy value (float32)."""
    k = node.kind
    if k == "const":
        c = F32(node.val)
        return lambda env: c
    if k == "var":
        name = node.val
        return lambda env: env[name]
    args = [_make_np(a) for a in node.args]
    if k == "promote":
        a = args[0]
        return lambda env: np.asarray(a(env))[..., None]
    if k == "neg":
        a = args[0]
        return lambda env: -a(env)
    if k == "not":
        a = args[0]
        return lambda env: _bool_f(a(env) == 0)
    if k == "bin":
        a, b = args
        op = node.val
        if op == "+":
            return lambda env: a(env) + b(env)
        if op == "-":
            return lambda env: a(env) - b(env)
        if op == "*":
            return lambda env: a(env) * b(env)
        if op == "/":
            return lambda env: a(env) / b(env)

        def mod(env):
            x, y = a(env), b(env)
            return x - y * np.floor(x / y)
        return mod
    if k == "ipow":
        a = args[0]
        n = node.val
        return lambda env: _np_ipow(a(env), n)
    if k == "cmp":
        a, b = args
        fn = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
              "==": np.equal, "!=": np.not_equal}[node.val]
        return lambda env: _bool_f(fn(a(env), b(env)))
    if k == "logic":
        a, b = args
        if node.val == "and":
            return lambda env: _bool_f((a(env) != 0) & (b(env) != 0))
        return lambda env: _bool_f((a(env) != 0) | (b(env) != 0))
    if k == "ifexp":
        c, a, b = args
        vec = node.ty != "f"

        def ifexp(env):
            cond = np.asarray(c(env)) != 0
            if vec:
                cond = cond[..., None]
            return np.where(cond, a(env), b(env))
        return ifexp
    if k == "swizzle":
        a = args[0]
        idx = list(node.val)
        if len(idx) == 1:
            i = idx[0]
            return lambda env: np.asarray(a(env))[..., i]
        return lambda env: np.asarray(a(env))[..., idx]
    if k == "vec":
        parts = [(_make_np(a), DIMS[a.ty]) for a in node.args]

        def vec(env):
            comps = []
            for fn, n in parts:
                v = np.asarray(fn(env))
                if n == 1:
                    comps.append(v)
                else:
                    comps.extend(v[..., i] for i in range(n))
            comps = np.broadcast_arrays(*comps)
            return np.stack(comps, axis=-1)
        return vec
    if k == "call":
        name = node.val
        if name == "length":
            a = args[0]
            if node.args[0].ty == "f":
                return lambda env: np.abs(a(env))
            return lambda env: np.sqrt(np.sum(np.square(a(env)), axis=-1, dtype=F32))
        if name == "dot":
            a, b = args
            if node.args[0].ty == "f":
                return lambda env: a(env) * b(env)
            return lambda env: np.sum(a(env) * b(env), axis=-1, dtype=F32)
        if name == "normalize":
            a = args[0]
            scalar = node.ty == "f"

            def normalize(env):
                v = np.asarray(a(env))
                if scalar:
                    return v / np.maximum(np.abs(v), F32(1e-20))
                ln = np.sqrt(np.sum(np.square(v), axis=-1, dtype=F32))
                return v / np.maximum(ln, F32(1e-20))[..., None]
            return normalize
        if name == "luma":
            a = args[0]

            def luma(env):
                v = np.asarray(a(env))
                return _LUMA[0] * v[..., 0] + _LUMA[1] * v[..., 1] + _LUMA[2] * v[..., 2]
            return luma
        if name == "noise":
            from . import np_noise
            a, b, c = args

            def noise(env):
                h, w = env["shape"]
                xs = np.broadcast_to(np.asarray(a(env), F32), (h, w))
                ys = np.broadcast_to(np.asarray(b(env), F32), (h, w))
                zs = np.broadcast_to(np.asarray(c(env), F32), (h, w))
                return np_noise.noise_field(1, xs, ys, zs, 0, 1.0, 1, 2.0, 0.5, False, 0.0)
            return noise
        if name == "smoothstep":
            a, b, c = args
            return lambda env: _np_smoothstep(a(env), b(env), c(env))
        fn = _np_call(name)
        return lambda env: fn(*(a(env) for a in args))
    raise AssertionError(k)


# ---------------------------------------------------------------------------
# GLSL backend: one SSA temporary per node
# ---------------------------------------------------------------------------

def _gl_float(v):
    s = "%.9g" % float(F32(v))
    if not any(ch in s for ch in ".e"):
        s += ".0"
    return s


class _Gl:
    def __init__(self):
        self.lines = []
        self.names = {}
        self.n = 0

    def emit(self, node):
        key = id(node)
        name = self.names.get(key)
        if name is not None:
            return name
        expr = self.expr(node)
        if node.kind == "const":
            name = "(%s)" % expr if expr.startswith("-") else expr
            self.names[key] = name
            return name
        name = "e%d" % self.n
        self.n += 1
        self.lines.append("    %s %s = %s;" % (GLSL_TYPES[node.ty], name, expr))
        self.names[key] = name
        return name

    def expr(self, node):
        k = node.kind
        ty = node.ty
        if k == "const":
            return _gl_float(node.val)
        if k == "var":
            return {
                "a": "in_A(texel)", "b": "in_B(texel)", "c": "in_C(texel)", "d": "in_D(texel)",
                "x": "(float(texel.x) + 0.5)", "y": "(float(texel.y) + 0.5)",
                "u": "((float(texel.x) + 0.5) / float(res.x))",
                "v": "((float(texel.y) + 0.5) / float(res.y))",
                "w": "float(res.x)", "h": "float(res.y)",
                "t": "e_time", "frame": "e_frame",
            }[node.val]
        a = [self.emit(x) for x in node.args]
        if k == "promote":
            return "%s(%s)" % (GLSL_TYPES[ty], a[0])
        if k == "neg":
            return "-%s" % a[0]
        if k == "not":
            return "(%s != 0.0) ? 0.0 : 1.0" % a[0]
        if k == "bin":
            op = node.val
            if op == "%":
                return "%s - %s * floor(%s / %s)" % (a[0], a[1], a[0], a[1])
            return "%s %s %s" % (a[0], op, a[1])
        if k == "ipow":
            n = node.val
            prod = " * ".join([a[0]] * abs(n))
            return prod if n > 0 else "%s / (%s)" % (_vec_one(ty), prod)
        if k == "cmp":
            return "(%s %s %s) ? 1.0 : 0.0" % (a[0], node.val, a[1])
        if k == "logic":
            sym = "&&" if node.val == "and" else "||"
            return "(%s != 0.0 %s %s != 0.0) ? 1.0 : 0.0" % (a[0], sym, a[1])
        if k == "ifexp":
            return "(%s != 0.0) ? %s : %s" % (a[0], a[1], a[2])
        if k == "swizzle":
            return "%s.%s" % (a[0], "".join("xyzw"[i] for i in node.val))
        if k == "vec":
            return "%s(%s)" % (GLSL_TYPES[ty], ", ".join(a))
        if k == "call":
            return self.call(node, a)
        raise AssertionError(k)

    def call(self, node, a):
        name = node.val
        ty = node.ty
        if name in ("sin", "cos", "tan", "atan", "abs", "floor", "ceil", "exp", "sign",
                    "min", "max"):
            return "%s(%s)" % (name, ", ".join(a))
        if name == "asin" or name == "acos":
            return "%s(clamp(%s, -1.0, 1.0))" % (name, a[0])
        if name == "sqrt":
            return "sqrt(max(%s, 0.0))" % a[0]
        if name == "fract":
            return "%s - floor(%s)" % (a[0], a[0])
        if name == "log":
            return "log(max(%s, 1e-30))" % a[0]
        if name == "atan2":
            return "atan(%s, %s)" % (a[0], a[1])
        if name == "pow":
            return "pow(max(%s, %s), %s)" % (a[0], _vec_zero(ty), a[1])
        if name == "step":
            return "step(%s, %s)" % (a[0], a[1])
        if name == "clamp":
            return "min(max(%s, %s), %s)" % (a[0], a[1], a[2])
        if name == "mix":
            return "%s + (%s - %s) * %s" % (a[0], a[1], a[0], a[2])
        if name == "smoothstep":
            t = "clamp((%s - %s) / (%s - %s), %s, %s)" % (
                a[2], a[0], a[1], a[0], _vec_zero(ty), _vec_one(ty))
            # t is needed twice: evaluate through a helper temporary.
            tmp = "e%d" % self.n
            self.n += 1
            self.lines.append("    %s %s = %s;" % (GLSL_TYPES[ty], tmp, t))
            return "%s * %s * (%s - 2.0 * %s)" % (tmp, tmp, _vec_const(ty, 3.0), tmp)
        if name == "length":
            return "length(%s)" % a[0] if node.args[0].ty != "f" else "abs(%s)" % a[0]
        if name == "dot":
            return "dot(%s, %s)" % (a[0], a[1]) if node.args[0].ty != "f" else \
                "%s * %s" % (a[0], a[1])
        if name == "normalize":
            if ty == "f":
                return "%s / max(abs(%s), 1e-20)" % (a[0], a[0])
            return "%s / max(length(%s), 1e-20)" % (a[0], a[0])
        if name == "luma":
            return "dot(%s.rgb, vec3(0.2126, 0.7152, 0.0722))" % a[0]
        if name == "noise":
            return "lab_noise_field(1, vec3(%s, %s, %s), 0u, 1.0, 1, 2.0, 0.5, false, 0.0)" % (
                a[0], a[1], a[2])
        raise AssertionError(name)


def _vec_const(ty, v):
    s = _gl_float(v)
    return s if ty == "f" else "%s(%s)" % (GLSL_TYPES[ty], s)


def _vec_one(ty):
    return _vec_const(ty, 1.0)


def _vec_zero(ty):
    return _vec_const(ty, 0.0)


# ---------------------------------------------------------------------------
# Compiled expression
# ---------------------------------------------------------------------------

class Compiled:
    """Result of ``compile_expr``: numpy evaluator and GLSL body for one expression."""

    def __init__(self, source, root, builder):
        self.source = source
        self.out_type = root.ty
        self.uses = frozenset(builder.vars)
        self.uses_noise = builder.uses_noise
        self._np = _make_np(root)
        gl = _Gl()
        res = gl.emit(root)
        if root.ty == "f":
            out = "vec4(%s, %s, %s, 1.0)" % (res, res, res)
        elif root.ty == "v3":
            out = "vec4(%s, 1.0)" % res
        else:
            out = res
        self.glsl_body = "\n".join(gl.lines) + "\n    out_Color = %s;\n" % out

    @property
    def inputs_used(self):
        """Colour inputs ("A".."D") the expression reads."""
        return [v.upper() for v in COLOR_VARS if v in self.uses]

    def eval_numpy(self, shape, colors, time=0.0, frame=0.0):
        """Evaluate over an image of ``shape`` = (H, W). ``colors``: {"a": (H, W, 4) array or
        anything broadcastable to it, ...}. Returns a new (H, W, 4) float32 array."""
        h, w = int(shape[0]), int(shape[1])
        env = {
            "shape": (h, w),
            "x": (np.arange(w, dtype=F32) + F32(0.5))[None, :],
            "y": (np.arange(h, dtype=F32) + F32(0.5))[:, None],
            "w": F32(w), "h": F32(h), "t": F32(time), "frame": F32(frame),
        }
        env["u"] = env["x"] / F32(w)
        env["v"] = env["y"] / F32(h)
        for name in COLOR_VARS:
            if name in self.uses:
                env[name] = np.asarray(colors[name], dtype=F32)
        with np.errstate(all="ignore"):
            r = np.asarray(self._np(env), dtype=F32)
            out = np.empty((h, w, 4), F32)
            if self.out_type == "f":
                out[..., :3] = np.broadcast_to(r, (h, w))[..., None]
                out[..., 3] = 1.0
            elif self.out_type == "v3":
                out[..., :3] = np.broadcast_to(r, (h, w, 3))
                out[..., 3] = 1.0
            else:
                out[...] = np.broadcast_to(r, (h, w, 4))
        return out


_cache = {}
_CACHE_MAX = 256


def compile_expr(text):
    """Compile (cached by string). Raises ExprError for anything not allowed."""
    comp = _cache.get(text)
    if comp is not None:
        return comp
    if not isinstance(text, str):
        raise ExprError("expression must be a string")
    src = text.strip()
    if not src:
        raise ExprError("empty expression")
    if len(src) > MAX_LENGTH:
        raise ExprError("expression is too long (max %d characters)" % MAX_LENGTH)
    if "\n" in src or "\r" in src:
        src = " ".join(src.split())
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as ex:
        raise ExprError("syntax error: %s" % (ex.msg,)) from None
    except (ValueError, RecursionError, MemoryError) as ex:
        raise ExprError("cannot parse expression: %s" % type(ex).__name__) from None
    builder = _Builder()
    try:
        root = builder.build(tree.body)
    except RecursionError:
        raise ExprError("expression is nested too deeply") from None
    if root.ty not in ("f", "v3", "v4"):
        raise ExprError("the result must be a scalar, vec3 or vec4 (got %s)" % _tname(root.ty))
    comp = Compiled(src, root, builder)
    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[text] = comp
    return comp
