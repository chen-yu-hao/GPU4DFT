"""Pure-stdlib symbolic expression DAG, automatic differentiator, and CSE."""

from __future__ import annotations

import math
from typing import Any, Sequence

_NEXT_ID = 0
_CANON_POOL: dict[tuple[Any, ...], "Expr"] = {}


class Expr:
    """Base immutable hash-consed DAG node with O(1) identity hash & equality."""

    __slots__ = ("_id",)

    def __init__(self, node_id: int) -> None:
        self._id = node_id

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: object) -> bool:
        return self is other

    def __add__(self, other: Expr | float | int) -> Expr:
        return add(self, ensure_expr(other))

    def __radd__(self, other: Expr | float | int) -> Expr:
        return add(ensure_expr(other), self)

    def __sub__(self, other: Expr | float | int) -> Expr:
        return sub(self, ensure_expr(other))

    def __rsub__(self, other: Expr | float | int) -> Expr:
        return sub(ensure_expr(other), self)

    def __mul__(self, other: Expr | float | int) -> Expr:
        return mul(self, ensure_expr(other))

    def __rmul__(self, other: Expr | float | int) -> Expr:
        return mul(ensure_expr(other), self)

    def __truediv__(self, other: Expr | float | int) -> Expr:
        return div(self, ensure_expr(other))

    def __rtruediv__(self, other: Expr | float | int) -> Expr:
        return div(ensure_expr(other), self)

    def __pow__(self, other: Expr | float | int) -> Expr:
        return power(self, ensure_expr(other))

    def __neg__(self) -> Expr:
        return neg(self)


class Const(Expr):
    __slots__ = ("value",)

    def __init__(self, node_id: int, value: float) -> None:
        super().__init__(node_id)
        self.value = value


class Var(Expr):
    __slots__ = ("name",)

    def __init__(self, node_id: int, name: str) -> None:
        super().__init__(node_id)
        self.name = name


class Unary(Expr):
    __slots__ = ("op", "arg")

    def __init__(self, node_id: int, op: str, arg: Expr) -> None:
        super().__init__(node_id)
        self.op = op
        self.arg = arg


class Binary(Expr):
    __slots__ = ("op", "lhs", "rhs")

    def __init__(self, node_id: int, op: str, lhs: Expr, rhs: Expr) -> None:
        super().__init__(node_id)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs


class Where(Expr):
    __slots__ = ("cond_op", "cond_lhs", "cond_rhs", "true_val", "false_val")

    def __init__(
        self,
        node_id: int,
        cond_op: str,
        cond_lhs: Expr,
        cond_rhs: Expr,
        true_val: Expr,
        false_val: Expr,
    ) -> None:
        super().__init__(node_id)
        self.cond_op = cond_op
        self.cond_lhs = cond_lhs
        self.cond_rhs = cond_rhs
        self.true_val = true_val
        self.false_val = false_val


def _alloc_id() -> int:
    global _NEXT_ID
    nid = _NEXT_ID
    _NEXT_ID += 1
    return nid


def c(val: float | int) -> Const:
    fval = float(val)
    if fval == 0.0:
        fval = 0.0
    # Distinguish int/float exact bit pattern
    key = ("const", fval)
    existing = _CANON_POOL.get(key)
    if existing is not None:
        return existing  # type: ignore[return-value]
    node = Const(_alloc_id(), fval)
    _CANON_POOL[key] = node
    return node


ZERO = c(0.0)
ONE = c(1.0)
TWO = c(2.0)
HALF = c(0.5)


def var(name: str) -> Var:
    key = ("var", name)
    existing = _CANON_POOL.get(key)
    if existing is not None:
        return existing  # type: ignore[return-value]
    node = Var(_alloc_id(), name)
    _CANON_POOL[key] = node
    return node


def _mk_unary(op: str, arg: Expr) -> Unary:
    key = ("un", op, arg._id)
    existing = _CANON_POOL.get(key)
    if existing is not None:
        return existing  # type: ignore[return-value]
    node = Unary(_alloc_id(), op, arg)
    _CANON_POOL[key] = node
    return node


def _mk_binary(op: str, lhs: Expr, rhs: Expr) -> Binary:
    key = ("bin", op, lhs._id, rhs._id)
    existing = _CANON_POOL.get(key)
    if existing is not None:
        return existing  # type: ignore[return-value]
    node = Binary(_alloc_id(), op, lhs, rhs)
    _CANON_POOL[key] = node
    return node


def _mk_where(
    cond_op: str, cond_lhs: Expr, cond_rhs: Expr, true_val: Expr, false_val: Expr
) -> Where:
    key = ("where", cond_op, cond_lhs._id, cond_rhs._id, true_val._id, false_val._id)
    existing = _CANON_POOL.get(key)
    if existing is not None:
        return existing  # type: ignore[return-value]
    node = Where(_alloc_id(), cond_op, cond_lhs, cond_rhs, true_val, false_val)
    _CANON_POOL[key] = node
    return node


def ensure_expr(x: Expr | float | int) -> Expr:
    if isinstance(x, Expr):
        return x
    return c(float(x))


def _is_const(x: Expr, target: float | None = None) -> bool:
    if not isinstance(x, Const):
        return False
    return True if target is None else x.value == target


def neg(a: Expr) -> Expr:
    if isinstance(a, Const):
        return c(-a.value)
    if isinstance(a, Unary) and a.op == "neg":
        return a.arg
    return _mk_unary("neg", a)


def add(a: Expr, b: Expr) -> Expr:
    if _is_const(a, 0.0):
        return b
    if _is_const(b, 0.0):
        return a
    if isinstance(a, Const) and isinstance(b, Const):
        return c(a.value + b.value)
    if a is b:
        return mul(TWO, a)
    if isinstance(b, Unary) and b.op == "neg":
        return sub(a, b.arg)
    if a._id > b._id:
        a, b = b, a
    return _mk_binary("add", a, b)


def sub(a: Expr, b: Expr) -> Expr:
    if _is_const(b, 0.0):
        return a
    if _is_const(a, 0.0):
        return neg(b)
    if isinstance(a, Const) and isinstance(b, Const):
        return c(a.value - b.value)
    if a is b:
        return ZERO
    if isinstance(b, Unary) and b.op == "neg":
        return add(a, b.arg)
    return _mk_binary("sub", a, b)


def mul(a: Expr, b: Expr) -> Expr:
    if _is_const(a, 0.0) or _is_const(b, 0.0):
        return ZERO
    if _is_const(a, 1.0):
        return b
    if _is_const(b, 1.0):
        return a
    if _is_const(a, -1.0):
        return neg(b)
    if _is_const(b, -1.0):
        return neg(a)
    if isinstance(a, Const) and isinstance(b, Const):
        return c(a.value * b.value)
    if isinstance(b, Const) and not isinstance(a, Const):
        a, b = b, a
    elif not isinstance(a, Const) and not isinstance(b, Const) and a._id > b._id:
        a, b = b, a
    if isinstance(a, Const) and isinstance(b, Binary) and b.op == "mul" and isinstance(b.lhs, Const):
        return mul(c(a.value * b.lhs.value), b.rhs)
    return _mk_binary("mul", a, b)


def div(a: Expr, b: Expr) -> Expr:
    if _is_const(a, 0.0):
        return ZERO
    if _is_const(b, 1.0):
        return a
    if _is_const(b, -1.0):
        return neg(a)
    if isinstance(a, Const) and isinstance(b, Const):
        return c(a.value / b.value)
    if a is b:
        return ONE
    return _mk_binary("div", a, b)


def power(a: Expr, b: Expr) -> Expr:
    if _is_const(b, 0.0):
        return ONE
    if _is_const(b, 1.0):
        return a
    if _is_const(b, -1.0):
        return div(ONE, a)
    if _is_const(b, 0.5):
        return sqrt(a)
    if _is_const(a, 1.0):
        return ONE
    if isinstance(a, Const) and isinstance(b, Const):
        return c(a.value ** b.value)
    return _mk_binary("pow", a, b)


def sq(a: Expr) -> Expr:
    return mul(a, a)


def cube(a: Expr) -> Expr:
    return mul(mul(a, a), a)


def sqrt(a: Expr) -> Expr:
    if isinstance(a, Const):
        return c(math.sqrt(a.value))
    return _mk_unary("sqrt", a)


def cbrt(a: Expr) -> Expr:
    if isinstance(a, Const):
        return c(math.copysign(abs(a.value) ** (1.0 / 3.0), a.value))
    return _mk_unary("cbrt", a)


def log(a: Expr) -> Expr:
    if isinstance(a, Const):
        return c(math.log(a.value))
    return _mk_unary("log", a)


def log1p(a: Expr) -> Expr:
    if isinstance(a, Const):
        return c(math.log1p(a.value))
    return _mk_unary("log1p", a)


def exp(a: Expr) -> Expr:
    if isinstance(a, Const):
        return c(math.exp(a.value))
    return _mk_unary("exp", a)


def expm1(a: Expr) -> Expr:
    if isinstance(a, Const):
        return c(math.expm1(a.value))
    return _mk_unary("expm1", a)


def atan(a: Expr) -> Expr:
    if isinstance(a, Const):
        return c(math.atan(a.value))
    return _mk_unary("atan", a)


def asinh(a: Expr) -> Expr:
    if isinstance(a, Const):
        return c(math.asinh(a.value))
    return _mk_unary("asinh", a)


def where(cond_op: str, cond_lhs: Expr, cond_rhs: Expr, true_val: Expr, false_val: Expr) -> Expr:
    if cond_op not in ("lt", "le", "gt", "ge"):
        raise ValueError(f"unsupported condition op {cond_op!r}")
    if true_val is false_val:
        return true_val
    return _mk_where(cond_op, cond_lhs, cond_rhs, true_val, false_val)


def sym_max(a: Expr, b: Expr) -> Expr:
    return where("gt", a, b, a, b)


_DIFF_MEMO: dict[tuple[int, str], Expr] = {}


def diff(expr: Expr, var_name: str) -> Expr:
    """Exact symbolic partial derivative d(expr)/d(var_name) memoized in O(N) over DAG."""
    key = (expr._id, var_name)
    cached = _DIFF_MEMO.get(key)
    if cached is not None:
        return cached

    if isinstance(expr, Const):
        res: Expr = ZERO
    elif isinstance(expr, Var):
        res = ONE if expr.name == var_name else ZERO
    elif isinstance(expr, Unary):
        u = expr.arg
        du = diff(u, var_name)
        if _is_const(du, 0.0):
            res = ZERO
        elif expr.op == "neg":
            res = neg(du)
        elif expr.op == "sqrt":
            res = div(du, mul(TWO, expr))
        elif expr.op == "cbrt":
            res = div(du, mul(c(3.0), sq(expr)))
        elif expr.op == "log":
            res = div(du, u)
        elif expr.op == "log1p":
            res = div(du, add(ONE, u))
        elif expr.op == "exp":
            res = mul(expr, du)
        elif expr.op == "expm1":
            res = mul(exp(u), du)
        elif expr.op == "atan":
            res = div(du, add(ONE, sq(u)))
        elif expr.op == "asinh":
            res = div(du, sqrt(add(ONE, sq(u))))
        else:
            raise ValueError(f"unknown unary op {expr.op!r}")
    elif isinstance(expr, Binary):
        u, w = expr.lhs, expr.rhs
        du, dw = diff(u, var_name), diff(w, var_name)
        if _is_const(du, 0.0) and _is_const(dw, 0.0):
            res = ZERO
        elif expr.op == "add":
            res = add(du, dw)
        elif expr.op == "sub":
            res = sub(du, dw)
        elif expr.op == "mul":
            res = add(mul(du, w), mul(u, dw))
        elif expr.op == "div":
            if _is_const(dw, 0.0):
                res = div(du, w)
            else:
                res = div(sub(mul(du, w), mul(u, dw)), sq(w))
        elif expr.op == "pow":
            if isinstance(w, Const):
                k = w.value
                if k == 2.0:
                    res = mul(TWO, mul(u, du))
                elif k == 3.0:
                    res = mul(c(3.0), mul(sq(u), du))
                elif k == 4.0:
                    res = mul(c(4.0), mul(cube(u), du))
                else:
                    res = mul(mul(w, power(u, c(k - 1.0))), du)
            else:
                res = mul(expr, add(mul(dw, log(u)), div(mul(w, du), u)))
        else:
            raise ValueError(f"unknown binary op {expr.op!r}")
    elif isinstance(expr, Where):
        res = where(
            expr.cond_op,
            expr.cond_lhs,
            expr.cond_rhs,
            diff(expr.true_val, var_name),
            diff(expr.false_val, var_name),
        )
    else:
        raise TypeError(f"unsupported Expr type {type(expr)!r}")

    _DIFF_MEMO[key] = res
    return res


def eval_numeric(expr: Expr, env: dict[str, float]) -> float:
    """Evaluate an Expr DAG in double precision."""
    memo: dict[int, float] = {}

    def _ev(node: Expr) -> float:
        cached = memo.get(node._id)
        if cached is not None:
            return cached
        if isinstance(node, Const):
            val = node.value
        elif isinstance(node, Var):
            val = float(env[node.name])
        elif isinstance(node, Unary):
            a = _ev(node.arg)
            if node.op == "neg":
                val = -a
            elif node.op == "sqrt":
                val = math.sqrt(a)
            elif node.op == "cbrt":
                val = math.copysign(abs(a) ** (1.0 / 3.0), a)
            elif node.op == "log":
                val = math.log(a)
            elif node.op == "log1p":
                val = math.log1p(a)
            elif node.op == "exp":
                val = math.exp(a)
            elif node.op == "expm1":
                val = math.expm1(a)
            elif node.op == "atan":
                val = math.atan(a)
            elif node.op == "asinh":
                val = math.asinh(a)
            else:
                raise ValueError(node.op)
        elif isinstance(node, Binary):
            l, r = _ev(node.lhs), _ev(node.rhs)
            if node.op == "add":
                val = l + r
            elif node.op == "sub":
                val = l - r
            elif node.op == "mul":
                val = l * r
            elif node.op == "div":
                val = l / r
            elif node.op == "pow":
                val = l ** r
            else:
                raise ValueError(node.op)
        elif isinstance(node, Where):
            cl, cr = _ev(node.cond_lhs), _ev(node.cond_rhs)
            cond = (
                (cl < cr)
                if node.cond_op == "lt"
                else (cl <= cr)
                if node.cond_op == "le"
                else (cl > cr)
                if node.cond_op == "gt"
                else (cl >= cr)
            )
            val = _ev(node.true_val if cond else node.false_val)
        else:
            raise TypeError(type(node))
        memo[node._id] = val
        return val

    return _ev(expr)


def cse(outputs: Sequence[Expr]) -> tuple[list[tuple[str, Expr]], list[Expr]]:
    """Topological Common Subexpression Elimination over a multi-output DAG."""
    topo: list[Expr] = []
    visited: set[int] = set()
    refcount: dict[int, int] = {}

    def _children(node: Expr) -> tuple[Expr, ...]:
        if isinstance(node, (Const, Var)):
            return ()
        if isinstance(node, Unary):
            return (node.arg,)
        if isinstance(node, Binary):
            return (node.lhs, node.rhs)
        if isinstance(node, Where):
            return (node.cond_lhs, node.cond_rhs, node.true_val, node.false_val)
        raise TypeError(type(node))

    def _visit(node: Expr) -> None:
        nid = node._id
        refcount[nid] = refcount.get(nid, 0) + 1
        if nid in visited:
            return
        visited.add(nid)
        for ch in _children(node):
            _visit(ch)
        topo.append(node)

    for root in outputs:
        _visit(root)

    bindings: list[tuple[str, Expr]] = []
    for node in topo:
        if isinstance(node, (Const, Var)):
            continue
        if isinstance(node, Unary) and node.op == "neg":
            continue
        if refcount.get(node._id, 0) >= 2:
            bindings.append((f"t{len(bindings)}", node))

    return bindings, list(outputs)
