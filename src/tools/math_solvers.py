"""Type-specific DETERMINISTIC maths solvers -- the "calculator pointed at the right question" idea.

The lesson from the reverted general calculator (Q6767): letting the MODEL'S (often wrong) set-up drive
an override is fatal. So here WE own the solving logic per question-TYPE -- correct by construction -- and
each solver is silent (returns None) on anything outside its type. A solver fires ONLY when it (a) detects
its type with high precision, (b) parses the inputs cleanly, and (c) maps its result to EXACTLY ONE option.
Any miss -> None -> the LLM handles the question untouched. So the suite can only ADD correct answers on the
computational subset; it can never override the LLM on the knowledge/abstract subset (those types are simply
not claimed). See [[maths-live-routing-stack]].

`solve_maths(question) -> (letter, evidence) | None` is the single entry point.
"""
from __future__ import annotations

import ast
import math
import re
from fractions import Fraction

from schemas import Question


# ---------------------------------------------------------------------------
# Expression cleanup + Fraction-exact safe eval (no eval(), AST-walk only).
# ---------------------------------------------------------------------------
def _clean_expr(s: str) -> str:
    """LaTeX / formatting -> plain Python arithmetic. Best-effort, lossless for the forms we solve."""
    s = s.strip()
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("×", "*")
    s = s.replace("\\div", "/").replace("÷", "/")
    # \frac{a}{b} / \dfrac / \cfrac -> (a)/(b); loop a few times for limited nesting.
    for _ in range(4):
        new = re.sub(r"\\[dct]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
        if new == s:
            break
        s = new
    s = s.replace("^", "**")
    s = re.sub(r"\{(\d+)\}", r"\1", s)   # bare ^{n} braces left after ^-> **
    s = s.replace("{", "(").replace("}", ")")
    return s


_ALLOWED = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b, ast.FloorDiv: lambda a, b: a // b,
}


def _eval_node(node) -> Fraction:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-number")
        return Fraction(str(node.value))
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            base = _eval_node(node.left)
            exp = _eval_node(node.right)
            if exp.denominator != 1:
                raise ValueError("non-integer power")
            return base ** int(exp)
        op = _ALLOWED.get(type(node.op))
        if op is None:
            raise ValueError("op")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ValueError("unary")
    raise ValueError("forbidden")


def _safe_eval(expr: str) -> Fraction | None:
    """A purely-numeric arithmetic expression -> exact Fraction, or None if it isn't one."""
    expr = _clean_expr(expr)
    if not expr or re.search(r"[A-Za-z\\]", expr):   # any letter/backslash left -> not pure numeric.
        return None
    try:
        return _eval_node(ast.parse(expr, mode="eval").body)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Option-matching helpers -- the gate that makes a solver safe (exactly-one match or abstain).
# ---------------------------------------------------------------------------
def _option_value(text: str) -> Fraction | None:
    """An option's text -> its numeric value (handles fractions, %, commas, 2^11-1), or None."""
    t = text.strip()
    t = re.sub(r"\b(inches|inch|points?|dollars?|degrees?|cm|m|kg|%)\b", "", t, flags=re.I)
    t = t.replace(",", "").replace("$", "").strip()
    return _safe_eval(t)


def _match_value(options: dict[str, str], target: Fraction, *, tol: Fraction | None = None) -> str | None:
    """The UNIQUE option letter whose numeric value equals target -- else None (0 or >1 matches)."""
    hits = []
    for k, v in options.items():
        ov = _option_value(v)
        if ov is None:
            continue
        if ov == target or (tol is not None and abs(ov - target) <= tol):
            hits.append(k)
    return hits[0] if len(hits) == 1 else None


def _int_set(text: str) -> set[int] | None:
    """An option like '0,4' or '0, 1, 2' -> {0,4} / {0,1,2}; None if it isn't a clean int list."""
    t = re.sub(r"[^\d,\-\s]", "", text.strip())
    parts = [p.strip() for p in re.split(r"[,\s]+", t) if p.strip() not in ("", "-")]
    try:
        return {int(p) for p in parts}
    except ValueError:
        return None


def _match_set(options: dict[str, str], target: set) -> str | None:
    hits = [k for k, v in options.items() if _int_set(v) == target]
    return hits[0] if len(hits) == 1 else None


def _moduli(text: str) -> list[int]:
    """Every Z_n subscript in the text -> [n, ...] (Z_4 x Z_12 -> [4, 12])."""
    return [int(m) for m in re.findall(r"\bZ_?\{?(\d+)\}?", text)]


# ---------------------------------------------------------------------------
# The solvers. Each returns (letter, evidence) or None (abstain).
# ---------------------------------------------------------------------------
def _solve_finite_field_roots(q: Question) -> tuple[str, str] | None:
    """'Find all zeros of <poly> in Z_n' -> evaluate the poly at every element of Z_n."""
    t = q.text
    if not re.search(r"\b(zeros?|roots?)\b", t, re.I):
        return None
    mods = _moduli(t)
    if not mods:
        return None
    n = mods[0]
    if not (2 <= n <= 97):
        return None
    # The polynomial: the run of x-terms (grab the longest such substring).
    cands = re.findall(r"[0-9x\^\+\-\*\s\{\}]*x[0-9x\^\+\-\*\s\{\}]*", _clean_expr(t).replace("**", "^"))
    coeffs = None
    for c in sorted(cands, key=len, reverse=True):
        coeffs = _parse_poly(c)
        if coeffs:
            break
    if not coeffs:
        return None
    roots = {c for c in range(n) if sum(co * (c ** p) for p, co in coeffs.items()) % n == 0}
    letter = _match_set(q.options, roots)
    if letter:
        return letter, f"finite-field roots in Z_{n}: {sorted(roots)}"
    return None


def _parse_poly(s: str) -> dict[int, int] | None:
    """A polynomial-in-x string -> {power: int coeff}, or None if any term won't parse."""
    s = _clean_expr(s).replace(" ", "")
    s = re.sub(r"in.*$", "", s)
    if "x" not in s:
        return None
    s = s.replace("-", "+-")
    coeffs: dict[int, int] = {}
    for term in (p for p in s.split("+") if p):
        m = re.fullmatch(r"(-?\d*)\*?x(?:\*\*(\d+))?", term)
        if m:
            c = m.group(1)
            c = -1 if c == "-" else (1 if c == "" else int(c))
            p = int(m.group(2)) if m.group(2) else 1
        elif re.fullmatch(r"-?\d+", term):
            c, p = int(term), 0
        else:
            return None      # an unparseable term -> abstain on the whole question.
        coeffs[p] = coeffs.get(p, 0) + c
    return coeffs or None


def _solve_ring_characteristic(q: Question) -> tuple[str, str] | None:
    """'characteristic of the ring Z_m x Z_n' -> lcm(m, n) (single Z_n -> n)."""
    if "characteristic" not in q.text.lower():
        return None
    mods = _moduli(q.text)
    if not mods:
        return None
    char = mods[0]
    for m in mods[1:]:
        char = char * m // math.gcd(char, m)
    letter = _match_value(q.options, Fraction(char))
    if letter:
        return letter, f"ring characteristic = lcm{tuple(mods)} = {char}"
    return None


def _solve_gcd(q: Question) -> tuple[str, str] | None:
    """'gcd of m^a-1 and m^b-1' = m^gcd(a,b)-1; or gcd of two plain integers."""
    if not re.search(r"greatest common divisor|\bgcd\b", q.text, re.I):
        return None
    powers = re.findall(r"(\d+)\s*\^\s*\{?(\d+)\}?\s*-\s*1", q.text)
    if len(powers) >= 2 and powers[0][0] == powers[1][0]:
        base = int(powers[0][0])
        a, b = int(powers[0][1]), int(powers[1][1])
        val = Fraction(base ** math.gcd(a, b) - 1)
        # options may be numeric OR symbolic ('2^11 - 1'); _option_value handles both via _safe_eval.
        letter = _match_value(q.options, val)
        if letter:
            return letter, f"gcd({base}^{a}-1, {base}^{b}-1) = {base}^{math.gcd(a,b)}-1 = {val}"
        return None
    ints = [int(x.replace(",", "")) for x in re.findall(r"\bof\s+([\d,]+)\s+and\s+([\d,]+)", q.text)[0]] \
        if re.search(r"\bof\s+[\d,]+\s+and\s+[\d,]+", q.text) else []
    if len(ints) == 2:
        letter = _match_value(q.options, Fraction(math.gcd(ints[0], ints[1])))
        if letter:
            return letter, f"gcd{tuple(ints)} = {math.gcd(*ints)}"
    return None


def _solve_sum_product(q: Question) -> tuple[str, str] | None:
    """'Two numbers ... sum is S ... product is P' -> solve t^2 - S t + P = 0."""
    t = q.text
    if not re.search(r"\btwo numbers\b", t, re.I):
        return None
    s_m = re.search(r"(?:added together|sum|add up)[^\d]*(\d+)", t, re.I)
    p_m = re.search(r"product[^\d]*(\d+)", t, re.I)
    if not (s_m and p_m):
        return None
    S, P = int(s_m.group(1)), int(p_m.group(1))
    disc = S * S - 4 * P
    if disc < 0:
        return None
    r = math.isqrt(disc)
    if r * r != disc or (S + r) % 2:
        return None
    pair = {(S + r) // 2, (S - r) // 2}
    letter = _match_set(q.options, pair)
    if letter:
        return letter, f"two numbers with sum {S}, product {P}: {sorted(pair)}"
    return None


def _solve_reflection_yx(q: Question) -> tuple[str, str] | None:
    """'<polygon> reflected across y = x' -> reflect every (a,b) to (b,a); match an option point."""
    t = q.text
    if not (re.search(r"reflect", t, re.I) and re.search(r"y\s*=\s*x", t)):
        return None
    pt = r"\(\s*(-?\s*\d+)\s*,\s*(-?\s*\d+)\s*\)"   # allow a space after the minus: '(- 2, - 4)'.
    def _i(s): return int(s.replace(" ", ""))
    verts = re.findall(pt, t)
    if len(verts) < 2:
        return None
    reflected = {(_i(b), _i(a)) for a, b in verts}
    hits = []
    for k, v in q.options.items():
        m = re.search(pt, v)
        if m and (_i(m.group(1)), _i(m.group(2))) in reflected:
            hits.append(k)
    if len(hits) == 1:
        return hits[0], f"reflection across y=x; P' contains {sorted(reflected)}"
    return None


def _solve_triangle_sides(q: Question) -> tuple[str, str] | None:
    """'Which could NOT be the sides of an (isosceles) triangle?' -> test each option triple."""
    t = q.text
    if not re.search(r"sides?\b.*triangle", t, re.I):
        return None
    negated = bool(re.search(r"\bnot\b|cannot|could\s+not", t, re.I))
    iso = "isosceles" in t.lower()

    def ok(sides: list[int]) -> bool:
        a, b, c = sorted(sides)
        valid = a + b > c          # strict triangle inequality (degenerate -> not a triangle).
        if iso:
            return valid and len({a, b, c}) <= 2
        return valid

    parsed = {}
    for k, v in q.options.items():
        nums = re.findall(r"\d+(?:\.\d+)?", v)
        if len(nums) != 3:
            return None            # an option that isn't a clean triple -> abstain.
        parsed[k] = [float(x) for x in nums]
    failing = [k for k, s in parsed.items() if not ok(s)]
    passing = [k for k, s in parsed.items() if ok(s)]
    want = failing if negated else passing
    if len(want) == 1:
        return want[0], f"triangle-inequality{'/isosceles' if iso else ''} test -> option {want[0]}"
    return None


def _solve_percentage_increase(q: Question) -> tuple[str, str] | None:
    """'<N> ... increases by P% ... (to the nearest whole)' -> N*(1+P/100), rounded if asked."""
    t = q.text
    m = re.search(r"increase[sd]?\s+by\s+(\d+(?:\.\d+)?)\s*%", t, re.I)
    if not m:
        return None
    p = Fraction(m.group(1))
    # the base = the number immediately before 'increase' (nearest preceding standalone number).
    pre = t[: m.start()]
    nums = re.findall(r"(\d+(?:\.\d+)?)", pre)
    if not nums:
        return None
    base = Fraction(nums[-1])
    val = base * (1 + p / 100)
    if re.search(r"nearest\s+whole", t, re.I):
        val = Fraction(round(val))
    letter = _match_value(q.options, val)
    if letter:
        return letter, f"{base} increased by {p}% = {float(val):g}"
    return None


def _solve_arith_expression(q: Question) -> tuple[str, str] | None:
    """A self-contained numeric expression (after Find/Simplify/Evaluate/compute) -> its exact value."""
    t = q.text
    m = re.search(r"\b(?:find|simplify|evaluate|compute|value of)\b(.+)", t, re.I | re.S)
    if not m:
        return None
    expr = m.group(1).strip().rstrip(".?")
    val = _safe_eval(expr)
    if val is None:
        return None
    letter = _match_value(q.options, val)
    if letter:
        return letter, f"evaluated expression = {val}"
    return None


_SOLVERS = (
    _solve_finite_field_roots,
    _solve_ring_characteristic,
    _solve_gcd,
    _solve_sum_product,
    _solve_reflection_yx,
    _solve_triangle_sides,
    _solve_percentage_increase,
    _solve_arith_expression,
)


_DASHES = {ord(c): "-" for c in "‐‑‒–—―−"}  # ‐‑‒–—―−  -> ASCII '-'.


def _norm(s: str) -> str:
    """Unicode dashes/minus -> ASCII hyphen (the live data writes '(– 2, – 4)', not '-2'); NBSP -> space."""
    return (s or "").translate(_DASHES).replace(" ", " ")


def solve_maths(question: Question) -> tuple[str, str] | None:
    """Try every type-specific solver; the first confident, single-option hit wins. None = defer to LLM.

    Crash-safe: any solver that raises is skipped (it just abstains). MCQ-only -- no options, no solve.
    """
    if not question.options:
        return None
    # Normalise unicode dashes/minus in BOTH the stem and the options before any parser sees them.
    q = Question(
        qid=question.qid,
        text=_norm(question.text),
        options={k: _norm(v) for k, v in question.options.items()},
        qtype=question.qtype,
    )
    for fn in _SOLVERS:
        try:
            res = fn(q)
        except Exception:
            res = None
        if res is not None:
            return res
    return None
