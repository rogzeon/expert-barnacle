from collections.abc import Callable

import torch
from sympy import (
    Add,
    Derivative,
    Mul,
    Number,
    Pow,
    Symbol,
    symbols,
)


def as_tf_expr(expr, coords: dict) -> Callable:
    rec = lambda e: as_tf_expr(e, coords)
    match expr:
        case Add(args=terms):
            fns = [rec(term) for term in terms]
            return lambda u, fns=fns: sum(f(u) for f in fns)

        case Mul(args=terms):
            fns = [rec(term) for term in terms]

            def _mul(u, fns=fns):
                result = fns[0](u)
                for f in fns[1:]:
                    result = result * f(u)
                return result

            return _mul

        case Pow(args=(a, b)):
            fa, fb = rec(a), rec(b)
            return lambda u, fa=fa, fb=fb: fa(u) ** fb(u)

        case Symbol():
            if expr in coords:
                return lambda u, expr=expr: coords[expr]
            return lambda u: u

        case Number():
            val = float(expr)
            return lambda u, val=val: val

        case Derivative():
            f = rec(expr.expr)
            for var in expr.variables:
                if var not in coords:
                    raise KeyError(f"Coordinate '{var}' not found in coords dict.")
                prev_f, wrt = f, coords[var]

                def _diff(u, prev_f=prev_f, wrt=wrt) -> tuple[torch.Tensor]:
                    y = prev_f(u)
                    (grad,) = torch.autograd.grad(
                        y,
                        wrt,
                        grad_outputs=torch.ones_like(y),
                        create_graph=True,
                        retain_graph=True,
                    )
                    return grad

                f = _diff
            return f

        case _:
            raise TypeError(
                f"Unsupported expression node: {expr!r} ({type(expr).__name__})"
            )


if __name__ == "__main__":
    as_tf_expr(symbols("x"))
