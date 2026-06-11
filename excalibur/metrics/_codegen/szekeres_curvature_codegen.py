#!/usr/bin/env python3
r"""
Symbolic code generator for the quasi-spherical Szekeres curvature.

Reproduces (and corrects) Celerier (2024) Appendix A by computing the Ricci
tensor and the screen tidal tensor ``T_{mu nu} = R_{mu alpha nu beta} k^alpha
k^beta`` directly from the diagonal metric in the ``H, F`` notation (eq. 74),

    ds^2 = -dt^2 + H^2 dr^2 + F^2 (dp^2 + dq^2)   (geometric units),

with ``H(t, r, p, q)`` and ``F(t, r, p, q)`` treated as free functions.  Run
this script to regenerate the Python expressions pasted into
``excalibur/observables/riemann_szekeres.py``::

    python -m excalibur.metrics._codegen.szekeres_curvature_codegen

It requires ``sympy`` (a *development*-time dependency only; the generated code
is plain NumPy arithmetic with no runtime sympy import).
"""
import sympy as sp

t, r, p, q = sp.symbols("t r p q")
X = [t, r, p, q]
NAMES = ["t", "r", "p", "q"]

H = sp.Function("H")(t, r, p, q)
F = sp.Function("F")(t, r, p, q)
kt, kr, kp, kq = sp.symbols("kt kr kp kq")   # geometric-basis photon vector
KVEC = [kt, kr, kp, kq]


def _christoffels(g):
    gi = g.inv()
    n = 4
    Gam = [[[sp.S(0)] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                s = sum(gi[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b])
                                    - sp.diff(g[b, c], X[d])) for d in range(n))
                Gam[a][b][c] = sp.simplify(s / 2)
    return Gam


def _riemann_lower(g, Gam):
    n = 4
    # R^rho_{sigma mu nu}
    Rup = [[[[sp.S(0)] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for rho in range(n):
        for sig in range(n):
            for mu in range(n):
                for nu in range(n):
                    term = sp.diff(Gam[rho][nu][sig], X[mu]) - sp.diff(Gam[rho][mu][sig], X[nu])
                    term += sum(Gam[rho][mu][l] * Gam[l][nu][sig]
                                - Gam[rho][nu][l] * Gam[l][mu][sig] for l in range(n))
                    Rup[rho][sig][mu][nu] = term
    # lower first index
    Rlow = [[[[sp.S(0)] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    Rlow[a][b][c][d] = sp.simplify(sum(g[a, l] * Rup[l][b][c][d] for l in range(n)))
    return Rlow


def _named_subs():
    subs = {H: sp.Symbol("H"), F: sp.Symbol("F")}
    for i, ni in enumerate(NAMES):
        subs[sp.diff(H, X[i])] = sp.Symbol("H" + ni)
        subs[sp.diff(F, X[i])] = sp.Symbol("F" + ni)
        for j, nj in enumerate(NAMES):
            if j >= i:
                subs[sp.diff(H, X[i], X[j])] = sp.Symbol("H" + ni + nj)
                subs[sp.diff(F, X[i], X[j])] = sp.Symbol("F" + ni + nj)
    return subs


def main():
    g = sp.diag(-1, H ** 2, F ** 2, F ** 2)
    Gam = _christoffels(g)
    Rlow = _riemann_lower(g, Gam)
    subs = _named_subs()

    def named(expr):
        return sp.simplify(sp.simplify(expr).subs(subs))

    # --- Ricci ---
    print("# ---- Ricci R[a,b] (geometric basis) ----")
    for a in range(4):
        for b in range(a, 4):
            Rab = sum(Rlow[m][a][m][b] for m in range(4))  # R_{ab}=R^m_{a m b}=g^{mm}R_{m a m b}
            # contract using inverse metric
            Rab = sum(g.inv()[m, n] * Rlow[m][a][n][b] for m in range(4) for n in range(4))
            print(f"R[{a},{b}] = {sp.ccode(named(Rab))}")

    # --- Screen tidal tensor T_{mu nu} = R_{mu a nu b} k^a k^b ---
    print("\n# ---- Tidal tensor T[mu,nu] = R_{mu a nu b} k^a k^b ----")
    for mu in range(4):
        for nu in range(mu, 4):
            T = sum(Rlow[mu][a][nu][b] * KVEC[a] * KVEC[b] for a in range(4) for b in range(4))
            print(f"T[{mu},{nu}] = {sp.ccode(named(T))}")


if __name__ == "__main__":
    main()
