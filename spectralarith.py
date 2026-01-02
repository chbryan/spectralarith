#!/usr/bin/env python3
"""
spectralarith.py

A small, research-oriented framework for building and analyzing
"trigonomatrices" of the form:

  T[j,k] = cos(2π * s_j*s_k / n_real) + i * sin(2π * s_j*s_k / n_imag)

with common choices:
  n_real = n
  n_imag = primorial(n) = product of first n primes

Includes:
  - fast nth-prime + primorial
  - vectorized matrix construction (float-safe path)
  - optional high-precision fallback for huge primorials (mpmath)
  - basic spectrum + spacing + spectral-gap utilities
  - simple CLI

Dependencies: numpy (required), mpmath (optional but recommended).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

try:
    import mpmath as mp  # optional fallback for huge denominators
except Exception:  # pragma: no cover
    mp = None


# ----------------------------
# Number theory utilities
# ----------------------------

def _nth_prime_upper_bound(n: int) -> int:
    """Upper bound for nth prime (Dusart-style)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n < 6:
        return 15
    nn = float(n)
    return int(nn * (math.log(nn) + math.log(math.log(nn))) * 1.3) + 10


def first_n_primes(n: int) -> np.ndarray:
    """Return first n primes as a numpy array of dtype=int64 where possible."""
    if n < 1:
        raise ValueError("n must be >= 1")

    limit = _nth_prime_upper_bound(n)
    while True:
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[:2] = False
        # basic sieve
        r = int(limit**0.5)
        for p in range(2, r + 1):
            if sieve[p]:
                sieve[p * p : limit + 1 : p] = False
        primes = np.flatnonzero(sieve)
        if primes.size >= n:
            return primes[:n].astype(np.int64, copy=False)
        limit *= 2  # retry with larger bound


def primorial(n: int) -> int:
    """Primorial of n: product of first n primes."""
    ps = first_n_primes(n)
    prod = 1
    for p in ps.tolist():
        prod *= int(p)
    return prod


def von_mangoldt(k: int) -> float:
    """
    Von Mangoldt Λ(k): log p if k = p^m (m>=1), else 0.
    Simple factor test via trial division (suitable for moderate k).
    """
    if k <= 0:
        return 0.0
    if k == 1:
        return 0.0

    x = k
    p = 2
    while p * p <= x:
        if x % p == 0:
            # count power
            m = 0
            while x % p == 0:
                x //= p
                m += 1
            # if remaining factor is 1, then k was p^m
            return math.log(p) if x == 1 else 0.0
        p = 3 if p == 2 else p + 2

    # x is prime now
    return math.log(x)  # k itself was prime -> p^1


def cyclotomic_character(n: int, modulus: int) -> complex:
    """
    A concrete "cyclotomic character" placeholder:
      χ(n) = exp(2π i * n / modulus)

    This is a simple, explicit phase function that is always defined.
    """
    if modulus <= 0:
        raise ValueError("modulus must be > 0")
    theta = math.tau * ((n % modulus) / modulus)
    return complex(math.cos(theta), math.sin(theta))


# ----------------------------
# Sequence generators
# ----------------------------

def seq_primes(n: int) -> np.ndarray:
    return first_n_primes(n)


def seq_prime_powers(n: int, power: int = 2) -> np.ndarray:
    ps = first_n_primes(n)
    return (ps.astype(object) ** power).astype(object)


def seq_integers(n: int, start: int = 1) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.int64)


# ----------------------------
# Kernel + matrix construction
# ----------------------------

DenomMode = Literal["n", "primorial", "custom"]


@dataclass(frozen=True)
class TrigonomatrixParams:
    n: int
    denom_real: DenomMode = "n"
    denom_imag: DenomMode = "primorial"
    custom_real: Optional[int] = None
    custom_imag: Optional[int] = None
    reduce_mod_imag: bool = True  # reduce term mod denom_imag to keep angles bounded
    high_precision: bool = False  # force mpmath path if available
    mp_dps: int = 50              # decimal digits for mpmath, if used


def _resolve_denoms(params: TrigonomatrixParams) -> Tuple[int, int]:
    n = params.n
    if n <= 0:
        raise ValueError("params.n must be > 0")

    def resolve(mode: DenomMode, custom: Optional[int]) -> int:
        if mode == "n":
            return n
        if mode == "primorial":
            return primorial(n)
        if mode == "custom":
            if custom is None or custom <= 0:
                raise ValueError("custom denominator must be provided and > 0")
            return int(custom)
        raise ValueError(f"unknown mode: {mode}")

    return resolve(params.denom_real, params.custom_real), resolve(params.denom_imag, params.custom_imag)


def canonical_trigonomatrix(sequence: Sequence[int] | np.ndarray, params: TrigonomatrixParams) -> np.ndarray:
    """
    Build T (n x n) for a provided length-n sequence.

    Vectorized float-safe path:
      - uses numpy outer product
      - uses float angles
      - safe when denominators fit well in float and are not astronomically large

    High-precision fallback (slow):
      - uses mpmath for angle computations if primorial/custom denom is huge
    """
    n = params.n
    if len(sequence) != n:
        raise ValueError(f"sequence length {len(sequence)} != params.n {n}")

    denom_real, denom_imag = _resolve_denoms(params)

    # Decide whether to use high precision
    use_mp = bool(params.high_precision) or (denom_imag > 10**300) or (denom_real > 10**300)
    if use_mp and mp is None:
        raise RuntimeError("mpmath not available but high-precision path was required/requested")

    if not use_mp:
        s = np.asarray(sequence)
        # Use object dtype if large integers appear
        if s.dtype.kind not in ("i", "u"):
            s = s.astype(object)

        prod = np.multiply.outer(s, s)  # may be object dtype for large ints
        # angle arguments: reduce mod denom to keep values bounded and avoid float overflow
        if params.reduce_mod_imag:
            prod_im = np.remainder(prod, denom_imag)
        else:
            prod_im = prod

        # Convert to float angles. We only use prod_mod/denom which is in [0,1),
        # so this remains well-scaled even for large denom, as long as denom fits as float.
        denom_real_f = float(denom_real)
        denom_imag_f = float(denom_imag)

        angles_re = (math.tau / denom_real_f) * (np.remainder(prod, denom_real) if denom_real != 0 else prod)
        angles_im = (math.tau / denom_imag_f) * prod_im

        re = np.cos(angles_re, dtype=float)
        im = np.sin(angles_im, dtype=float)
        return re.astype(np.complex128) + 1j * im.astype(np.complex128)

    # High-precision fallback (slow; intended for small n)
    mp.mp.dps = int(params.mp_dps)
    T = np.empty((n, n), dtype=np.complex128)

    s_list = [int(x) for x in sequence]
    denom_real_mp = mp.mpf(denom_real)
    denom_imag_mp = mp.mpf(denom_imag)

    for j in range(n):
        sj = s_list[j]
        for k in range(n):
            term = sj * s_list[k]

            # keep angles in [0, 2π) via modular reduction
            tr = term % denom_real if denom_real != 0 else term
            ti = (term % denom_imag) if params.reduce_mod_imag else term

            ang_re = mp.tau * (mp.mpf(tr) / denom_real_mp)
            ang_im = mp.tau * (mp.mpf(ti) / denom_imag_mp)

            val = mp.cos(ang_re) + (1j * mp.sin(ang_im))
            T[j, k] = complex(val.real, val.imag)

    return T


# ----------------------------
# Harmonic determinant (phase-adjusted det)
# ----------------------------

def harmonic_determinant(T: np.ndarray, thetas: Optional[Sequence[float]] = None) -> complex:
    """
    Phase-adjusted determinant:
      det( D(theta) @ T )
    where D(theta) is diagonal with entries exp(-i*theta_j).

    If thetas is None, returns det(T).
    """
    T = np.asarray(T, dtype=np.complex128)
    n = T.shape[0]
    if T.shape[0] != T.shape[1]:
        raise ValueError("T must be square")
    if thetas is None:
        return complex(np.linalg.det(T))
    if len(thetas) != n:
        raise ValueError("len(thetas) must equal matrix dimension")

    phases = np.exp(-1j * np.asarray(thetas, dtype=float))
    # left-multiply each row by phase (same as D @ T)
    Tp = (phases[:, None] * T)
    return complex(np.linalg.det(Tp))


def logdet(T: np.ndarray) -> Tuple[complex, float]:
    """
    Numerically stable log-determinant via LU-ish sign+logabs for complex:
    returns (sign_complex, log_abs_det) such that det ≈ sign * exp(log_abs_det).
    """
    T = np.asarray(T, dtype=np.complex128)
    det = np.linalg.det(T)
    if det == 0:
        return 0j, float("-inf")
    sign = det / abs(det)
    return complex(sign), float(math.log(abs(det)))


# ----------------------------
# Spectral analysis
# ----------------------------

def eigvals(T: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(np.asarray(T, dtype=np.complex128))


def spectral_gap(evals: np.ndarray, mode: Literal["abs", "real"] = "abs") -> float:
    """
    Simple "gap" between the top-2 eigenvalues under a chosen ordering.

    mode="abs": sort by |λ|
    mode="real": sort by Re(λ)
    """
    ev = np.asarray(evals, dtype=np.complex128)
    if ev.size < 2:
        return float("nan")

    if mode == "abs":
        order = np.argsort(np.abs(ev))[::-1]
        a, b = np.abs(ev[order[0]]), np.abs(ev[order[1]])
    elif mode == "real":
        order = np.argsort(ev.real)[::-1]
        a, b = float(ev[order[0]].real), float(ev[order[1]].real)
    else:
        raise ValueError("mode must be 'abs' or 'real'")

    return float(a - b)


def nearest_neighbor_spacings(
    evals: np.ndarray,
    unwrap: Literal["real", "angle", "abs"] = "real"
) -> np.ndarray:
    """
    Nearest-neighbor spacings after sorting a 1D projection of eigenvalues:
      unwrap="real": sort by Re(λ)
      unwrap="abs": sort by |λ|
      unwrap="angle": sort by arg(λ) in [0,2π)
    Returns consecutive differences.
    """
    ev = np.asarray(evals, dtype=np.complex128)
    if ev.size < 2:
        return np.array([], dtype=float)

    if unwrap == "real":
        x = np.sort(ev.real.astype(float))
    elif unwrap == "abs":
        x = np.sort(np.abs(ev).astype(float))
    elif unwrap == "angle":
        x = np.sort(np.mod(np.angle(ev).astype(float), math.tau))
    else:
        raise ValueError("unwrap must be 'real', 'abs', or 'angle'")

    return np.diff(x)


# ----------------------------
# CLI
# ----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and analyze trigonomatrices.")
    p.add_argument("--n", type=int, default=64, help="matrix dimension")
    p.add_argument("--sequence", type=str, default="primes",
                   choices=["primes", "integers", "prime_squares", "prime_cubes"],
                   help="sequence generator")
    p.add_argument("--denom-real", type=str, default="n", choices=["n", "primorial", "custom"])
    p.add_argument("--denom-imag", type=str, default="primorial", choices=["n", "primorial", "custom"])
    p.add_argument("--custom-real", type=int, default=None)
    p.add_argument("--custom-imag", type=int, default=None)
    p.add_argument("--no-reduce-mod-imag", action="store_true", help="do not reduce term mod denom-imag")
    p.add_argument("--high-precision", action="store_true", help="force mpmath path (slow)")
    p.add_argument("--mp-dps", type=int, default=50, help="mpmath decimal digits")
    p.add_argument("--gap-mode", type=str, default="abs", choices=["abs", "real"])
    p.add_argument("--spacing", type=str, default="real", choices=["real", "abs", "angle"])
    return p.parse_args()


def _make_sequence(n: int, kind: str) -> np.ndarray:
    if kind == "primes":
        return seq_primes(n)
    if kind == "integers":
        return seq_integers(n, start=1)
    if kind == "prime_squares":
        return seq_prime_powers(n, power=2)
    if kind == "prime_cubes":
        return seq_prime_powers(n, power=3)
    raise ValueError(f"unknown sequence kind: {kind}")


def main() -> None:
    args = _parse_args()
    n = int(args.n)

    S = _make_sequence(n, args.sequence)

    params = TrigonomatrixParams(
        n=n,
        denom_real=args.denom_real,  # type: ignore[arg-type]
        denom_imag=args.denom_imag,  # type: ignore[arg-type]
        custom_real=args.custom_real,
        custom_imag=args.custom_imag,
        reduce_mod_imag=not args.no_reduce_mod_imag,
        high_precision=bool(args.high_precision),
        mp_dps=int(args.mp_dps),
    )

    T = canonical_trigonomatrix(S, params)
    ev = eigvals(T)

    gap = spectral_gap(ev, mode=args.gap_mode)  # type: ignore[arg-type]
    spacings = nearest_neighbor_spacings(ev, unwrap=args.spacing)  # type: ignore[arg-type]

    detT = complex(np.linalg.det(T))
    sgn, lad = logdet(T)

    print(f"n={n} sequence={args.sequence}")
    dr, di = _resolve_denoms(params)
    print(f"denom_real={dr}")
    print(f"denom_imag={di}")
    print(f"det(T)={detT}")
    print(f"log|det(T)|={lad:.6f}  sign(det)≈{sgn}")
    print(f"spectral_gap({args.gap_mode})={gap:.6g}")
    if spacings.size:
        print(f"spacings({args.spacing}): mean={spacings.mean():.6g} std={spacings.std(ddof=1) if spacings.size>1 else 0.0:.6g}")
    else:
        print("spacings: n/a")


if __name__ == "__main__":
    main()
