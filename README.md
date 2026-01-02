# SpectralArith (Trigonomatrix Research Kit)

A small, research-oriented Python toolkit for constructing and analyzing complex **trigonomatrices**: matrices whose entries are trigonometric phases coupled to arithmetic sequences (e.g., primes).

This README documents how to install and use the project. 

---

## Concept

Given a length-`n` integer sequence

\[
S = (s_0, s_1, \dots, s_{n-1})
\]

define a complex matrix \(T \in \mathbb{C}^{n \times n}\) by

\[
T_{j,k}
=
\cos\!\left(2\pi \frac{s_js_k}{d_{\Re}}\right)
+
i\sin\!\left(2\pi \frac{s_js_k}{d_{\Im}}\right)
\]

Common denominator choices:
- \(d_{\Re} = n\)
- \(d_{\Im} = \prod_{m=1}^{n} p_m\) (the **primorial** of `n`)

The goal is to explore spectral structure (eigenvalues, gaps, spacing statistics) induced by number-theoretic sequences.

---

## Requirements

- Python 3.10+
- `numpy`
- Optional: `mpmath` (for high-precision mode when denominators are huge)

Install dependencies:
```bash
pip install numpy
pip install mpmath  # optional
```

---

## Quick start

Run the CLI (expects `spectralarith.py` to be present in the repo root):

```bash
python spectralarith.py --n 64 --sequence primes
```

The output prints:
- denominators used
- determinant and log-determinant magnitude
- a simple spectral gap metric
- basic spacing statistics

---

## CLI reference

### Core arguments

- `--n N`  
  Matrix dimension.

- `--sequence {primes,integers,prime_squares,prime_cubes}`  
  Sequence generator for \(S\).

- `--denom-real {n,primorial,custom}`  
  Denominator used in the cosine term.

- `--denom-imag {n,primorial,custom}`  
  Denominator used in the sine term.

- `--custom-real X` / `--custom-imag Y`  
  Required if the corresponding denom mode is `custom`.

- `--gap-mode {abs,real}`  
  How the “top two” eigenvalues are ranked for the gap:
  - `abs`: by magnitude \(|\lambda|\)
  - `real`: by real part \(\Re(\lambda)\)

- `--spacing {real,abs,angle}`  
  1D projection used for nearest-neighbor spacing:
  - `real`: \(\Re(\lambda)\)
  - `abs`: \(|\lambda|\)
  - `angle`: \(\arg(\lambda)\) in \([0,2\pi)\)

### Numerics / precision

- `--no-reduce-mod-imag`  
  Disables modular reduction of the imaginary argument. This can be numerically risky for very large denominators.

- `--high-precision`  
  Forces `mpmath` path (slow). Recommended only for small `n`.

- `--mp-dps K`  
  Decimal digits for `mpmath` (default: 50).

---

## Examples

### Default (real denom = n, imag denom = primorial(n))

```bash
python spectralarith.py --n 128 --sequence primes
```

### Use integer sequence

```bash
python spectralarith.py --n 128 --sequence integers
```

### Use n for both denominators

```bash
python spectralarith.py --n 128 --sequence primes --denom-imag n
```

### Custom denominators

```bash
python spectralarith.py --n 128 --sequence primes   --denom-real custom --custom-real 9973   --denom-imag custom --custom-imag 1000003
```

### High precision (slow)

```bash
python spectralarith.py --n 32 --sequence primes --high-precision --mp-dps 80
```

---

## Python API (import usage)

If you want to use it as a module, ensure `spectralarith.py` is on your `PYTHONPATH` or placed next to your script:

```python
import numpy as np
from spectralarith import (
    seq_primes,
    TrigonomatrixParams,
    canonical_trigonomatrix,
    eigvals,
    spectral_gap,
    nearest_neighbor_spacings,
    harmonic_determinant,
)

n = 64
S = seq_primes(n)

params = TrigonomatrixParams(
    n=n,
    denom_real="n",
    denom_imag="primorial",
    reduce_mod_imag=True,
)

T = canonical_trigonomatrix(S, params)
ev = eigvals(T)

print("gap(abs) =", spectral_gap(ev, mode="abs"))
print("mean spacing(real) =", nearest_neighbor_spacings(ev, unwrap="real").mean())

thetas = np.linspace(0, 2*np.pi, n, endpoint=False)
print("harmonic det =", harmonic_determinant(T, thetas=thetas))
```

---

## Notes on numerics

- The primorial grows extremely fast. Even moderate `n` can produce denominators too large for comfortable floating computation.
- By default, the sine-term argument is reduced modulo `denom_imag` to keep angles bounded.
- For correctness with extreme denominators, use `--high-precision` with small `n` and increase `--mp-dps` as needed.

---

## Project structure (suggested)

- `spectralarith.py` — implementation (matrix construction + analysis + CLI)
- `README.md` — documentation (this file)

---
