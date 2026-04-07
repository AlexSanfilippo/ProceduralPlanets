import numpy as np
from noise import pnoise3


def fBm_1d(n, H, dt=1.0, mean=0.0, seed=None, eps=1e-12):
    """
    Generate 1D fractional Brownian motion (fBm) samples using Cholesky decomposition.

    Parameters
    - n: int, number of samples (samples at t = 0, dt, 2*dt, ..., (n-1)*dt)
    - H: float in (0,1), Hurst exponent
    - dt: float, time step
    - mean: float, add a constant offset to the result
    - seed: None | int | np.random.Generator - controls RNG
    - eps: float, initial jitter for numerical stability if covariance is not PD

    Returns
    - samples: np.ndarray, shape (n,), dtype float64
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)

    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0, 1)")

    # RNG handling
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # time samples
    t = np.arange(n, dtype=np.float64) * float(dt)

    # Precompute t^{2H}
    t2H = np.zeros_like(t)
    nonzero = t > 0.0
    t2H[nonzero] = (t[nonzero] ** (2.0 * H))
    # Covariance matrix: C[i,j] = 0.5*(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})
    diff = np.abs(t[:, None] - t[None, :])
    C = 0.5 * (t2H[:, None] + t2H[None, :] - (diff ** (2.0 * H)))

    # Ensure symmetry (numerical)
    C = 0.5 * (C + C.T)

    # Try Cholesky, add jitter adaptively if needed
    jitter = eps
    for attempt in range(10):
        try:
            L = np.linalg.cholesky(C)
            break
        except np.linalg.LinAlgError:
            # add jitter to diagonal and retry
            C[np.diag_indices_from(C)] += jitter
            jitter *= 10.0
    else:
        raise np.linalg.LinAlgError("Covariance matrix not positive definite even after jittering")

    # draw standard normals and produce correlated Gaussian vector
    z = rng.standard_normal(size=(n,))
    samples = mean + L.dot(z)

    return samples

def plot_fBm_1d(n, H, dt=1.0, mean=0.0, seed=None, eps=1e-12,
                figsize=(10, 4), title=None, save_path=None, show=True):
    """
    Generate 1D fBm samples with `fBm_1d` and plot them using matplotlib.

    Parameters
    - n, H, dt, mean, seed, eps: forwarded to fBm_1d
    - figsize: tuple, matplotlib figure size
    - title: optional title string
    - save_path: optional path to save the figure (PNG, PDF, etc.)
    - show: if True, call plt.show(); otherwise just save/return the figure

    Raises RuntimeError if matplotlib is not available.
    """
    samples = fBm_1d(n, H, dt=dt, mean=mean, seed=seed, eps=eps)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required to plot fBm_1d; install it (pip install matplotlib)") from e

    t = np.arange(n, dtype=np.float64) * float(dt)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, samples, lw=1.25, color="tab:blue")
    ax.set_xlabel("t")
    ax.set_ylabel("fBm value")
    ax.grid(True, linestyle="--", alpha=0.4)

    if title is None:
        title = f"1D fBm (n={n}, H={H})"
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def fBm_2d(nx, ny, H, dx=1.0, seed=None, k0=None):
    """
    Generate a 2D fractal Gaussian field (fractional Gaussian field) by
    spectral synthesis: create white noise in real space, multiply its
    Fourier coefficients by a power-law spectrum ~ (k^2 + k0^2)^-(H+1),
    then inverse transform.

    Parameters
    - nx, ny: int, output grid size (x, y)
    - H: float in (0, 1), Hurst exponent controlling roughness (larger => smoother)
    - dx: float, spatial sampling interval
    - seed: None | int | np.random.Generator for reproducibility
    - k0: float | None, small regularization wavenumber; if None chosen automatically

    Returns
    - field: np.ndarray shape (ny, nx), zero-mean, unit-std
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers")
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")

    # RNG
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # regularize low-frequency singularity
    if k0 is None:
        # choose k0 roughly inverse of the largest dimension
        k0 = 1.0 / (max(nx, ny) * float(dx))

    # white noise in real space
    w = rng.standard_normal(size=(ny, nx))

    # forward real FFT
    W = np.fft.rfftn(w)

    # frequency axes (cycles per unit length) -> angular frequency
    fx = np.fft.fftfreq(nx, d=dx)[: nx // 2 + 1]
    fy = np.fft.fftfreq(ny, d=dx)

    kx = 2.0 * np.pi * fx
    ky = 2.0 * np.pi * fy[:, None]
    k = np.sqrt(ky * ky + kx[None, :] * kx[None, :])

    # power spectrum for fractional field; exponent chosen so that spectral slope
    # corresponds to roughness controlled by H. The +1 keeps spectrum integrable.
    S = (k * k + (k0 ** 2)) ** (-(H + 1.0))

    # apply sqrt of spectrum to Fourier coefficients (filter)
    W *= np.sqrt(S)

    # inverse transform to real space and normalize
    field = np.fft.irfftn(W, s=(ny, nx))
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std

    return field


def plot_fBm_2d(nx, ny, H, dx=1.0, seed=None, k0=None,
                cmap="viridis", figsize=(6, 6), title=None, save_path=None, show=True):
    """
    Generate a 2D fBm field with `fBm_2d` and plot it using matplotlib.

    Parameters
    - nx, ny, H, dx, seed, k0: forwarded to fBm_2d
    - cmap: matplotlib colormap
    - figsize: matplotlib figure size
    - title: optional title string
    - save_path: optional path to save figure
    - show: if True, call plt.show()
    """
    field = fBm_2d(nx, ny, H, dx=dx, seed=seed, k0=k0)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required to plot fBm_2d; install it (pip install matplotlib)") from e

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(field, origin="lower", cmap=cmap, extent=[0, nx * dx, 0, ny * dx])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is None:
        title = f"2D fractal field (nx={nx}, ny={ny}, H={H})"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized value")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return fig

def fbm_terrain_3d(x, y, z,
                   octaves=6,
                   lacunarity=2.0,
                   amplitude=1.0,
                   gain=0.5,
                   frequency=1.0,
                   seed=0,
                   ridged=False,
                   domain_warp=True,
                   warp_strength=0.5,
                   warp_octaves=2,
                   normalize=False):
    """
    3D fbm terrain-like noise.

    Parameters
    - x, y, z: scalar or array_like coordinates (must broadcast to same shape)
    - octaves: number of fbm octaves
    - lacunarity: frequency multiplier per octave
    - gain: amplitude multiplier per octave (persistence)
    - frequency: base frequency (applied to coordinates)
    - seed: integer base seed (passed to pnoise3 via base)
    - ridged: if True, use a ridged multifractal (sharper peaks / valleys)
    - domain_warp: if True, do lightweight domain warping before main fbm
    - warp_strength: scale of warp displacement (in world units)
    - warp_octaves: octaves used for warp noise (cheap if small)
    - normalize: if True, normalize result to roughly [-1, 1] (or [0,1] for ridged)

    Returns
    - float (if inputs scalar) or np.ndarray same shape as inputs
    """
    # broadcast inputs to common shape
    X, Y, Z = np.broadcast_arrays(np.asarray(x), np.asarray(y), np.asarray(z))
    out = np.empty_like(X, dtype=np.float64)
    it = np.nditer(X, flags=['multi_index', 'refs_ok'])

    # helper: single-point fbm using per-octave base offsets
    def _fbm_point(px, py, pz, octaves_local, lac, g, freq, base_seed, ridged_local, amp):
        val = 0.0
        max_amp = 0.0
        f = freq
        for o in range(octaves_local):
            n = pnoise3(px * f, py * f, pz * f, base=int(base_seed + 0))
            # pnoise3 typically returns in [-1,1] (implementation dependent)
            if ridged_local:
                # classic ridged multifractal: invert abs and square to emphasize ridges
                n = 1.0 - abs(n)
                n = n * n
            val += n * amp
            max_amp += amp
            amp *= g
            f *= lac
        if max_amp != 0.0:
            val = val / max_amp
        # map ridged result to [0,1] (optional) then shift to [-1,1] if requested by caller
        return float(val)

    # optional lightweight domain warping per sample
    def _warp_point(px, py, pz, warp_octs, lac, g, freq, base_seed, amp):
        wx = wy = wz = 0.0
        f = freq
        base_seed = int(base_seed)
        for o in range(warp_octs):
            wx += pnoise3((px + 31.1) * f, (py + 17.3) * f, (pz + 47.2) * f, base=base_seed + 101 + o) * amp
            wy += pnoise3((px + 13.7) * f, (py + 29.9) * f, (pz + 23.4) * f, base=base_seed + 201 + o) * amp
            wz += pnoise3((px + 11.9) * f, (py + 41.2) * f, (pz + 19.6) * f, base=base_seed + 301 + o) * amp
            amp *= g
            f *= lac
        # normalize warp by total amplitude to keep magnitude predictable
        total_amp = sum((g ** i) for i in range(warp_octs)) if warp_octs > 0 else 1.0
        return (wx / total_amp, wy / total_amp, wz / total_amp)

    while not it.finished:
        idx = it.multi_index
        px = float(X[idx]) * 1.0
        py = float(Y[idx]) * 1.0
        pz = float(Z[idx]) * 1.0

        # apply base frequency
        px *= frequency
        py *= frequency
        pz *= frequency

        # domain warp
        if domain_warp and warp_strength != 0.0:
            wx, wy, wz = _warp_point(px, py, pz, warp_octaves, lacunarity, gain, frequency, seed, amplitude)
            px += wx * warp_strength
            py += wy * warp_strength
            pz += wz * warp_strength
        v = _fbm_point(px, py, pz, octaves, lacunarity, gain, 1.0, seed, ridged, amplitude)

        # normalization/mapping:
        if normalize:
            if ridged:
                # ridged _fbm_point yields values roughly in [0,1], map to [-1,1] if desired
                v = (v * 2.0) - 1.0
            else:
                # plain fbm roughly in [-1,1] after normalization step above; clamp to be safe
                v = float(np.clip(v, -1.0, 1.0))
        out[idx] = v
        it.iternext()

    # return scalar if inputs were scalars
    if np.isscalar(x) and np.isscalar(y) and np.isscalar(z):
        return float(out.reshape(-1)[0])
    return out

def fbm_noise(x, y, z, octaves=8, lacunarity=2.0, gain=0.1, seed=0, amplitude=2.5, frequency=0.06):
    """
    Fractal Brownian Motion noise for 3D coordinates.
    Returns a float in [-1, 1].
    """
    value = 0.0
    for _ in range(octaves):
        value += amplitude * pnoise3(x * frequency, y * frequency, z * frequency, base=int(seed))
        frequency *= lacunarity
        amplitude *= gain
    return value