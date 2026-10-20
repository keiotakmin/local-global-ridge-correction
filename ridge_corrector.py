"""Local and global recursive ridge correctors for a frozen multi-horizon forecaster.

Each block of H steps, a corrector predicts the leading DCT coefficients of the residual (target minus the frozen
forecast) from the DCT coefficients of the forecast and of the newest input sub-blocks of H steps, plus the last
input value. The map is an exponentially weighted ridge regression with a penalized intercept; its sufficient
statistics are updated recursively once the block's target is complete (delay d = H), so nothing else is kept.

  local    one ridge per channel (state grows with the channel count)
  global   one ridge shared by all channels (state independent of the channel count)
  both     the global ridge, then a local ridge fitted to what the global one leaves

The guard keeps, per channel, an exponentially weighted scale of the frozen forecast's residual power over the
leading GUARD_COMPONENTS DCT coefficients. A block whose power exceeds tau^2 times the scale is left out of both
ridges and of the scale; the issued correction coefficients are scaled down to norm at most tau * sqrt(k * scale).
The correction then goes through a blending weight in [0, 1] shared by all channels and fitted by least squares to
the last 32 blocks (the global blend of EPOC). Every quantity used for a block is computed before its target.
"""
from dataclasses import dataclass

import numpy as np

GUARD_COMPONENTS = 8  # residual DCT coefficients watched by the guard (and the width of the correction vector)
BLEND_WINDOW = 32


def dct_basis(h, k):
    """Orthonormal DCT-II basis, h x k."""
    ids = np.arange(k)
    u = np.cos(np.pi * (np.arange(h)[:, None] + .5) * ids / h) * np.sqrt(2 / h)
    u[:, ids == 0] /= np.sqrt(2)
    return u


class Ridge:
    """Exponentially weighted ridge with a penalized intercept; one system per channel or one shared."""

    def __init__(self, channels, width, outputs, shared, ridge=1., half_life=128.):
        self.shared, self.ridge, self.decay = shared, ridge, 2. ** (-1. / half_life)
        self.i, self.j = np.triu_indices(width)
        self.diag = self.i == self.j
        groups = 1 if shared else channels
        self.gram = np.zeros((groups, len(self.i)))  # upper triangle of the Gram matrix
        self.gram[:, self.diag] = ridge
        self.rhs = np.zeros((groups, width, outputs))
        self.width = width

    def predict(self, x):  # x: channel, width -> channel, outputs
        g = np.empty((self.gram.shape[0], self.width, self.width))
        g[:, self.i, self.j] = self.gram
        g[:, self.j, self.i] = self.gram
        beta = np.linalg.solve(g, self.rhs)
        return x @ beta[0] if self.shared else np.einsum('cp,cpk->ck', x, beta)

    def update(self, x, y, keep):
        """Add the kept channels' samples; a skipped channel's own system (local) is left untouched."""
        if self.shared:
            if not keep.any():
                return
            x, y = x[keep], y[keep]
            self.gram *= self.decay
            self.gram += (x[:, self.i] * x[:, self.j]).sum(0, keepdims=True)
            self.gram[:, self.diag] += (1 - self.decay) * self.ridge
            self.rhs *= self.decay
            self.rhs += (x[:, :, None] * y[:, None, :]).sum(0, keepdims=True)
            return
        gram, rhs = self.gram * self.decay, self.rhs * self.decay
        gram += x[:, self.i] * x[:, self.j]
        gram[:, self.diag] += (1 - self.decay) * self.ridge
        rhs += x[:, :, None] * y[:, None, :]
        self.gram[keep], self.rhs[keep] = gram[keep], rhs[keep]

    @property
    def state_bytes(self):
        return self.gram.nbytes + self.rhs.nbytes


class Blend:
    """Blending weight shared by all channels: clip(<c, r> / <c, c>, 0, 1) over the last `window` blocks."""

    def __init__(self, window=BLEND_WINDOW):
        self.history, self.sums, self.t = np.zeros((window, 2)), np.zeros(2), 0

    def weight(self):
        a, b = self.sums
        return float(np.clip(b / a, 0, 1)) if a > 0 else 0.

    def update(self, correction, residual):
        value = np.array([np.mean(correction ** 2), np.mean(correction * residual)])
        slot = self.t % len(self.history)
        self.sums += value - self.history[slot]
        self.history[slot] = value
        self.t += 1

    @property
    def state_bytes(self):
        return self.history.nbytes + self.sums.nbytes


@dataclass(frozen=True)
class Config:
    local: int = 8           # DCT components of the local ridge (input and output); 0 = no local ridge
    local_blocks: int = 1    # newest input sub-blocks the local ridge reads; 0 = forecast only
    global_: bool = False    # global ridge on the forecast and all input sub-blocks, GUARD_COMPONENTS components
    guard: bool = True
    tau: float = 10.
    ridge: float = 1.
    half_life: float = 128.  # ridges, in blocks
    guard_half_life: float = 128.


class Corrector:
    def __init__(self, channels, horizon=24, lookback=96, config=Config()):
        if lookback % horizon or not (config.local or config.global_):
            raise ValueError('lookback must be a multiple of horizon; at least one ridge is required')
        self.cfg, self.h, self.blocks = config, horizon, lookback // horizon
        k, kl = GUARD_COMPONENTS, config.local
        self.u = dct_basis(horizon, k)
        self.glob = Ridge(channels, 2 + k + self.blocks * k, k, True, config.ridge, config.half_life) \
            if config.global_ else None
        self.loc = Ridge(channels, 1 + kl + (kl * config.local_blocks + 1 if config.local_blocks else 0), kl, False,
                         config.ridge, config.half_life) if kl else None
        self.scale = np.full(channels, np.nan) if config.guard else None
        self.decay = 2. ** (-1. / config.guard_half_life)
        self.blend = Blend()

    def _features(self, base, window):
        u, c = self.u, base.shape[1]
        ones, a = np.ones((1, c)), u.T @ base
        sub = window.reshape(self.blocks, self.h, c)[::-1]  # newest first
        last = np.sqrt(self.h) * window[-1:]
        xg = np.concatenate([ones, a] + [u.T @ s for s in sub] + [last]).T if self.glob else None
        kl, nb = self.cfg.local, self.cfg.local_blocks
        xl = np.concatenate([ones, a[:kl]] + ([(u.T @ s)[:kl] for s in sub[:nb]] + [last] if nb else [])).T \
            if self.loc else None
        return xg, xl

    def correct(self, base, window):
        """Correction (horizon x channels) before blending, and the internals needed by `update`."""
        xg, xl = self._features(base, window)
        c, k = base.shape[1], GUARD_COMPONENTS
        zg = self.glob.predict(xg) if self.glob else np.zeros((c, k))
        z = zg.copy()
        if self.loc:
            z[:, :self.cfg.local] += self.loc.predict(xl)
        if self.scale is not None:
            limit = self.cfg.tau * np.sqrt(k * np.nan_to_num(self.scale, nan=np.inf))
            z *= np.minimum(1., limit / np.maximum(np.linalg.norm(z, axis=1), 1e-300))[:, None]
        return self.u @ z.T, (xg, xl, zg)

    def issue(self, base, window):
        correction, pending = self.correct(base, window)
        return base + self.blend.weight() * correction, correction, pending

    def update(self, base, target, correction, pending):
        xg, xl, zg = pending
        residual = target - base
        self.blend.update(correction, residual)
        z = (self.u.T @ residual).T
        keep = np.ones(len(z), bool)
        if self.scale is not None:
            power = np.mean(z ** 2, axis=1)
            fresh = np.isnan(self.scale)
            keep = fresh | (power <= self.cfg.tau ** 2 * self.scale)
            self.scale = np.where(fresh, power, np.where(keep, self.decay * self.scale + (1 - self.decay) * power,
                                                         self.scale))
        if self.glob:
            self.glob.update(xg, z, keep)
        if self.loc:
            self.loc.update(xl, (z - zg)[:, :self.cfg.local], keep)
        return keep

    @property
    def state_bytes(self):
        """Ridges, DCT basis and component ids, guard scale and blend."""
        state = self.u.nbytes + 8 * GUARD_COMPONENTS + self.blend.state_bytes
        state += sum(r.state_bytes for r in (self.glob, self.loc) if r)
        return state + (self.scale.nbytes if self.scale is not None else 0)


def replay(base, target, windows, config=Config()):
    """Chronological replay over blocks [n, horizon, channels]: issue each block, then learn from its target.
    Returns the blended and unblended forecasts and the number of skipped channel updates."""
    n, h, c = base.shape
    corrector = Corrector(c, h, windows.shape[1], config)
    issued, raw, skipped = np.empty_like(base), np.empty_like(base), 0
    for t in range(n):
        issued[t], correction, pending = corrector.issue(base[t], windows[t])
        raw[t] = base[t] + correction
        skipped += int((~corrector.update(base[t], target[t], correction, pending)).sum())
    return issued, raw, skipped, corrector.state_bytes
