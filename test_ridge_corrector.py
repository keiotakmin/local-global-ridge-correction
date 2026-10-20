import unittest

import numpy as np

from ridge_corrector import GUARD_COMPONENTS, Config, Corrector, dct_basis, replay

H, L = 24, 96


def stream(n=80, c=5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n * H + L)
    series = np.sin(2 * np.pi * t / 24)[:, None] * rng.uniform(.5, 2, c) + .1 * rng.standard_normal((len(t), c))
    windows = np.stack([series[i * H:i * H + L] for i in range(n)])
    target = np.stack([series[i * H + L:i * H + L + H] for i in range(n)])
    base = .8 * target + .05 * rng.standard_normal(target.shape)  # a biased frozen forecast
    return base, target, windows


CONFIGS = [Config(local=4), Config(local=8), Config(local=0, global_=True), Config(local=8, global_=True),
           Config(local=8, guard=False)]


class TestCorrector(unittest.TestCase):
    def test_basis_orthonormal(self):
        u = dct_basis(H, GUARD_COMPONENTS)
        np.testing.assert_allclose(u.T @ u, np.eye(GUARD_COMPONENTS), atol=1e-12)

    def test_causal(self):
        """Changing the target of block k leaves every forecast issued up to block k unchanged."""
        base, target, windows = stream()
        for config in CONFIGS:
            issued, _, _, _ = replay(base, target, windows, config)
            changed = target.copy()
            changed[40:] += 3.
            again, _, _, _ = replay(base, changed, windows, config)
            np.testing.assert_array_equal(issued[:41], again[:41])
            self.assertFalse(np.array_equal(issued[41:], again[41:]))

    def test_improves_biased_forecast(self):
        base, target, windows = stream()
        for config in CONFIGS:
            issued, _, _, _ = replay(base, target, windows, config)
            self.assertLess(np.mean((issued - target) ** 2), np.mean((base - target) ** 2), config)

    def test_state_bytes(self):
        for c in (1, 7, 30):
            q4, q8, qg = 10, 18, 2 + 8 + 4 * 8
            shared = 24 * 8 * 8 + 8 * 8 + 66 * 8 + 8 * c
            self.assertEqual(Corrector(c, H, L, Config(local=4)).state_bytes, shared + c * 8 * (q4 * (q4 + 1) // 2 + q4 * 4))
            self.assertEqual(Corrector(c, H, L, Config(local=8)).state_bytes, shared + c * 8 * (q8 * (q8 + 1) // 2 + q8 * 8))
            self.assertEqual(Corrector(c, H, L, Config(local=0, global_=True)).state_bytes,
                             shared + 8 * (qg * (qg + 1) // 2 + qg * 8))

    def test_guard_skips_outlier_block(self):
        base, target, windows = stream()
        bad = target.copy()
        bad[50, :12, 2] = -162.  # a missing-value code after standardization
        _, guarded, skipped, _ = replay(base, bad, windows, Config(local=8))
        _, open_, none, _ = replay(base, bad, windows, Config(local=8, guard=False))
        self.assertGreaterEqual(skipped, 1)
        self.assertEqual(none, 0)
        later = slice(51, None)  # unblended corrections: the shared blend sees the outlier in both runs
        self.assertLess(np.mean((guarded[later] - target[later]) ** 2), np.mean((open_[later] - target[later]) ** 2))

    def test_clip_bound(self):
        base, target, windows = stream()
        corrector = Corrector(5, H, L, Config(local=8, tau=1.))
        for t in range(30):
            correction, pending = corrector.correct(base[t], windows[t])
            if t:
                coef = corrector.u.T @ correction
                limit = np.sqrt(GUARD_COMPONENTS * corrector.scale)
                self.assertTrue(np.all(np.linalg.norm(coef, axis=0) <= limit * (1 + 1e-9)))
            corrector.update(base[t], target[t], correction, pending)


if __name__ == '__main__':
    unittest.main()
