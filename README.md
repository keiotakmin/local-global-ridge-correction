# Local and Global Recursive Ridge Correction

Reference implementation and numerical records for *Local and Global Recursive Ridge Correction of Frozen Multivariate Forecasters under Label Delay* (T. Fujimoto and H. Nishi, 2026).

A frozen multi-horizon forecaster is corrected online from its own residuals. Each block of `H` steps, the corrector predicts the leading DCT coefficients of the next residual block from the leading DCT coefficients of the frozen forecast and of the latest input block. The prediction uses an exponentially weighted ridge regression whose sufficient statistics are updated only after the block's target is complete (label delay equal to the horizon).

- **Local corrector:** one regression per channel.
- **Global corrector:** one regression shared by all channels; its learned state does not depend on the channel count.
- **Guard:** keeps a running scale of each channel's residual. Blocks far outside that scale are left out of the update, and the issued correction is clipped by the same scale.

## Install and test

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest test_ridge_corrector
.venv/bin/python reproduce.py
```

`reproduce.py` recomputes the paper's summary table (mean MSE reductions, retained state, paired wins against ELF) from the condition-level records and checks it against `records/summary.csv`.

## Correct your own forecasts

Save chronological arrays in an NPZ file:

- `base` and `target`, each `[blocks, horizon, channels]`;
- `windows`, `[blocks, lookback, channels]`, the input window of each forecast. `lookback` must be a multiple of `horizon`.

Then run:

```bash
.venv/bin/python replay.py forecasts.npz --corrector local --components 8
.venv/bin/python replay.py forecasts.npz --corrector global
```

The defaults are the paper's settings: ridge penalty 1, a forgetting half-life of 128 blocks, guard threshold `tau = 10`, and a blending weight shared by all channels and fitted on the last 32 blocks. In Python:

```python
from ridge_corrector import Config, Corrector
corrector = Corrector(channels, horizon=24, lookback=96, config=Config(local=8))
issued, correction, pending = corrector.issue(base_block, window)   # before the target is known
corrector.update(base_block, target_block, correction, pending)     # after the block is complete
```

## Records

| File | Content |
|---|---|
| `records/conditions_main.csv` | MSE, reduction relative to the frozen forecast, and retained state for every condition and every method of the main comparison. Covers 7 datasets × 3 backbones × 3 seeds × legacy/refit, plus the standard Weather file. |
| `records/conditions_correctors_all_settings.csv` | The correctors' component and sensitivity variants, with the number of skipped channel updates |
| `records/panel_means.csv` | Per dataset and backbone means plotted in the paper's accuracy–state figure |
| `records/summary.csv` | Values of the paper's comparison table |

In the `table` column, `filled` marks the Weather file with its −9999 missing-value codes forward-filled (Weather* in the paper). `standard` marks every other dataset, including the original Weather file. The evaluation protocol is summarized in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Related repository

EPOC, the compressed-residual corrector compared in the paper, is at <https://github.com/keiotakmin/endpoint-preserving-residual-correction>.

Code is provided under the [MIT License](LICENSE). Dataset licenses remain with their providers.
