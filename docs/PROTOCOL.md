# Evaluation protocol

## Data
- ETTh1, ETTh2, ETTm1, ETTm2 (7 channels each), and the Weather (21), Electricity (321), and Traffic (862) CSV files distributed with Autoformer / Time-Series-Library. All numerical columns are forecast.
- **Split:** with series length `T`, `n = floor(T/2)` and `a = floor(0.8 n)`. Legacy bases are trained on `[0, a)`, `[a, n)` is used for selection, and refit bases are retrained on `[0, n)`.
- **Normalization:** each channel is standardized with the mean and standard deviation of `[0, n)`. Errors are reported on this scale.
- **Forecasts:** lookback 96, horizon 24, origins every 24 samples from `n`. Only complete target blocks are scored.
- **Weather:** the standard Weather file contains 81 cells with the code −9999. The main view uses a copy in which these cells are forward-filled, with every base retrained on it. The standard file is reported separately.
- **Exchange:** omitted, because the compact PatchTST base without instance normalization fails on its evaluation period.

## Frozen forecasters
- **Backbones:** DLinear, a compact PatchTST, and iTransformer of the same size.
- **Training:** seeds 0–2, Adam at 1e-3, batch size 32. The update count is chosen from {200, 500, 1000, 2000, 4000, 8000, 20000} by validation MSE.
- **Conditions:** 7 datasets × 3 backbones × 3 seeds × legacy/refit = 126 conditions, plus 18 on the standard Weather file.

## Methods
- **Order of information:** every method issues a block before its target is available and updates only after the target block is complete.
- **Blending:** every correction is scaled by one weight in [0, 1], shared by all channels and fitted by least squares on the last 32 completed blocks.
- **Correctors:** ridge penalty 1, half-life 128 blocks, guard threshold 10. The local corrector uses `K = 4` or `8` DCT components. The global corrector uses 8 components and all four input sub-blocks.
- **Development:** the correctors' structure, features, and settings were chosen during development on these same 126 conditions. The results are empirical results on the development data, not an independent validation.

## Retained state
Retained state counts the arrays kept between forecasts, at the dtype they are implemented with:
- **Not counted:** the frozen forecaster, the shared input window, and issued forecasts awaiting their targets.
- **Counted for the correctors:** the ridge statistics (upper-triangular Gram matrix and right-hand side), the DCT basis and component indices, one guard scale per channel, and the blending weight's history (528 bytes).
