# Loading output-cadence screen

The available loading data do not support blanket temporal thinning. Delta-wing aggregate forces tolerate the screened factors in the saved interval, but individual panel loads do not. Rotor aggregate and panel histories reject every screened factor. All source files, samples and configured output cadences remain unchanged.

## Reproduction and coverage

```bash
python studies/analyze_loading_cadence.py build/archive-restoration-check
```

The argument is any restored archive root containing `vpm/05_delta_wing` and `vpm/06_rotor_flow`; it is not a machine-specific path. The script hashes its CSV inputs and streams one file at a time into numeric arrays. The largest array is 22.21 MB. It reads four channels for every chordwise panel (circulation and three forces), and aggregate force, moment and power histories. The four analytic regression checks pass.

- Delta wing: 2,410 accepted times, 0.0025–6.025 s, 288 panels on each of two surfaces; 6.0225 forcing cycles at 1 Hz.
- Rotor: 1,092 accepted times, 0.006–6.552 s, 132 panels on each of three blades; 8.5083 nominal revolutions. Nominal angular velocity is 49/6 rad/s, rotational frequency 1.299765 Hz, and three-blade passage frequency 3.899296 Hz.
- Rows beyond each archived accepted solver clock are excluded, not deleted: delta two aggregate rows and 576 rows per panel file; rotor three aggregate rows and 396 rows per panel file.

Current chordwise CSV inputs total 679.14 MB for delta and 261.61 MB for rotor. The lossless archive preserves them. These are problematic storage streams, but their size does not prove that their physical transient history is oversampled.

## Method and measured results

Each candidate retains every nth time plus the final endpoint. Linear interpolation reconstructs original saved times. RMS error is normalized by signal RMS; peak error compares maximum absolute loads; integral error is normalized by the original integral of absolute load, so sign cancellation does not create an inflated relative integral error. A linear-detrended Hann periodogram estimates the energy above candidate Nyquist. Channels below 1% of the largest RMS within the same unit family are classified as near-zero and counted in the JSON; symmetry roundoff does not set physical output frequency.

The provisional screening limits are 1% RMS error, 1% peak error, 0.5% integral error and 0.1% spectral energy above Nyquist. They are explicit study criteria, not solver controls or a claim of final numerical convergence. The table reports the worst significant channel; panel rows combine all files of that case.

| Case / history | Factor | Interval [s] | Nyquist [Hz] | RMS error [%] | Peak error [%] | Integral error [%] | High-frequency energy [%] | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| delta / aggregate | 2 | 0.0050 | 100.000 | 0.1051 | 0.1212 | 0.0023 | 0.0001 | Available interval passes |
| delta / aggregate | 5 | 0.0125 | 40.000 | 0.2920 | 0.2045 | 0.0131 | 0.0001 | Available interval passes |
| delta / aggregate | 10 | 0.0250 | 20.000 | 0.9007 | 0.2624 | 0.0490 | 0.0002 | Available interval passes |
| delta / panels | 2 | 0.0050 | 100.000 | 3.2306 | 3.3168 | 0.1961 | 0.9999 | Reject |
| delta / panels | 5 | 0.0125 | 40.000 | 4.1390 | 11.5106 | 0.1775 | 1.6457 | Reject |
| delta / panels | 10 | 0.0250 | 20.000 | 8.9835 | 13.3244 | 0.5822 | 1.8211 | Reject |
| rotor / aggregate | 2 | 0.0120 | 41.667 | 5.6526 | 0.0179 | 0.1744 | 0.0000 | Reject |
| rotor / aggregate | 4 | 0.0240 | 20.833 | 10.5760 | 0.2208 | 0.5229 | 0.0000 | Reject |
| rotor / panels | 2 | 0.0120 | 41.667 | 45.8541 | 0.0963 | 3.8815 | 0.0017 | Reject |
| rotor / panels | 4 | 0.0240 | 20.833 | 85.7945 | 0.3824 | 11.6449 | 0.0028 | Reject |

## Recommendations and limits

- Keep the existing panel-load cadence in both cases. Even factor two loses significant local transient information; acceptable aggregate reconstruction does not establish acceptable local-load reconstruction.
- Delta aggregate factors 2, 5 and 10 are qualified only for reconstructing the archived aggregate interval. There is insufficient evidence to change the production default: the 20 s case is incomplete and its later wake evolution is absent.
- Retain rotor aggregate cadence. Factors 2 and 4 miss startup/transient information despite a small high-frequency Hann energy fraction. The taper suppresses endpoints, so a smooth spectrum alone cannot certify startup peaks or interpolation accuracy.
- Neither source is established stationary, and the rotor archive ends before the reported instability near 7.5 s. These measurements cannot certify the missing late state or the complete requested simulation. Repeat this exact script on final accepted histories before any future cadence change.
- Marker spacing in line plots is a visual choice and does not itself discard samples. Retain original histories; lossless compression and milestone LFS snapshots currently provide the supported storage reduction.

Exact input hashes, panel counts, ignored-channel counts and per-column/per-factor measurements are in [loading_cadence_screen.json](loading_cadence_screen.json). The implementation is [analyze_loading_cadence.py](analyze_loading_cadence.py). This study evaluates output retention, not particle injection or coupled exchange accuracy.
