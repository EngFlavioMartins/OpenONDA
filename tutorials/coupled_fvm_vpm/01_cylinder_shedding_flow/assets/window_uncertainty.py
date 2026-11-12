#!/usr/bin/env python3
"""Statistical uncertainty of the study statistics over the 40-55 s window.

Every run of the study is compared with its FVM-only reference over the same window of a few
shedding periods, while the lift amplitude is still growing. Two things limit how small an error
the comparison resolves:

- the spread of a statistic between windows of the developed flow, sigma_w: the standard deviation
  of the statistic over 15 s windows slid in steps of 2.5 s from t = 55 s to the end of a run, in
  each run that reaches 95 s;
- the spread between runs of the transient drift of the 40-55 s window, s_d: the drift of a run
  is its 40-55 s value over its developed mean, minus one, and s_d is the standard deviation of
  the drift over the same runs.

The error of a run is a difference of two runs over the same window, so its standard uncertainty
is sigma = sqrt(2 sigma_w^2 + s_d^2), with sigma_w^2 averaged over the runs. The figures draw
+-2 sigma; an error inside that bar is not resolved.

Run from the case directory, python -m assets.window_uncertainty, to print the terms.
"""

import numpy as np

WINDOW = 15.0  # s, length of the study window
SLIDE = 2.5  # s, offset between sliding windows
LONG_RUN_END = 95.0  # s, runs that reach this time give the uncertainty


def sliding_statistics(forces, start, statistics_of):
    """Statistics of every full window from ``start`` to the end of a run."""
    end_time = forces["time"][-1]
    rows = []
    for begin in np.arange(start, end_time - WINDOW + 1.0e-9, SLIDE):
        try:
            rows.append(statistics_of(forces, begin, begin + WINDOW))
        except (IndexError, ValueError):
            continue
    return rows


def window_uncertainty(long_runs, study_window, steady_start, statistics_of, keys):
    """Standard uncertainty (fraction) of each statistic's error, and the terms behind it.

    ``long_runs`` maps a run name to its force history; runs shorter than LONG_RUN_END are
    ignored. Returns {"sigma": {key: value}, "sigma_w": ..., "s_d": ..., "runs": [...],
    "drift": {run: {key: value}}}; with no long run every value is NaN, and with one long run
    s_d is NaN and sigma uses sigma_w alone.
    """
    variances, drifts, used = {key: [] for key in keys}, {}, []
    for name, forces in long_runs.items():
        if forces["time"][-1] < LONG_RUN_END:
            continue
        rows = sliding_statistics(forces, steady_start, statistics_of)
        try:
            early = statistics_of(forces, *study_window)
        except (IndexError, ValueError):
            continue
        if len(rows) < 3:
            continue
        used.append(name)
        drifts[name] = {}
        for key in keys:
            values = np.array([row[key] for row in rows])
            variances[key].append((values.std(ddof=1) / values.mean()) ** 2)
            drifts[name][key] = early[key] / values.mean() - 1.0
    sigma_w, s_d, sigma = {}, {}, {}
    for key in keys:
        sigma_w[key] = float(np.sqrt(np.mean(variances[key]))) if used else np.nan
        values = [drifts[name][key] for name in used]
        s_d[key] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        terms = 2.0 * sigma_w[key] ** 2 + (0.0 if np.isnan(s_d[key]) else s_d[key] ** 2)
        sigma[key] = float(np.sqrt(terms)) if used else np.nan
    return {"sigma": sigma, "sigma_w": sigma_w, "s_d": s_d, "runs": used, "drift": drifts}


def main():
    from assets.plot_study import STATISTICS, STEADY_START, STUDY_WINDOW, long_runs
    from assets.plot_study import shedding_statistics

    result = window_uncertainty(long_runs(), STUDY_WINDOW, STEADY_START, shedding_statistics, STATISTICS)
    print(f"runs reaching {LONG_RUN_END:g} s: {', '.join(result['runs']) or 'none'}")
    for key in STATISTICS:
        print(
            f"  {key:7s} sigma_w {100 * result['sigma_w'][key]:.3f} %  s_d {100 * result['s_d'][key]:.3f} %"
            f"  2 sigma {200 * result['sigma'][key]:.3f} %"
        )
    for name, drift in result["drift"].items():
        print(f"  drift of {STUDY_WINDOW[0]:g}-{STUDY_WINDOW[1]:g} s, {name}: "
              + ", ".join(f"{key} {100 * value:+.2f} %" for key, value in drift.items()))


if __name__ == "__main__":
    main()
