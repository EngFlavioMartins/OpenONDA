"""UNWIRED O(N) all-native-target Gaussian paired-image remainder census.

This applies the proved envelope in _slip_periodic_gaussian_oracle.py, using
the farthest source-AABB corner as a safe upper bound on every source-target
distance. It evaluates no induction field and changes no simulation input.
The real-arithmetic inequality is rigorous; padded f64 arithmetic here is not
an interval certificate and does not include finite-field/FFT/FMM roundoff.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gaussian_integral_envelope(gap, sigma_max, period):
    """Bound erfc(gap/sigma)/(2*pi*period*sigma²) for all sigma<=sigma_max.

    For r>=sqrt(3/2)*sigma_max the Gaussian density increases with sigma.
    Then use erfc(x)<=exp(-x²)/(x*sqrt(pi)). A positive1e-300 upper floor
    prevents reporting a mathematically positive tail as numerical zero.
    """
    if np.any(gap < math.sqrt(1.5)*sigma_max):
        raise ValueError("core-uniform far-tail density monotonicity not admitted")
    ratio = gap/sigma_max
    log_value = (-ratio*ratio-np.log(ratio)-.5*math.log(math.pi)
                 -math.log(2*math.pi*period*sigma_max**2))
    return np.exp(np.maximum(log_value, math.log(1e-300)))


def summary(values, threshold, targets):
    index = int(np.argmax(values))
    return {
        "minimum": float(values.min()), "maximum": float(values[index]),
        "quantiles_0_25_50_75_95_100": np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist(),
        "max_target_index": index, "max_target_position": targets[index].tolist(),
        "targets_with_bound_at_or_below_requested_tolerance": int(np.count_nonzero(values <= threshold)),
        "all_targets_bounded_at_requested_tolerance": bool(np.all(values <= threshold)),
    }


def census(x, gamma, sigma, *, zmin, zmax, shells, tolerance, velocity_scale, gradient_scale):
    if (x.ndim != 2 or x.shape[1:] != (3,) or gamma.shape != x.shape
            or sigma.shape != (len(x),) or not len(x)
            or not all(np.isfinite(v).all() for v in (x, gamma, sigma))
            or np.any(sigma <= 0) or np.any(x[:, 2] < zmin) or np.any(x[:, 2] > zmax)):
        raise ValueError("finite native source fields inside the physical slab required")
    period = 2*(zmax-zmin)
    magnitudes = np.linalg.norm(gamma, axis=1)
    absolute_strength = math.fsum(magnitudes.tolist())
    if not math.isfinite(absolute_strength) or absolute_strength <= 0:
        raise ValueError("positive finite absolute circulation required for native census")
    padded_strength = np.nextafter(absolute_strength*(1+128*np.finfo(float).eps), np.inf)
    sigma_max = float(sigma.max())
    fields = {k: {"u": np.zeros(len(x)), "j": np.zeros(len(x)), "u_rms": np.zeros(len(x))}
              for k in shells}
    families = []
    for odd in (False, True):
        source = x.copy()
        if odd:
            source[:, 2] = 2*zmin-source[:, 2]
        lower, upper = source.min(axis=0), source.max(axis=0)
        coordinate_scale = max(float(np.abs(source).max()), float(np.abs(x).max()), 1.)
        padding = 128*np.finfo(float).eps*coordinate_scale
        axis_distance = np.maximum(np.abs(x-lower), np.abs(x-upper))+padding
        extent = np.nextafter(np.linalg.norm(axis_distance, axis=1), np.inf)
        # Optional tighter numerator: sum w_j D_j <= S*sqrt(sum w_j D_j²/S).
        centre = np.array([math.fsum((magnitudes*source[:, axis]).tolist())/absolute_strength
                           for axis in range(3)])
        variance = math.fsum((magnitudes*np.sum((source-centre)**2, axis=1)).tolist())/absolute_strength
        rms = np.nextafter(np.sqrt(np.sum((x-centre)**2, axis=1)+variance)+padding, np.inf)
        family = {"odd": odd, "source_aabb": [lower.tolist(), upper.tolist()],
                  "maximum_target_to_source_aabb_extent": float(extent.max()),
                  "strength_weighted_centre": centre.tolist(), "weighted_spread_variance": variance,
                  "maximum_weighted_rms_extent": float(rms.max()), "shells": []}
        for k in shells:
            gap = period*k-extent
            if np.any(gap <= 0):
                raise ValueError(f"far-tail expansion not admitted at K={k}")
            s3 = 1/(2*period*gap**2)
            exponential = gaussian_integral_envelope(gap, sigma_max, period)
            common = s3/(2*math.pi)+exponential
            fields[k]["u"] += 2*extent*padded_strength*common
            fields[k]["u_rms"] += 2*rms*padded_strength*common
            fields[k]["j"] += 2*padded_strength*(math.sqrt(5)*s3/(4*math.pi)+exponential)
            family["shells"].append({"K": k, "minimum_gap": float(gap.min()),
                                      "gaussian_integral_upper_maximum": float(exponential.max())})
        families.append(family)
    records = []
    for k in shells:
        value = fields[k]
        if not all(np.isfinite(v).all() for v in value.values()):
            raise FloatingPointError("nonfinite tail envelope")
        records.append({
            "K": k, "explicit_image_count_excluding_physical_primary": 4*k+1,
            "velocity_absolute": summary(value["u"], tolerance*velocity_scale, x),
            "gradient_frobenius_absolute": summary(value["j"], tolerance*gradient_scale, x),
            "velocity_absolute_with_weighted_rms_numerator": summary(value["u_rms"], tolerance*velocity_scale, x),
            "maximum_scaled_envelope": float(np.max(np.maximum(value["u"]/velocity_scale, value["j"]/gradient_scale))),
            "maximum_scaled_envelope_with_weighted_rms_numerator": float(np.max(np.maximum(value["u_rms"]/velocity_scale, value["j"]/gradient_scale))),
        })
    return {"absolute_strength": absolute_strength, "core_min": float(sigma.min()), "core_max": sigma_max,
            "period": period, "families": families, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    proof = Path(__file__).with_name("_slip_periodic_gaussian_oracle.py")
    source_hash, proof_hash = digest(__file__), digest(proof)
    reference = json.loads(args.reference_report.read_text())
    original = digest(args.checkpoint)
    if original != reference["identity"]["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from native field evidence")
    reference_hash = digest(args.reference_report)
    start = perf_counter()
    with h5py.File(args.checkpoint, "r") as saved:
        config = json.loads(saved["solver"].attrs["numerical_configuration"])
        if config != reference["identity"]["configuration"]:
            raise ValueError("checkpoint numerical configuration differs from native field evidence")
        x = saved["particles/position"][:].astype(np.float64)
        gamma = saved["particles/vortex_strength"][:].astype(np.float64)
        sigma = saved["particles/core_radius"][:].astype(np.float64)
        time = float(saved["solver"].attrs["time"])
    if len(x) != reference["identity"]["particles"] or time != reference["identity"]["time"]:
        raise ValueError("native point count/time mismatch")
    induction = config["induction"]
    if config["particle_kernel"] != "GAUSSIAN" or induction["method"] != "SLIP_SLAB":
        raise ValueError("Gaussian slab configuration required")
    result = census(x, gamma, sigma, zmin=induction["z_min"], zmax=induction["z_max"],
                    shells=[32, 64, 128, 256], tolerance=induction["tail_tolerance"],
                    velocity_scale=induction["velocity_scale"], gradient_scale=induction["gradient_scale"])
    if digest(args.checkpoint) != original or digest(args.reference_report) != reference_hash:
        raise RuntimeError("preserved evidence changed during census")
    if digest(__file__) != source_hash or digest(proof) != proof_hash:
        raise RuntimeError("qualification/proof source changed during census")
    result.update(
        status="complete", particles=len(x), time=time, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=original, native_field_report=str(args.reference_report.resolve()),
        native_field_report_sha256=reference_hash, source_sha256=source_hash,
        proved_envelope_source=str(proof.resolve()), proved_envelope_source_sha256=proof_hash,
        induction_configuration=induction, seconds=perf_counter()-start,
        targets="every original saved physical particle; no subset or N² pair allocation",
        inequality="Oracle paired-shell inequality, replacing each D_ij by farthest source-AABB corner distance per target. Weighted-RMS variant improves only the numerator by Cauchy-Schwarz; denominator retains maximum AABB extent.",
        scope="absolute infinite-image remainder AFTER the complete finite +/-K sum, not error against legacy FMM or interpolation/roundoff",
        arithmetic="real-arithmetic bound evaluated in padded f64; not a directed interval certificate; positive1e-300 Gaussian integral upper floor",
        interpretation="An upper bound exceeding tolerance means this certificate is inconclusive, not that the actual remainder exceeds tolerance.",
        production_admissible=False, unchanged_config=True, original_two_consecutive_block_gate_replaced=False,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{key: row[key] for key in ("K", "maximum_scaled_envelope",
                      "maximum_scaled_envelope_with_weighted_rms_numerator")}
                     for row in result["records"]], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
