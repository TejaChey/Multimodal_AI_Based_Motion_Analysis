"""Descriptive comparison of an athlete's features with the reference-set tier means.

This is NOT a validated coaching norm: the reference set is a handful of recordings from one session.
It only shows where this recording sits relative to those recordings.
"""

# feature -> (label, unit, higher_is_better or None)
FEATURES = {
    "gct_ms": ("Ground contact time", "ms", False),
    "step_freq_video_hz": ("Step frequency", "Hz", True),
    "knee_sym_index": ("Knee asymmetry index", "%", False),
    "vert_osc_stature": ("Vertical oscillation", "stature", False),
    "peak_accel": ("Peak watch acceleration", "m/s2", True),
    "trunk_lean_deg": ("Trunk lean", "deg", None),
}


def generate_coaching_feedback(metrics: dict, reference: dict | None = None) -> list[str]:
    if not reference or "Elite" not in reference.get("tier_means", {}):
        return ["Reference set unavailable, so no comparison can be shown."]
    elite = reference["tier_means"]["Elite"]
    n = reference["tier_counts"].get("Elite", 0)
    out = [f"Reference: mean of the {n} Elite-tier recording(s) in the training set (small sample; descriptive only)."]
    for key, (name, unit, higher) in FEATURES.items():
        if key not in metrics or key not in elite:
            continue
        you, ref = float(metrics[key]), float(elite[key])
        side = "" if higher is None else (" (higher is faster)" if higher else " (lower is faster)")
        gap = "similar to" if abs(you - ref) <= 0.1 * abs(ref) else ("above" if you > ref else "below")
        out.append(f"{name}: {you:.2f} {unit}, {gap} the Elite mean of {ref:.2f} {unit}{side}.")
    return out
