"""counterfactual.py — actionable, ethically bounded counterfactual search.

Only attributes a student can realistically change are edited.  Demographic
attributes and historical school records are immutable, and optional Kumbhar
fields are skipped when they are missing (``None``) rather than crashing or
inventing a value.
"""

from __future__ import annotations

# Immutable attributes: never suggest changing demographics or past records.
IMMUTABLE_FEATURES = {
    "gender", "ssc_b", "hsc_b", "ssc_p", "hsc_p",
    # Stream/degree/specialisation/work-experience are also treated as fixed
    # context for the current application cycle.
    "hsc_s", "degree_t", "specialisation", "workex",
}

# Actionable attributes with scale-specific bounds and search steps.
ACTIONABLE_SPECS = {
    "cgpa":         {"bounds": (0.0, 10.0), "steps": [0.2, 0.5, 1.0], "integer": False},
    "soft_skills":  {"bounds": (0.0, 10.0), "steps": [0.2, 0.5, 1.0], "integer": False},
    "internships":  {"bounds": (0.0, 10.0), "steps": [1, 2], "integer": True},
    "projects":     {"bounds": (0.0, 20.0), "steps": [1, 2], "integer": True},
    "workshops":    {"bounds": (0.0, 20.0), "steps": [1, 2], "integer": True},
    "etest_p":      {"bounds": (0.0, 100.0), "steps": [2.0, 5.0, 8.0], "integer": False},
    "degree_p":     {"bounds": (0.0, 100.0), "steps": [2.0, 5.0, 8.0], "integer": False},
    "mba_p":        {"bounds": (0.0, 100.0), "steps": [2.0, 5.0, 8.0], "integer": False},
}


def _as_num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value, bounds):
    low, high = bounds
    return max(low, min(high, float(value)))


def _candidate_edits(work, direction):
    """Yield (column, proposed_value) for actionable attributes only."""
    for col, spec in ACTIONABLE_SPECS.items():
        current = _as_num(work.get(col))
        if current is None:
            # Missing optional field — do not fabricate a counterfactual.
            continue
        for step in spec["steps"]:
            candidate = _clamp(current + direction * step, spec["bounds"])
            if spec["integer"]:
                candidate = float(round(candidate))
            if abs(candidate - current) < 1e-9:
                continue
            yield col, candidate


def suggest(predictor, raw, target="Placed", max_edits=6):
    work = dict(raw)
    p_current, pred_current = predictor.probability(work)
    p_initial = p_current
    direction = 1 if target == "Placed" else -1

    if pred_current == target:
        return {
            "target": target,
            "already_true": True,
            "achieved": False,
            "from_probability": p_current,
            "to_probability": p_current,
            "steps": [],
            "final_prediction": pred_current,
            "actionable_features": sorted(ACTIONABLE_SPECS),
            "immutable_features": sorted(IMMUTABLE_FEATURES),
        }

    steps = []
    for _ in range(max_edits):
        best = None
        for col, proposed in _candidate_edits(work, direction):
            trial = dict(work)
            trial[col] = proposed
            p = predictor.probability(trial)[0]
            gain = (p - p_current) * direction
            if gain <= 1e-5:
                continue
            candidate = {
                "gain": gain,
                "col": col,
                "to_value": proposed,
                "p": p,
            }
            if best is None or candidate["gain"] > best["gain"] + 1e-9:
                best = candidate
        if best is None:
            break

        col = best["col"]
        specs = ACTIONABLE_SPECS[col]
        from_value = _clamp(_as_num(work[col]), specs["bounds"])
        if specs["integer"]:
            from_value = float(round(from_value))
        steps.append(
            {
                "feature": col,
                "from": from_value,
                "to": best["to_value"],
                "probability_before": round(p_current, 4),
                "probability_after": round(best["p"], 4),
            }
        )
        work[col] = best["to_value"]
        p_current = best["p"]
        if predictor.probability(work)[1] == target:
            break

    p_final, pred_final = predictor.probability(work)
    return {
        "target": target,
        "already_true": False,
        "achieved": pred_final == target,
        "from_probability": round(p_initial, 4),
        "to_probability": round(p_final, 4),
        "steps": steps,
        "final_prediction": pred_final,
        "actionable_features": sorted(ACTIONABLE_SPECS),
        "immutable_features": sorted(IMMUTABLE_FEATURES),
    }
