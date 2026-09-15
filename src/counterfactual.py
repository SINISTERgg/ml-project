from . import config
from .settings import DEFAULT_CATEGORY_OPTIONS

NUMERIC_STEPS = [2, 4, 6, 8]


def _clamp(value):
    return max(0.0, min(100.0, float(value)))


def _as_num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _candidate_edits(work):
    edits = []
    for col in config.NUMERIC_FEATURES:
        for step in NUMERIC_STEPS:
            edits.append(("numeric", col, step))
    for col in config.CATEGORICAL_FEATURES:
        for value in DEFAULT_CATEGORY_OPTIONS.get(col, []):
            edits.append(("categorical", col, value))
    return edits


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
        }

    steps = []
    for _ in range(max_edits):
        best = None
        for kind, col, delta in _candidate_edits(work):
            trial = dict(work)
            if kind == "numeric":
                trial[col] = _clamp(trial[col] + direction * delta)
            else:
                if trial.get(col) == delta:
                    continue
                trial[col] = delta
            p = predictor.probability(trial)[0]
            gain = (p - p_current) * direction
            if gain <= 1e-5:
                continue
            candidate = {
                "gain": gain,
                "kind": kind,
                "col": col,
                "to_value": trial[col],
                "p": p,
            }
            if best is None or (candidate["gain"] > best["gain"] + 1e-9):
                best = candidate
        if best is None:
            break
        steps.append(
            {
                "feature": best["col"],
                "from": _as_num(work[best["col"]]),
                "to": _as_num(best["to_value"]),
                "probability_before": round(p_current, 4),
                "probability_after": round(best["p"], 4),
            }
        )
        work[best["col"]] = best["to_value"]
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
    }