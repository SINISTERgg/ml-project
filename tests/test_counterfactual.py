from src.counterfactual import ACTIONABLE_SPECS, IMMUTABLE_FEATURES, suggest


class DummyPredictor:
    def __init__(self):
        self.threshold = 0.5

    def probability(self, raw):
        # Simple heuristic: higher etest_p and internships increase chance
        p = 0.2
        p += float(raw.get("etest_p", 0)) * 0.004
        p += float(raw.get("internships") or 0) * 0.05
        p += float(raw.get("projects") or 0) * 0.03
        p = max(0.01, min(0.99, p))
        pred = "Placed" if p >= 0.5 else "Not Placed"
        return p, pred


class TestCounterfactual:
    def test_none_optional_fields_do_not_crash(self):
        pred = DummyPredictor()
        raw = {
            "gender": "M",
            "ssc_p": 60,
            "ssc_b": "Others",
            "hsc_p": 60,
            "hsc_b": "Others",
            "hsc_s": "Commerce",
            "degree_p": 55,
            "degree_t": "Comm&Mgmt",
            "workex": "No",
            "etest_p": 40,
            "specialisation": "Mkt&HR",
            "mba_p": 50,
            "cgpa": None,
            "internships": None,
            "projects": None,
            "workshops": None,
            "aptitude_score": None,
            "soft_skills": None,
        }
        result = suggest(pred, raw, target="Placed", max_edits=6)
        assert "achieved" in result
        assert result["actionable_features"]
        # Immutable and actionable sets must be disjoint
        assert not IMMUTABLE_FEATURES.intersection(ACTIONABLE_SPECS)

    def test_immutable_features_never_suggested(self):
        pred = DummyPredictor()
        raw = {
            "gender": "F",
            "ssc_p": 45,
            "ssc_b": "Central",
            "hsc_p": 48,
            "hsc_b": "Central",
            "hsc_s": "Arts",
            "degree_p": 50,
            "degree_t": "Others",
            "workex": "No",
            "etest_p": 35,
            "specialisation": "Mkt&HR",
            "mba_p": 48,
            "internships": 0,
            "projects": 0,
            "workshops": 0,
        }
        result = suggest(pred, raw, target="Placed", max_edits=4)
        for step in result["steps"]:
            assert step["feature"] not in IMMUTABLE_FEATURES

    def test_already_true_short_circuits(self):
        pred = DummyPredictor()
        raw = {
            "gender": "M",
            "ssc_p": 80,
            "ssc_b": "Central",
            "hsc_p": 85,
            "hsc_b": "Central",
            "hsc_s": "Science",
            "degree_p": 82,
            "degree_t": "Sci&Tech",
            "workex": "Yes",
            "etest_p": 90,
            "specialisation": "Mkt&Fin",
            "mba_p": 78,
            "internships": 3,
            "projects": 4,
        }
        result = suggest(pred, raw, target="Placed", max_edits=6)
        assert result["already_true"] is True
        assert result["steps"] == []
