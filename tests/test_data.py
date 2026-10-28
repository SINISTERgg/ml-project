"""test_data.py — schema migration and masters-field synthesis tests."""

import pandas as pd

from src import config
from src.data import _rename_legacy_columns, add_masters_fields, load_data


class TestLegacyRename:
    def test_ssc_hsc_columns_are_renamed(self):
        df = pd.DataFrame({
            "ssc_p": [70.0], "ssc_b": ["Central"],
            "hsc_p": [75.0], "hsc_b": ["Others"], "hsc_s": ["Science"],
        })
        out = _rename_legacy_columns(df)
        assert {"tenth_p", "tenth_b", "twelfth_p", "twelfth_b", "twelfth_s"} <= set(out.columns)
        assert not any(c.startswith(("ssc_", "hsc_")) for c in out.columns)

    def test_existing_new_columns_take_precedence(self):
        df = pd.DataFrame({"ssc_p": [70.0], "tenth_p": [80.0]})
        out = _rename_legacy_columns(df)
        assert out["tenth_p"].iloc[0] == 80.0


class TestMastersFields:
    def test_fields_are_added_when_absent(self):
        df = pd.DataFrame({"mba_p": [70.0, 65.0, None, None]})
        out = add_masters_fields(df, seed=42)
        assert set(out["has_masters"]).issubset({"Yes", "No"})
        assert set(out["masters_type"]).issubset({"None", "MBA", "MS", "Other"})

    def test_mba_cohort_always_has_masters(self):
        df = pd.DataFrame({"mba_p": [70.0, 65.0, 60.0]})
        out = add_masters_fields(df, seed=42)
        assert (out["has_masters"] == "Yes").all()
        assert (out["masters_type"] != "None").all()

    def test_no_masters_has_none_type(self):
        df = pd.DataFrame({
            "mba_p": [None, None],
            "has_masters": ["No", "Yes"],
            "masters_type": ["MBA", None],
        })
        out = add_masters_fields(df, seed=42)
        assert out.loc[0, "masters_type"] == "None"
        assert out.loc[1, "masters_type"] != "None"

    def test_synthesis_is_deterministic(self):
        df = pd.DataFrame({"mba_p": [None] * 20})
        a = add_masters_fields(df, seed=7)
        b = add_masters_fields(df, seed=7)
        pd.testing.assert_frame_equal(a, b)


class TestCacheRead:
    def test_none_masters_type_survives_csv_roundtrip(self, tmp_path, monkeypatch):
        row = {c: 0 for c in config.BENROSHAN_NUMERIC + config.BENROSHAN_CATEGORICAL}
        row.update({
            "mba_p": 0.0,
            "has_masters": "No",
            "masters_type": "None",
            config.TARGET: "Not Placed",
        })
        path = tmp_path / "merged.csv"
        pd.DataFrame([row]).to_csv(path, index=False)
        monkeypatch.setattr(config, "DATA_PATH", path)

        out = load_data()
        assert out.loc[0, "masters_type"] == "None"
