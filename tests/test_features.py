"""test_features.py — unit tests for src/features.py"""

import pandas as pd

from src.features import add_feature_engineering, build_preprocessor, feature_names


class TestFeatureEngineering:
    def test_academic_avg_computed(self, sample_benroshan_df):
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        out = add_feature_engineering(df)
        assert "academic_avg" in out.columns
        assert out["academic_avg"].notna().all()

    def test_performance_gap_non_negative(self, sample_benroshan_df):
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        out = add_feature_engineering(df)
        assert (out["performance_gap"] >= 0).all()

    def test_stem_flag_binary(self, sample_benroshan_df):
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        out = add_feature_engineering(df)
        assert set(out["stem_flag"].unique()).issubset({0, 1})

    def test_activity_score_zero_without_kumbhar(self, sample_benroshan_df):
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        out = add_feature_engineering(df)
        # No kumbhar cols → activity_score should be 0
        assert (out["activity_score"] == 0).all()

    def test_activity_score_with_kumbhar(self, sample_kumbhar_df):
        df = sample_kumbhar_df.drop(columns=["status"])
        out = add_feature_engineering(df)
        expected = df["internships"] + df["projects"] + df["workshops"]
        pd.testing.assert_series_equal(
            out["activity_score"].reset_index(drop=True).astype(float),
            expected.reset_index(drop=True).astype(float),
            check_names=False,
        )

    def test_skills_composite_range(self, sample_kumbhar_df):
        df = sample_kumbhar_df.drop(columns=["status"])
        out = add_feature_engineering(df)
        assert out["skills_composite"].between(0, 100).all()

    def test_engineering_preserves_rows(self, sample_benroshan_df):
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        out = add_feature_engineering(df)
        assert len(out) == len(df)

    def test_engineering_is_non_destructive(self, sample_benroshan_df):
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        original_cols = set(df.columns)
        _ = add_feature_engineering(df)
        assert set(df.columns) == original_cols  # original df unchanged


class TestPreprocessor:
    def test_fit_transform_shape(self, sample_benroshan_df):
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        X = add_feature_engineering(df)
        pp = build_preprocessor(list(X.columns))
        Xt = pp.fit_transform(X)
        assert Xt.shape[0] == len(df)
        assert Xt.shape[1] > 0

    def test_feature_names_match_output(self, sample_benroshan_df):
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        X = add_feature_engineering(df)
        pp = build_preprocessor(list(X.columns))
        pp.fit(X)
        names = feature_names(pp)
        assert len(names) == pp.transform(X).shape[1]

    def test_handles_missing_kumbhar_cols(self, sample_benroshan_df):
        """Preprocessor built without Kumbhar cols should still work."""
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        X = add_feature_engineering(df)
        # Build without optional Kumbhar cols — should not raise
        pp = build_preprocessor(list(X.columns))
        Xt = pp.fit_transform(X)
        assert Xt.shape[0] == len(df)

    def test_transform_single_row(self, sample_benroshan_df):
        df = sample_benroshan_df.drop(columns=["status", "salary"])
        X = add_feature_engineering(df)
        pp = build_preprocessor(list(X.columns))
        pp.fit(X)
        single = X.iloc[[0]]
        Xt = pp.transform(single)
        assert Xt.shape[0] == 1
