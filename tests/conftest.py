"""conftest.py — shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_benroshan_df():
    """Minimal 10-row DataFrame mimicking the Ben Roshan dataset."""
    return pd.DataFrame({
        "gender":        ["M", "F", "M", "F", "M", "F", "M", "M", "F", "M"],
        "tenth_p":       [67.0, 74.0, 55.0, 83.0, 60.0, 71.0, 90.0, 45.0, 78.0, 65.0],
        "tenth_b":       ["Others"] * 10,
        "twelfth_p":     [91.0, 68.0, 60.0, 77.0, 55.0, 85.0, 88.0, 50.0, 72.0, 69.0],
        "twelfth_b":     ["Central"] * 10,
        "twelfth_s":     ["Commerce", "Science", "Arts", "Commerce", "Science",
                          "Commerce", "Science", "Arts", "Commerce", "Science"],
        "degree_p":      [58.0, 65.0, 53.0, 70.0, 57.0, 62.0, 80.0, 48.0, 66.0, 59.0],
        "degree_t":      ["Sci&Tech", "Comm&Mgmt", "Others", "Sci&Tech", "Comm&Mgmt",
                          "Sci&Tech", "Sci&Tech", "Others", "Comm&Mgmt", "Sci&Tech"],
        "workex":        ["No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No"],
        "etest_p":       [55.0, 72.0, 48.0, 80.0, 50.0, 65.0, 85.0, 40.0, 70.0, 58.0],
        "specialisation":["Mkt&HR", "Mkt&Fin", "Mkt&HR", "Mkt&Fin", "Mkt&HR",
                          "Mkt&Fin", "Mkt&HR", "Mkt&HR", "Mkt&Fin", "Mkt&HR"],
        "mba_p":         [58.8, 66.0, 55.0, 72.0, 57.0, 61.0, 76.0, 50.0, 68.0, 60.0],
        "has_masters":   ["Yes", "Yes", "No", "Yes", "No",
                          "Yes", "Yes", "No", "Yes", "No"],
        "masters_type":  ["MBA", "MS", "None", "MBA", "None",
                          "Other", "MBA", "None", "MS", "None"],
        "status":        ["Placed", "Placed", "Not Placed", "Placed", "Not Placed",
                          "Placed", "Placed", "Not Placed", "Placed", "Not Placed"],
        "salary":        [270000.0, 300000.0, np.nan, 350000.0, np.nan,
                          280000.0, 400000.0, np.nan, 320000.0, np.nan],
    })


@pytest.fixture
def sample_kumbhar_df():
    """Minimal DataFrame mimicking the Kumbhar dataset schema (after normalisation)."""
    return pd.DataFrame({
        "cgpa":        [7.5, 8.0, 6.2, 8.5, 6.8, 7.1, 9.0, 5.5, 7.8, 6.6],
        "internships": [1, 2, 0, 3, 0, 1, 2, 0, 1, 0],
        "projects":    [2, 3, 1, 4, 1, 2, 4, 1, 3, 1],
        "workshops":   [2, 3, 1, 4, 1, 3, 5, 0, 2, 1],
        "aptitude_score": [70.0, 80.0, 55.0, 85.0, 60.0, 72.0, 90.0, 45.0, 78.0, 62.0],
        "soft_skills": [7.5, 8.0, 6.0, 8.5, 6.5, 7.0, 9.0, 5.5, 7.5, 6.5],
        "gender":      ["M", "F", "M", "F", "M", "F", "M", "M", "F", "M"],
        "workex":      ["No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No"],
        "status":      ["Placed", "Placed", "Not Placed", "Placed", "Not Placed",
                        "Placed", "Placed", "Not Placed", "Placed", "Not Placed"],
    })


@pytest.fixture
def predict_payload():
    """Valid prediction request payload (Ben Roshan schema)."""
    return {
        "gender": "M",
        "tenth_p": 67.0, "tenth_b": "Others",
        "twelfth_p": 91.0, "twelfth_b": "Others", "twelfth_s": "Commerce",
        "degree_p": 58.0, "degree_t": "Sci&Tech",
        "workex": "No",
        "etest_p": 55.0,
        "specialisation": "Mkt&HR",
        "mba_p": 58.8,
        "has_masters": "Yes", "masters_type": "MBA",
    }
