# -*- coding: utf-8 -*-
"""
Script to generate the comprehensive Academic Minor Project Report:
1. reports/project_report.md
2. reports/project_report.html
"""

import os
import json
import base64

WORKSPACE_DIR = r"c:\Users\M S\Desktop\ml project"
REPORTS_DIR = os.path.join(WORKSPACE_DIR, "reports")
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")
EDA_PLOTS_DIR = os.path.join(PLOTS_DIR, "eda")

os.makedirs(REPORTS_DIR, exist_ok=True)

# Load metadata
with open(os.path.join(MODELS_DIR, "meta.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

print("Loaded metadata successfully.")

