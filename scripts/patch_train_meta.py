path = "src/train.py"
with open(path, encoding="utf-8") as f:
    s = f.read()

old = '''        "cv_roc_auc_mean": best_row["cv_roc_auc_mean"],
        "cv_roc_auc_std": best_row["cv_roc_auc_std"],
        "study_scores": study_scores,
        "tuned_params": algo_params,'''
new = '''        "cv_roc_auc_mean": best_row["cv_roc_auc_mean"],
        "cv_roc_auc_std": best_row["cv_roc_auc_std"],
        "cv_brier_mean": best_row.get("cv_brier_mean"),
        "cv_f1_mean": best_row.get("cv_f1_mean"),
        "study_scores": study_scores,
        "optuna_param_importances": study_importances,
        "tuned_params": algo_params,'''

if old not in s:
    print("MISS")
else:
    s2 = s.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s2)
    print("OK")
