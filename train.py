"""
train.py - trains and evaluates the Random Forest phishing URL classifier.

Usage (local):   python train.py path/to/dataset.csv
On Kaggle the dataset is found automatically under /kaggle/input.

Recommended dataset (2020 or later): "Web page Phishing Detection Dataset"
(Hannousse & Yahiouche, 2021): 11,430 URLs, 50% phishing / 50% legitimate,
columns include 'url' and 'status' (legitimate / phishing).

Outputs: model.joblib, results.json, confusion_matrix.png, feature_importance.png
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
from features import FEATURE_NAMES, feature_vector, rule_analyse

RANDOM_STATE = 42
N_TREES = 100
TEST_SIZE = 0.20
OUT = os.environ.get("OUT_DIR", ".")


def find_dataset():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for p in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        return p
    raise SystemExit("No dataset found. Pass the CSV path as an argument.")


def load(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    url_col = cols.get("url") or cols.get("urls") or cols.get("domain")
    lab_col = next((cols[c] for c in ("status", "label", "type", "result", "class") if c in cols), None)
    if url_col is None or lab_col is None:
        raise SystemExit("Could not find URL / label columns in %s" % list(df.columns))
    df = df[[url_col, lab_col]].rename(columns={url_col: "url", lab_col: "label"}).dropna()

    def to01(v):
        s = str(v).strip().lower()
        if s in ("phishing", "phish", "bad", "malicious", "1"):
            return 1
        if s in ("legitimate", "good", "benign", "0", "safe"):
            return 0
        return None
    df["y"] = df["label"].map(to01)
    df = df.dropna(subset=["y"]).astype({"y": int})
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""].drop_duplicates(subset="url")
    return df


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y, p) * 100, 2),
        "precision": round(precision_score(y, p, zero_division=0) * 100, 2),
        "recall": round(recall_score(y, p, zero_division=0) * 100, 2),
        "f1": round(f1_score(y, p, zero_division=0) * 100, 2),
        "false_positive_rate": round(fp / (fp + tn) * 100, 2) if (fp + tn) else 0.0,
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
    }


def main():
    path = find_dataset()
    print("Dataset:", path)
    df = load(path)
    print("Usable URLs: %d  (phishing %d, legitimate %d)" % (len(df), df.y.sum(), (df.y == 0).sum()))

    rows, keep = [], []
    for i, u in enumerate(df["url"]):
        try:
            rows.append(feature_vector(u)); keep.append(i)
        except ValueError:
            pass
    df = df.iloc[keep].reset_index(drop=True)
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = df["y"].values

    X_tr, X_te, y_tr, y_te, u_tr, u_te = train_test_split(
        X, y, df["url"], test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    model = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    t0 = time.time(); model.fit(X_tr, y_tr); train_s = time.time() - t0
    t0 = time.time(); p_rf = model.predict(X_te); pred_ms = (time.time() - t0) / len(X_te) * 1000
    prob = model.predict_proba(X_te)[:, 1]

    p_rule = np.array([int(rule_analyse(u)[1]) for u in u_te])
    p_hyb = np.maximum(p_rf, p_rule)          # deployed fusion: phishing if EITHER flags it

    cv = cross_val_score(RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1),
                         X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="accuracy")

    res = {
        "dataset": os.path.basename(path),
        "total_urls": int(len(df)), "phishing": int(y.sum()), "legitimate": int((y == 0).sum()),
        "train_size": int(len(y_tr)), "test_size": int(len(y_te)),
        "n_features": len(FEATURE_NAMES), "n_trees": N_TREES,
        "random_forest": metrics(y_te, p_rf),
        "auc": round(roc_auc_score(y_te, prob), 4),
        "rule_based_only": metrics(y_te, p_rule),
        "hybrid_either": metrics(y_te, p_hyb),
        "cv5_accuracy_mean": round(cv.mean() * 100, 2), "cv5_accuracy_std": round(cv.std() * 100, 2),
        "training_seconds": round(train_s, 2), "prediction_ms_per_url": round(pred_ms, 4),
        "feature_importance": dict(sorted(
            {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, model.feature_importances_)}.items(),
            key=lambda kv: -kv[1])),
    }

    test_urls = [
        "https://chatgpt.com/codex/get-started/", "http://ux.pu_you-456", "http://ux.pu-you-456",
        "https://www.tripwire.com/state-of-security/evolution-phishing-attacks-why-traditional-detection-methods-are-failing",
        "https://forms.gle/7TZjDVSotiab6LG28",
        "https://www.ijraset.com/research-paper/ai-based-student-performance-prediction-system",
    ]
    res["interface_urls"] = []
    for u in test_urls:
        s, rp, _ = rule_analyse(u)
        mp = int(model.predict(pd.DataFrame([feature_vector(u)], columns=FEATURE_NAMES))[0])
        res["interface_urls"].append({"url": u, "rule_score": s, "rule": int(rp), "model": mp,
                                      "final": int(rp or mp)})

    joblib.dump(model, os.path.join(OUT, "model.joblib"))
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    # Figures (black and white)
    cm = np.array([[res["random_forest"]["TP"], res["random_forest"]["FN"]],
                   [res["random_forest"]["FP"], res["random_forest"]["TN"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Greys", alpha=0.35)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Phishing", "Legitimate"]); ax.set_xlabel("Predicted")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Phishing", "Legitimate"]); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=200); plt.close()

    fi = res["feature_importance"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(fi.keys())[::-1], list(fi.values())[::-1], color="black")
    ax.set_xlabel("Importance (mean decrease in Gini impurity)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=200); plt.close()

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
