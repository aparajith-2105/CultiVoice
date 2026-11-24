"""Crop recommendation: model comparison, tuning, evaluation, explainability, robustness."""
import json, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             classification_report, top_k_accuracy_score)
from sklearn.inspection import permutation_importance
from common import *
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
OUT = RESULTS / "crop"

df = load_crop_rec()
FEATS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
le = LabelEncoder(); y = le.fit_transform(df["label"]); X = df[FEATS].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, random_state=SEED),
}

# 1) 5-fold CV on the TRAIN split only, then ONE evaluation on the untouched test split
rows = []
for name, m in models.items():
    cvr = cross_validate(m, Xtr, ytr, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1)
    m.fit(Xtr, ytr); p = m.predict(Xte)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
    rows.append({"Model": name, "CV accuracy": cvr["test_accuracy"].mean(), "CV std": cvr["test_accuracy"].std(),
                 "CV macro-F1": cvr["test_f1_macro"].mean(), "Test accuracy": accuracy_score(yte, p),
                 "Test precision": pr, "Test recall": rc, "Test F1": f1})
    print(f"{name:20s} CV {rows[-1]['CV accuracy']:.4f}+-{rows[-1]['CV std']:.4f}  test {rows[-1]['Test accuracy']:.4f}")
res = pd.DataFrame(rows).sort_values("CV accuracy", ascending=False)
res.round(4).to_csv(OUT / "model_comparison.csv", index=False)

# 2) Light hyper-parameter tuning (CV on train only) for the two leading families
rf_grid = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=-1),
    {"n_estimators": [50, 100, 200], "max_depth": [None, 12, 20], "min_samples_leaf": [1, 2, 4]}, cv=cv, scoring="accuracy")
rf_grid.fit(Xtr, ytr)
svm_grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
    {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.05, 0.2]}, cv=cv, scoring="accuracy")
svm_grid.fit(Xtr, ytr)
tuning = {"RandomForest": {"best_params": rf_grid.best_params_, "cv_accuracy": rf_grid.best_score_},
          "SVM": {"best_params": {k: (v if isinstance(v, (int, float)) else str(v)) for k, v in svm_grid.best_params_.items()}, "cv_accuracy": svm_grid.best_score_}}
print("tuning:", json.dumps(tuning, default=str))

# 3) Final model. Rule decided BEFORE looking at test results: prefer Random Forest if its CV accuracy is
#    within 0.5 points of the best model, because trees give probabilities, importances and a simple export.
best_name = res.iloc[0]["Model"]; best_cv = res.iloc[0]["CV accuracy"]
rf_cv = rf_grid.best_score_
final_name = "Random Forest (tuned)" if rf_cv >= best_cv - 0.005 else best_name
rf = rf_grid.best_estimator_
p = rf.predict(Xte); proba = rf.predict_proba(Xte)
pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro")
final = {"model": final_name, "params": rf_grid.best_params_, "test_accuracy": accuracy_score(yte, p),
         "test_precision_macro": pr, "test_recall_macro": rc, "test_f1_macro": f1,
         "top3_accuracy": top_k_accuracy_score(yte, proba, k=3, labels=np.arange(len(le.classes_))),
         "n_train": len(ytr), "n_test": len(yte), "cv_best_model": best_name}
print("FINAL:", json.dumps(final, indent=1, default=float))
pd.DataFrame(classification_report(yte, p, target_names=le.classes_, output_dict=True, zero_division=0)).T.round(3).to_csv(OUT / "per_class_report.csv")

# confusion matrix
cmx = confusion_matrix(yte, p)
fig, ax = plt.subplots(figsize=(10, 8.5))
sns.heatmap(cmx, annot=True, fmt="d", cmap="Greens", xticklabels=le.classes_, yticklabels=le.classes_, cbar=False, ax=ax, annot_kws={"size": 7})
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix: {final_name} (test set, n={len(yte)})", fontweight="bold")
plt.xticks(rotation=60); fig.tight_layout(); fig.savefig(OUT / "02_confusion_matrix.png"); plt.close(fig)
errs = [(le.classes_[a], le.classes_[b]) for a, b in zip(yte, p) if a != b]
final["misclassified"] = errs
# model comparison chart
fig, ax = plt.subplots(figsize=(8.5, 4.6))
r = res.sort_values("CV accuracy")
ax.barh(r["Model"], r["CV accuracy"] * 100, xerr=r["CV std"] * 100, color=GREEN, capsize=3)
for i, v in enumerate(r["Test accuracy"] * 100): ax.plot(v, i, "D", color=AMBER, ms=6)
ax.set_xlim(r["CV accuracy"].min() * 100 - 3, 100.6); ax.set_xlabel("Accuracy (%)  |  bars: 5-fold CV mean +/- std  |  diamonds: held-out test")
ax.set_title("Crop recommendation: model comparison", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "01_model_comparison.png"); plt.close(fig)

# 4) Explainability: permutation importance (test) + built-in importance
pi = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=SEED, scoring="accuracy")
imp = pd.DataFrame({"feature": FEATS, "perm_importance": pi.importances_mean, "perm_std": pi.importances_std,
                    "gini_importance": rf.feature_importances_}).sort_values("perm_importance", ascending=False)
imp.round(4).to_csv(OUT / "feature_importance.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
i2 = imp.sort_values("perm_importance")
ax[0].barh(i2.feature, i2.perm_importance * 100, xerr=i2.perm_std * 100, color=GREEN, capsize=3)
ax[0].set_xlabel("Accuracy drop when shuffled (percentage points)"); ax[0].set_title("Permutation importance (test set)", fontweight="bold")
i3 = imp.sort_values("gini_importance"); ax[1].barh(i3.feature, i3.gini_importance, color=AMBER)
ax[1].set_title("Built-in (Gini) importance", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "03_feature_importance.png"); plt.close(fig)

# 5) Robustness: noise on test inputs, and drop-one-feature ablation
rng = np.random.default_rng(SEED); sd = Xtr.std(0); levels = [0, .1, .2, .3, .5]
rob = {}
for name in ["Random Forest", "SVM (RBF)", "Logistic Regression", "Naive Bayes"]:
    m = models[name]; accs = []
    for lv in levels:
        a = [accuracy_score(yte, m.predict(Xte + rng.normal(0, 1, Xte.shape) * sd * lv)) for _ in range(10)]
        accs.append(np.mean(a))
    rob[name] = accs
rob_df = pd.DataFrame(rob, index=[f"{int(l*100)}%" for l in levels]); rob_df.index.name = "noise (x feature std)"
rob_df.round(4).to_csv(OUT / "robustness_noise.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for name, accs in rob.items(): ax.plot([l * 100 for l in levels], np.array(accs) * 100, marker="o", label=name)
ax.set_xlabel("Gaussian noise added to inputs (% of each feature's std)"); ax.set_ylabel("Test accuracy (%)")
ax.set_title("Robustness to measurement noise", fontweight="bold"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "04_robustness_noise.png"); plt.close(fig)

abl = []
base = cross_validate(RandomForestClassifier(**rf_grid.best_params_, random_state=SEED, n_jobs=-1), Xtr, ytr, cv=cv, scoring="accuracy")["test_score"].mean()
for j, f in enumerate(FEATS):
    keep = [k for k in range(len(FEATS)) if k != j]
    a = cross_validate(RandomForestClassifier(**rf_grid.best_params_, random_state=SEED, n_jobs=-1), Xtr[:, keep], ytr, cv=cv, scoring="accuracy")["test_score"].mean()
    abl.append({"dropped_feature": f, "cv_accuracy": a, "change_vs_full": a - base})
abl = pd.DataFrame(abl).sort_values("change_vs_full"); abl.round(4).to_csv(OUT / "ablation.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.barh(abl.dropped_feature, abl.change_vs_full * 100, color=AMBER); ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("Change in CV accuracy when the feature is removed (points)"); ax.set_title("Drop-one-feature ablation", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "05_ablation.png"); plt.close(fig)
final["full_cv_accuracy"] = base

# save model + metadata
joblib.dump({"model": rf, "label_encoder": le, "features": FEATS}, MODELS / "crop_rf.joblib")
json.dump({"final": final, "tuning": tuning}, open(OUT / "final_metrics.json", "w"), indent=1, default=float)
print("\nTop features:\n", imp.head(7).round(3).to_string(index=False))
print("\nRobustness:\n", rob_df.round(3).to_string())
print("\nAblation:\n", abl.round(4).to_string(index=False))
print("\nMisclassified:", errs)
