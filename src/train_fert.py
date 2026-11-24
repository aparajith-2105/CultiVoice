"""Fertilizer recommendation on a very small dataset (99 rows): repeated CV, honest reporting."""
import json, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from common import *
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
OUT = RESULTS / "fert"
df = load_fert()
CAT = ["Soil Type", "Crop Type"]; NUMS = {"all": ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorus"],
                                       "nutrients only": ["Nitrogen", "Potassium", "Phosphorus"]}
y = df["Fertilizer Name"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
def prep(num, cats, scale): return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cats), ("n", StandardScaler() if scale else "passthrough", num)])
rows = []
for fs_name, num in NUMS.items():
    cats = CAT if fs_name == "all" else []
    for mname, est, scale in [("Majority-class baseline", DummyClassifier(strategy="most_frequent"), False),
                              ("Logistic Regression", LogisticRegression(max_iter=3000), True),
                              ("KNN", KNeighborsClassifier(5), True),
                              ("Decision Tree", DecisionTreeClassifier(max_depth=4, random_state=SEED), False),
                              ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), False)]:
        pipe = Pipeline([("p", prep(num, cats, scale)), ("m", est)])
        r = cross_validate(pipe, df[num + cats], y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1)
        rows.append({"Features": fs_name, "Model": mname, "CV accuracy": r["test_accuracy"].mean(), "Std": r["test_accuracy"].std(), "CV macro-F1": r["test_f1_macro"].mean()})
res = pd.DataFrame(rows); res.round(3).to_csv(OUT / "fertilizer_model_comparison.csv", index=False)
print(res.round(3).to_string(index=False))

# choose final: best CV accuracy; models within 1 point count as tied and the simpler one wins
ESTS = {"Logistic Regression": (LogisticRegression(max_iter=3000), True), "Decision Tree": (DecisionTreeClassifier(max_depth=4, random_state=SEED), False),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), False), "KNN": (KNeighborsClassifier(5), True)}
SIMPLICITY = ["Logistic Regression", "Decision Tree", "KNN", "Random Forest"]
top = res[res.Model != "Majority-class baseline"].sort_values("CV accuracy", ascending=False)
best_acc = top.iloc[0]["CV accuracy"]
cand = top[top["CV accuracy"] >= best_acc - 0.01]
choice = sorted(cand.to_dict("records"), key=lambda r: (len(NUMS[r["Features"]]) + (2 if r["Features"] == "all" else 0), SIMPLICITY.index(r["Model"])))[0]
print("chosen:", choice["Features"], choice["Model"], round(choice["CV accuracy"], 3))
fs = choice["Features"]; num = NUMS[fs]; cats = CAT if fs == "all" else []
est, scale = ESTS[choice["Model"]]
from sklearn.base import clone
final = Pipeline([("p", prep(num, cats, scale)), ("m", clone(est))]).fit(df[num + cats], y)
from sklearn.model_selection import StratifiedKFold
pred = cross_val_predict(Pipeline([("p", prep(num, cats, scale)), ("m", clone(est))]), df[num + cats], y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
labels = sorted(y.unique()); cm = confusion_matrix(y, pred, labels=labels)
# readable rules from a depth-limited tree on nutrients, for the report
dtr = Pipeline([("p", prep(NUMS["nutrients only"], [], False)), ("m", DecisionTreeClassifier(max_depth=6, random_state=SEED))]).fit(df[NUMS["nutrients only"]], y)
(OUT / "decision_tree_rules.txt").write_text(export_text(dtr.named_steps["m"], feature_names=NUMS["nutrients only"]))
fig, ax = plt.subplots(figsize=(6.2, 5.2)); sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Fertilizer ({choice['Model']}): 5-fold CV predictions", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "01_fert_confusion.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4.6)); s2 = res[res.Features == "nutrients only"].sort_values("CV accuracy"); s3 = res[res.Features == "all"].set_index("Model").loc[s2.Model]
yy = np.arange(len(s2)); ax.barh(yy + .2, s2["CV accuracy"] * 100, .4, xerr=s2["Std"] * 100, color=GREEN, capsize=3, label="N, P, K only")
ax.barh(yy - .2, s3["CV accuracy"].values * 100, .4, xerr=s3["Std"].values * 100, color=AMBER, capsize=3, label="+ crop, soil, climate")
ax.set_yticks(yy); ax.set_yticklabels(s2.Model); ax.set_xlabel("CV accuracy (%), 10 x 5-fold"); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Fertilizer models (99 rows): extra features do not help", fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "02_fert_model_comparison.png"); plt.close(fig)
joblib.dump({"model": final, "num": num, "cats": cats, "labels": labels, "name": choice["Model"]}, MODELS / "fert_model.joblib")
json.dump({"features": fs, "model": choice["Model"], "cv_accuracy": float(choice["CV accuracy"]), "n_rows": len(df)}, open(OUT / "fert_meta.json", "w"), indent=1)
print("single 5-fold accuracy:", round(accuracy_score(y, pred), 3))
