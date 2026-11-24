"""Yield prediction (log scale), honest validation (random AND future-year split), what-if analysis."""
import json, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from common import *
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
OUT = RESULTS / "yield"

df = load_yield()
df["logArea"] = np.log1p(df["Area"])
CAT = ["Crop", "Season", "State"]
NUM = ["Crop_Year", "logArea", "Annual_Rainfall", "Fert_per_ha", "Pest_per_ha"]
FEATS = CAT + NUM
# NOTE: Production is deliberately NOT a feature: Yield = Production / Area, so it would leak the answer.
y = np.log1p(df["Yield"].values)

def ord_enc(): return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
def pre_tree(): return ColumnTransformer([("c", ord_enc(), CAT)], remainder="passthrough")
def pre_lin(): return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT), ("n", StandardScaler(), NUM)])

class CropMedian:  # naive baseline: predict the crop's typical yield
    def fit(self, X, y): self.m = pd.Series(y, index=X["Crop"].values).groupby(level=0).median(); self.g = np.median(y); return self
    def predict(self, X): return X["Crop"].map(self.m).fillna(self.g).values

def models():
    return {
        "Baseline: crop median": CropMedian(),
        "Ridge regression": Pipeline([("p", pre_lin()), ("m", Ridge(alpha=1.0))]),
        "Random Forest": Pipeline([("p", pre_tree()), ("m", RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("p", pre_tree()), ("m", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=SEED,
                              categorical_features=[0, 1, 2]))]),
    }

def score(m, Xa, ya, Xb, yb):
    m.fit(Xa, ya); p = m.predict(Xb)
    yt, yp = np.expm1(yb), np.expm1(p)
    return {"R2 (log scale)": r2_score(yb, p), "RMSE (t/ha)": mean_squared_error(yt, yp) ** .5,
            "MAE (t/ha)": mean_absolute_error(yt, yp), "Median abs % error": float(np.median(np.abs(yp - yt) / yt) * 100)}, p

X = df[FEATS]
splits = {}
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
splits["Random 80/20"] = (itr, ite)
tr = np.where(df.Crop_Year <= 2015)[0]; te = np.where(df.Crop_Year >= 2016)[0]
splits["Future years (train <=2015, test 2016-2020)"] = (tr, te)
print({k: (len(a), len(b)) for k, (a, b) in splits.items()})

rows = []; preds = {}
for sname, (a, b) in splits.items():
    for mname, m in models().items():
        Xa, Xb = X.iloc[a], X.iloc[b]
        if mname.startswith("Baseline"): Xa, Xb = df.iloc[a], df.iloc[b]
        met, p = score(m, Xa, y[a], Xb, y[b]); preds[(sname, mname)] = (b, p)
        rows.append({"Split": sname, "Model": mname, **met})
res = pd.DataFrame(rows); res.round(3).to_csv(OUT / "yield_model_comparison.csv", index=False)
print(res.round(3).to_string(index=False))

# chart: R2 per model per split
fig, ax = plt.subplots(figsize=(8.5, 4.4)); w = .38
names = list(models().keys())
for i, (s, c) in enumerate(zip(splits, [GREEN, AMBER])):
    v = [res[(res.Split == s) & (res.Model == n)]["R2 (log scale)"].iloc[0] for n in names]
    ax.bar(np.arange(len(names)) + i * w, v, w, label=s, color=c)
ax.set_xticks(np.arange(len(names)) + w / 2); ax.set_xticklabels([n.replace("Baseline: ", "Baseline:\n") for n in names], fontsize=8.5)
ax.set_ylabel("R-squared (log yield)"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8); ax.set_title("Yield prediction: model comparison", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "01_yield_model_comparison.png"); plt.close(fig)

# choose final by FUTURE-YEAR R2 among tree models (harder, more honest split)
fy = res[res.Split.str.startswith("Future") & ~res.Model.str.startswith("Baseline")].sort_values("R2 (log scale)", ascending=False)
best = fy.iloc[0]["Model"]; print("best on future years:", best)

# predicted vs actual (future years, best model)
b, p = preds[(list(splits)[1], best)]
fig, ax = plt.subplots(figsize=(5.4, 5.2)); yt = np.expm1(y[b]); yp = np.expm1(p)
ax.scatter(yt, yp, s=6, alpha=.35, color=GREEN); lim = [0.05, 120]
ax.plot(lim, lim, color=AMBER, lw=1.2); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual yield (t/ha)"); ax.set_ylabel("Predicted yield (t/ha)"); ax.set_title(f"{best}: predicted vs actual (2016-2020)", fontweight="bold", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "02_pred_vs_actual.png"); plt.close(fig)

# final model: refit on ALL data for the app; permutation importance measured on the future-year test
a, b = splits[list(splits)[1]]
mi = models()[best]; mi.fit(X.iloc[a], y[a])
pi = permutation_importance(mi, X.iloc[b], y[b], n_repeats=10, random_state=SEED, scoring="r2")
imp = pd.DataFrame({"feature": FEATS, "r2_drop": pi.importances_mean, "std": pi.importances_std}).sort_values("r2_drop")
imp.round(4).sort_values("r2_drop", ascending=False).to_csv(OUT / "yield_feature_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(imp.feature, imp.r2_drop, xerr=imp["std"], color=GREEN, capsize=3)
ax.set_xlabel("Drop in R-squared when shuffled (future-year test)"); ax.set_title("What drives predicted yield", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "03_yield_importance.png"); plt.close(fig)

final_model = models()[best].fit(X, y)
joblib.dump({"model": final_model, "features": FEATS, "best": best}, MODELS / "yield_model.joblib")

# ---- What-if (associations learned from state-year data, NOT causal). Reference scenario = a typical row.
def scenario(crop, state, season, area=5000.0, year=2019):
    base = df[(df.Crop == crop) & (df.State == state)]
    ref = base[base.Crop_Year == base.Crop_Year.max()].iloc[0] if len(base) else df[df.Crop == crop].iloc[0]
    return pd.DataFrame([{"Crop": crop, "Season": season, "State": state, "Crop_Year": year, "logArea": np.log1p(area),
        "Annual_Rainfall": ref.Annual_Rainfall, "Fert_per_ha": ref.Fert_per_ha, "Pest_per_ha": ref.Pest_per_ha}])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); whatif = []
for crop, state, season in [("Rice", "Punjab", "Kharif"), ("Wheat", "Uttar Pradesh", "Rabi"), ("Maize", "Karnataka", "Kharif")]:
    base = scenario(crop, state, season); y0 = float(np.expm1(final_model.predict(base))[0])
    rr, ff = [], []
    for ch in np.arange(-40, 41, 10):
        s = base.copy(); s["Annual_Rainfall"] *= 1 + ch / 100; rr.append(float(np.expm1(final_model.predict(s))[0]) / y0 * 100 - 100)
        s = base.copy(); s["Fert_per_ha"] *= 1 + ch / 100; ff.append(float(np.expm1(final_model.predict(s))[0]) / y0 * 100 - 100)
    ax[0].plot(np.arange(-40, 41, 10), rr, marker="o", label=f"{crop} ({state})"); ax[1].plot(np.arange(-40, 41, 10), ff, marker="o", label=f"{crop} ({state})")
    whatif.append({"crop": crop, "state": state, "baseline_yield": y0, "rain_-20%": rr[2], "rain_+20%": rr[6], "fert_-20%": ff[2], "fert_+20%": ff[6]})
for a_, t in zip(ax, ["Change in rainfall", "Change in fertilizer per hectare"]):
    a_.axhline(0, color="k", lw=.7); a_.set_xlabel(f"{t} (%)"); a_.set_ylabel("Change in predicted yield (%)"); a_.legend(frameon=False, fontsize=8)
fig.suptitle("What-if analysis (associations in historical state-year data, not proof of cause)", fontweight="bold", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "04_whatif.png"); plt.close(fig)
pd.DataFrame(whatif).round(2).to_csv(OUT / "whatif_summary.csv", index=False); print(pd.DataFrame(whatif).round(2).to_string(index=False))
json.dump({"best": best, "n_rows": len(df), "split_sizes": {k: [len(a), len(b)] for k, (a, b) in splits.items()}}, open(OUT / "yield_meta.json", "w"), indent=1)
