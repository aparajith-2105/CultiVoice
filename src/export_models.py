"""Export the three trained models + app metadata to JSON for the web frontend."""
import json
import numpy as np, pandas as pd, joblib
from common import *

OUT = ROOT / "frontend" / "model"; OUT.mkdir(parents=True, exist_ok=True)
def dump(obj, name, indent=None):
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":") if indent is None else None, indent=indent, default=float))
    print(f"{name}: {(OUT/name).stat().st_size/1024:.0f} KB")

# ---------- 1. Crop Random Forest ----------
bundle = joblib.load(MODELS / "crop_rf.joblib"); rf, le, FEATS = bundle["model"], bundle["label_encoder"], bundle["features"]
trees = []
for est in rf.estimators_:
    t = est.tree_; val = t.value[:, 0, :]; val = val / val.sum(1, keepdims=True)
    leaf = t.children_left == -1
    trees.append({"f": t.feature.tolist(), "t": [float(x) for x in t.threshold], "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                  "v": [[round(float(p), 5) for p in val[i]] if leaf[i] else 0 for i in range(len(leaf))]})
crop_df = load_crop_rec()
profiles = {}
for c, g in crop_df.groupby("label"):
    profiles[c] = {f: {"mean": round(float(g[f].mean()), 3), "p10": round(float(g[f].quantile(.1)), 3), "p90": round(float(g[f].quantile(.9)), 3),
                       "min": round(float(g[f].min()), 3), "max": round(float(g[f].max()), 3)} for f in FEATS}
feat_stats = {f: {"min": round(float(crop_df[f].min()), 3), "max": round(float(crop_df[f].max()), 3), "mean": round(float(crop_df[f].mean()), 3),
                  "std": round(float(crop_df[f].std()), 3)} for f in FEATS}
dump({"classes": list(le.classes_), "features": FEATS, "trees": trees}, "crop_rf.json")

# ---------- 2. Yield gradient boosting (categorical splits as left-category lists) ----------
yb = joblib.load(MODELS / "yield_model.joblib"); pipe = yb["model"]; gb = pipe.named_steps["m"]
oe = pipe.named_steps["p"].transformers_[0][1]
cats = {c: [str(x) for x in cl] for c, cl in zip(["Crop", "Season", "State"], oe.categories_)}
ytrees = []
for pr in gb._predictors:
    p = pr[0]; n = p.nodes; f, th, L, R, V = [], [], [], [], []
    for i in range(len(n)):
        if n["is_leaf"][i]: f.append(-1); th.append(0); V.append(float(n["value"][i]))
        else:
            f.append(int(n["feature_idx"][i])); V.append(0)
            if n["is_categorical"][i]:
                words = p.raw_left_cat_bitsets[n["bitset_idx"][i]]
                th.append([c for c in range(256) if (int(words[c >> 5]) >> (c & 31)) & 1])
            else: th.append(float(n["num_threshold"][i]))
        L.append(int(n["left"][i])); R.append(int(n["right"][i]))
    ytrees.append({"f": f, "t": th, "l": L, "r": R, "v": V})
dump({"baseline": float(gb._baseline_prediction.ravel()[0]), "cats": cats, "num_order": ["Crop_Year", "logArea", "Annual_Rainfall", "Fert_per_ha", "Pest_per_ha"],
      "trees": ytrees, "transform": "expm1"}, "yield_gbm.json")

# yield app metadata
yd = load_yield()
last = yd.groupby("State")["Crop_Year"].max()
state_defaults = {}
for s, g in yd.groupby("State"):
    r = g[g.Crop_Year == last[s]].iloc[0]
    state_defaults[s] = {"rainfall": round(float(r.Annual_Rainfall), 1), "fert_per_ha": round(float(r.Fert_per_ha), 2), "pest_per_ha": round(float(r.Pest_per_ha), 4),
                         "year": int(last[s]),
                         "crops": {c: {"seasons": sorted(gg.Season.unique().tolist()), "median_area": round(float(gg.Area.median()), 0),
                                       "median_yield": round(float(gg.Yield.median()), 3), "n": int(len(gg))} for c, gg in g.groupby("Crop")}}
dump(state_defaults, "yield_defaults.json")

# ---------- 3. Fertilizer logistic regression ----------
fb = joblib.load(MODELS / "fert_model.joblib"); fp = fb["model"]
assert fb["name"] == "Logistic Regression" and fb["cats"] == [], "export expects the nutrient-only LR"
sc = fp.named_steps["p"].named_transformers_["n"]; lr = fp.named_steps["m"]
fert_comp = {"Urea": "46-0-0", "DAP": "18-46-0", "14-35-14": "14-35-14", "28-28": "28-28-0", "20-20": "20-20-0", "17-17-17": "17-17-17", "10-26-26": "10-26-26"}
dump({"features": fb["num"], "classes": [str(c) for c in lr.classes_], "mean": sc.mean_.tolist() if hasattr(sc, "mean_") else None,
      "scale": sc.scale_.tolist() if hasattr(sc, "scale_") else None, "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(), "composition_NPK": fert_comp}, "fert_lr.json")
fd = load_fert(); fert_stats = {f: {"min": float(fd[f].min()), "max": float(fd[f].max())} for f in fb["num"]}

# ---------- 4. Price snapshot + crop-name mapping ----------
pr = pd.read_csv(DATA / "price_clean.csv")
price_map = {"rice": "Paddy(Common)", "maize": "Maize", "chickpea": "Bengal Gram(Gram)(Whole)", "blackgram": "Black Gram(Urd Beans)(Whole)",
             "mungbean": "Green Gram(Moong)(Whole)", "lentil": "Lentil(Masur)(Whole)", "pigeonpeas": "Red gram/Arhar/Tur(whole)", "cotton": "Cotton", "jute": "Jute"}
prices = {}
for k, v in price_map.items():
    r = pr[pr.Commodity == v]
    if len(r):
        r = r.iloc[0]; prices[k] = {"commodity": v, "msp": float(r.MSP), "market": float(r.Market_price), "suspect": bool(r.Suspect_unit)}

# ---------- 5. Metrics bundle for the app's "Model transparency" tab ----------
cm = json.load(open(RESULTS / "crop" / "final_metrics.json"))
metrics = {
    "crop": {"comparison": pd.read_csv(RESULTS / "crop" / "model_comparison.csv").to_dict("records"), "final": cm["final"], "tuning": cm["tuning"],
             "importance": pd.read_csv(RESULTS / "crop" / "feature_importance.csv").to_dict("records"),
             "robustness": pd.read_csv(RESULTS / "crop" / "robustness_noise.csv").to_dict("list"),
             "ablation": pd.read_csv(RESULTS / "crop" / "ablation.csv").to_dict("records")},
    "yield": {"comparison": pd.read_csv(RESULTS / "yield" / "yield_model_comparison.csv").to_dict("records"),
              "importance": pd.read_csv(RESULTS / "yield" / "yield_feature_importance.csv").to_dict("records"), "meta": json.load(open(RESULTS / "yield" / "yield_meta.json"))},
    "fertilizer": {"comparison": pd.read_csv(RESULTS / "fert" / "fertilizer_model_comparison.csv").to_dict("records"), "meta": json.load(open(RESULTS / "fert" / "fert_meta.json"))},
}
dump(metrics, "metrics.json")
dump({"crop_features": FEATS, "feature_stats": feat_stats, "profiles": profiles, "fert_stats": fert_stats, "prices": prices,
      "states": sorted(state_defaults), "crops_all": cats["Crop"], "seasons": cats["Season"]}, "metadata.json")
print("exported")
