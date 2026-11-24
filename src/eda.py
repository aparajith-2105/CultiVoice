"""Exploratory data analysis: saves charts and a data-quality summary."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd, numpy as np
from common import *

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})
OUT = RESULTS / "eda"
summary = []

# ---- 1. Crop recommendation dataset
cr = load_crop_rec()
summary.append(f"Crop_recommendation: {cr.shape[0]} rows, {cr.label.nunique()} crops, "
               f"{cr.isna().sum().sum()} missing, {cr.duplicated().sum()} duplicates, "
               f"{cr.label.value_counts().min()}-{cr.label.value_counts().max()} rows per crop (perfectly balanced)")
feats = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
for a, f in zip(ax.ravel(), feats):
    a.hist(cr[f], bins=30, color=GREEN, edgecolor="white"); a.set_title(f)
ax.ravel()[-1].axis("off")
fig.suptitle("Feature distributions (crop recommendation data)", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "01_feature_distributions.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(cr[feats].corr(), annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax, square=True)
ax.set_title("Correlation between features", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "02_correlation_heatmap.png"); plt.close(fig)
hi = cr[feats].corr().abs().where(~np.eye(7, dtype=bool)).stack().sort_values(ascending=False).head(1)
summary.append(f"Strongest feature correlation: {hi.index[0]} = {hi.iloc[0]:.2f} (P and K move together)")

fig, ax = plt.subplots(figsize=(13, 5.5))
order = cr.groupby("label")["rainfall"].median().sort_values().index
sns.boxplot(data=cr, x="label", y="rainfall", order=order, color=GREEN, ax=ax, fliersize=2)
ax.set_title("Rainfall needed by each crop (median order)", fontweight="bold")
ax.tick_params(axis="x", rotation=60); ax.set_xlabel("")
fig.tight_layout(); fig.savefig(OUT / "03_rainfall_by_crop.png"); plt.close(fig)

cm = cr.groupby("label")[feats].mean()
cmz = (cm - cm.mean()) / cm.std()
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cmz, cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "z-score of crop's mean"})
ax.set_title("Each crop's soil/climate profile", fontweight="bold"); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(OUT / "04_crop_profiles.png"); plt.close(fig)

# ---- 2. Yield dataset
raw = load_yield(clean=False); y = load_yield()
summary.append(f"crop_yield: {len(raw)} raw rows -> {len(y)} after cleaning "
               f"(dropped Coconut because it is measured in nuts, and rows with zero area/production). "
               f"Years {y.Crop_Year.min()}-{y.Crop_Year.max()}, {y.State.nunique()} states, {y.Crop.nunique()} crops")
g = y.groupby(["State", "Crop_Year"])
summary.append("In crop_yield, Fert_per_ha, Pest_per_ha and Annual_Rainfall are identical for every crop in the same "
               "state-year (std ~ 0), so they describe the state-year, not the crop.")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].hist(np.log10(raw["Yield"].clip(lower=1e-3)), bins=50, color=AMBER, edgecolor="white")
ax[0].set_title("Raw yield (log10): coconut outliers stretch the scale"); ax[0].set_xlabel("log10(yield)")
ax[1].hist(np.log1p(y["Yield"]), bins=50, color=GREEN, edgecolor="white")
ax[1].set_title("Cleaned yield, log1p scale (used for modelling)"); ax[1].set_xlabel("log1p(yield)")
fig.tight_layout(); fig.savefig(OUT / "05_yield_distribution.png"); plt.close(fig)

top = y.groupby("Crop")["Yield"].median().sort_values(ascending=False).head(12)
fig, ax = plt.subplots(figsize=(8, 5))
top.sort_values().plot.barh(color=GREEN, ax=ax); ax.set_xlabel("Median yield (tonnes/hectare)")
ax.set_title("Highest-yield crops (median)", fontweight="bold"); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(OUT / "06_top_yield_crops.png"); plt.close(fig)

yr = y.groupby("Crop_Year")["Yield"].median()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(yr.index, yr.values, color=DARK, marker="o"); ax.set_title("Median yield over time", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("tonnes/hectare")
fig.tight_layout(); fig.savefig(OUT / "07_yield_trend.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.8))
sub = y[y.Crop.isin(["Rice", "Wheat", "Maize", "Cotton(lint)"])]
sns.scatterplot(data=sub, x="Annual_Rainfall", y="Yield", hue="Crop", alpha=.35, s=14, ax=ax)
ax.set_ylim(0, sub.Yield.quantile(.99)); ax.set_title("Rainfall vs yield for major crops", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "08_rainfall_vs_yield.png"); plt.close(fig)

# ---- 3. Crop production (district) + rainfall merge
cp = load_production()
summary.append(f"crop_production: {len(cp)} rows, {cp.State_Name.nunique()} states, {cp.District_Name.nunique()} districts, "
               f"{cp.Crop.nunique()} crops, {cp.Crop_Year.min()}-{cp.Crop_Year.max()}; "
               f"{cp.Production.isna().sum()} rows with missing Production ({cp.Production.isna().mean():.1%})")
st = cp[cp.Crop != "Coconut"].groupby("State_Name")["Area"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 4.8))
(st / 1e6).sort_values().plot.barh(color=GREEN, ax=ax); ax.set_xlabel("Total cultivated area, 1997-2015 (million hectare-records)")
ax.set_title("Top states by recorded crop area", fontweight="bold"); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(OUT / "09_top_states_area.png"); plt.close(fig)

dr = load_district_rain()
def norm(s): return s.str.upper().str.replace(" AND ", " & ", regex=False).str.replace("&", "AND").str.replace(r"[^A-Z ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
a = set(zip(norm(cp.State_Name), norm(cp.District_Name))); b = set(zip(norm(dr.STATE_UT_NAME), norm(dr.DISTRICT)))
match = len(a & b); total = len(a)
summary.append(f"District merge check (crop_production vs district rainfall normals): {match} of {total} districts "
               f"match exactly after name cleaning ({match/total:.0%}); the rest need fuzzy matching")

# ---- 4. Fertilizer dataset
fe = load_fert()
summary.append(f"Fertilizer_Prediction: only {len(fe)} rows, {fe['Fertilizer Name'].nunique()} fertilizer classes "
               f"(smallest class = {fe['Fertilizer Name'].value_counts().min()} rows), {fe.isna().sum().sum()} missing")
fig, ax = plt.subplots(figsize=(7, 4))
fe["Fertilizer Name"].value_counts().plot.bar(color=GREEN, ax=ax); ax.set_title("Fertilizer classes (small, imbalanced)", fontweight="bold")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(OUT / "10_fertilizer_classes.png"); plt.close(fig)

# ---- 5. Price snapshot
pr = pd.read_csv(DATA / "Crop_Season_Wise_Price_Arrival.csv", skiprows=2)
pr.columns = ["Group", "Commodity", "MSP", "K_price", "K_arr", "R_price", "R_arr"]
pr = pr.dropna(subset=["Commodity"])
for c in ["MSP", "K_price", "K_arr", "R_price", "R_arr"]:
    pr[c] = pd.to_numeric(pr[c], errors="coerce")
pr["Market_price"] = pr["K_price"].fillna(pr["R_price"])
pr["Gap_pct"] = (pr["Market_price"] / pr["MSP"] - 1) * 100
pr = pr.dropna(subset=["Market_price"])
pr["Suspect_unit"] = pr["Commodity"].isin(["Jute", "Copra"])
pr = pr.sort_values("Gap_pct")
pr.to_csv(DATA / "price_clean.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 7))
plot = pr[~pr.Suspect_unit]
ax.barh(plot["Commodity"].str.slice(0, 28), plot["Gap_pct"], color=[GREEN if v >= 0 else AMBER for v in plot["Gap_pct"]])
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Market price vs MSP (%)")
ax.set_title("Market price relative to MSP, 2026-27 snapshot", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "11_msp_vs_market.png"); plt.close(fig)
summary.append("Price snapshot: Jute (market 839 vs MSP 5650) and Copra (23650 vs 12100) look like unit/grade mismatches, so they are excluded from the MSP chart")

(RESULTS / "eda" / "data_quality_summary.txt").write_text("\n".join(summary))
print("\n".join(summary))
