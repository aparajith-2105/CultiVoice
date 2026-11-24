"""Shared paths, colours and data loaders for the CultiVoice project."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
for _p in (RESULTS, MODELS):
    _p.mkdir(exist_ok=True)

SEED = 42
GREEN, DARK, AMBER = "#2E7D4F", "#1B3A2B", "#D98324"


def load_crop_rec():
    """Crop Recommendation dataset: N, P, K, temperature, humidity, ph, rainfall -> label (22 crops)."""
    return pd.read_csv(DATA / "Crop_recommendation.csv")


def load_yield(clean=True):
    """Crop Yield in Indian States.
    clean=True: strip padded text, drop Coconut (measured in nuts, not tonnes), drop rows with zero
    area, production or yield, and add per-hectare fertilizer/pesticide use. Production is kept only for reference:
    it is NOT a model feature because Yield = Production / Area (that would leak the answer)."""
    df = pd.read_csv(DATA / "crop_yield.csv")
    for c in ("Crop", "Season", "State"):
        df[c] = df[c].str.strip()
    if not clean:
        return df
    df = df[(df.Crop != "Coconut") & (df.Area > 0) & (df.Production > 0) & (df.Yield > 0)].copy()
    df["Fert_per_ha"] = df["Fertilizer"] / df["Area"]
    df["Pest_per_ha"] = df["Pesticide"] / df["Area"]
    return df.reset_index(drop=True)


def load_production():
    """Crop Production in India (state, district, year, season, crop, area, production)."""
    df = pd.read_csv(DATA / "crop_production.csv")
    for c in ("State_Name", "District_Name", "Season", "Crop"):
        df[c] = df[c].str.strip()
    return df


def load_district_rain():
    """District-wise normal monthly rainfall."""
    df = pd.read_csv(DATA / "district_wise_rainfall_normal.csv")
    for c in ("STATE_UT_NAME", "DISTRICT"):
        df[c] = df[c].str.strip()
    return df


def load_fert():
    """Fertilizer Prediction (99 rows). Fixes the typos/padding in the original column names."""
    df = pd.read_csv(DATA / "Fertilizer_Prediction.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Temparature": "Temperature", "Phosphorous": "Phosphorus"})
    for c in ("Soil Type", "Crop Type", "Fertilizer Name"):
        df[c] = df[c].str.strip()
    return df
