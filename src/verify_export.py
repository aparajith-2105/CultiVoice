"""Check that the JavaScript engine reproduces the Python models (same answers on thousands of inputs)."""
import json, subprocess, shutil, tempfile
import numpy as np, pandas as pd, joblib
from pathlib import Path
from common import *
rng = np.random.default_rng(7)
M = ROOT / "frontend" / "model"
cb = joblib.load(MODELS / "crop_rf.joblib"); rf, le, FEATS = cb["model"], cb["label_encoder"], cb["features"]
meta = json.load(open(M / "metadata.json"))
crop_df = load_crop_rec()
# 1) crop: all 2200 real rows + 1000 random points inside feature ranges
rand = pd.DataFrame({f: rng.uniform(meta["feature_stats"][f]["min"], meta["feature_stats"][f]["max"], 1000) for f in FEATS})
Xc = pd.concat([crop_df[FEATS], rand]).reset_index(drop=True)
crop_expected = rf.predict_proba(Xc.values)
# 2) yield: 3000 random valid rows
yb = joblib.load(MODELS / "yield_model.joblib"); pipe = yb["model"]; yd = load_yield()
s = yd.sample(3000, random_state=1).copy()
s["Crop_Year"] = rng.integers(1997, 2021, len(s)); s["Annual_Rainfall"] *= rng.uniform(.6, 1.4, len(s)); s["Fert_per_ha"] *= rng.uniform(.7, 1.3, len(s)); s["Area"] = s["Area"] * rng.uniform(.5, 2, len(s))
s["logArea"] = np.log1p(s.Area)
yield_expected = np.expm1(pipe.predict(s[["Crop", "Season", "State", "Crop_Year", "logArea", "Annual_Rainfall", "Fert_per_ha", "Pest_per_ha"]]))
yin = s.rename(columns={"Crop": "crop", "Season": "season", "State": "state", "Crop_Year": "year", "Area": "area", "Annual_Rainfall": "rainfall", "Fert_per_ha": "fert_per_ha", "Pest_per_ha": "pest_per_ha"})[
    ["crop", "season", "state", "year", "area", "rainfall", "fert_per_ha", "pest_per_ha"]]
# 3) fertilizer: real rows + random
fb = joblib.load(MODELS / "fert_model.joblib"); fd = load_fert()
fr = pd.DataFrame({f: rng.uniform(0, 45, 500) for f in fb["num"]}); fx = pd.concat([fd[fb["num"]], fr]).reset_index(drop=True)
fert_expected = fb["model"].predict_proba(fx)
tests = {"crop_in": Xc.to_dict("records"), "crop_exp": crop_expected.tolist(), "classes": list(le.classes_),
         "yield_in": yin.to_dict("records"), "yield_exp": yield_expected.tolist(),
         "fert_in": fx.to_dict("records"), "fert_exp": fert_expected.tolist(), "fert_classes": [str(c) for c in fb["model"].named_steps["m"].classes_]}
tmp = Path(tempfile.mkdtemp()); (tmp / "tests.json").write_text(json.dumps(tests)); shutil.copy(M / "predict.js", tmp / "predict.mjs")
(tmp / "run.mjs").write_text(f'''
import fs from "fs"; import * as E from "./predict.mjs";
const M = "{M}"; const rd = (n) => JSON.parse(fs.readFileSync(M + "/" + n)); const T = JSON.parse(fs.readFileSync("{tmp}/tests.json"));
const crop = rd("crop_rf.json"), yld = rd("yield_gbm.json"), fert = rd("fert_lr.json");
let dc = 0, mism = 0; T.crop_in.forEach((x, i) => {{ const p = E.cropProbabilities(crop, x); dc = Math.max(dc, ...p.map((v, k) => Math.abs(v - T.crop_exp[i][k]))); const a = p.indexOf(Math.max(...p)); const b = T.crop_exp[i].indexOf(Math.max(...T.crop_exp[i])); if (a !== b) mism++; }});
let dy = 0, rel = 0; T.yield_in.forEach((x, i) => {{ const v = E.predictYield(yld, x); dy = Math.max(dy, Math.abs(v - T.yield_exp[i])); rel = Math.max(rel, Math.abs(v - T.yield_exp[i]) / T.yield_exp[i]); }});
let df = 0, fm = 0; T.fert_in.forEach((x, i) => {{ const p = E.recommendFertilizer(fert, x); const by = Object.fromEntries(p.map(o => [o.fertilizer, o.prob])); T.fert_classes.forEach((c, k) => {{ df = Math.max(df, Math.abs(by[c] - T.fert_exp[i][k])); }}); if (p[0].fertilizer !== T.fert_classes[T.fert_exp[i].indexOf(Math.max(...T.fert_exp[i]))]) fm++; }});
console.log(JSON.stringify({{crop_rows: T.crop_in.length, crop_max_prob_diff: dc, crop_top1_mismatch: mism, yield_rows: T.yield_in.length, yield_max_abs_diff_tha: dy, yield_max_rel_diff: rel, fert_rows: T.fert_in.length, fert_max_prob_diff: df, fert_top1_mismatch: fm}}, null, 1));
''')
out = subprocess.run(["node", str(tmp / "run.mjs")], capture_output=True, text=True); print(out.stdout, out.stderr[:800])
