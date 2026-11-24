# CultiVoice

**An Explainable, Multilingual AI Assistant for Crop and Yield Recommendation**
Phase 2 project for *Fundamentals of AI using Agriculture DataSets*.

A farmer enters soil and climate values (N, P, K, pH, temperature, humidity, rainfall) and gets the top-3 suitable
crops with a confidence score, a reason for the top crop, an expected yield, a fertilizer suggestion and an MSP price hint,
in English, Tamil or Hindi, with voice input and read-aloud.

## Folder layout

```
CultiVoice/
├── data/        the CSV files used (copied from Kaggle / government portals)
├── src/         Python scripts (run_all.py runs everything)
├── models/      trained scikit-learn models (.joblib)
├── results/     charts and metric tables (eda/, crop/, yield/, fert/)
├── frontend/    the web app (index.html + model/ with the exported JSON and predict.js)
├── docs/        RESULTS_SUMMARY.md, VIVA_QA.md, AI_STUDIO_PROMPT.md
└── requirements.txt
```

## Run the web app (no Python needed)

The page loads its model files with `fetch`, so it must be served over http. Opening `index.html` by double-click will not work.

- **VS Code:** install the *Live Server* extension, right-click `frontend/index.html`, choose *Open with Live Server*.
- **Or from a terminal:** `cd frontend` then `python -m http.server 8000` and open http://localhost:8000

## Re-run the analysis (optional)

```
pip install -r requirements.txt
python src/run_all.py
```
This regenerates every chart in `results/`, retrains the models, re-exports them to `frontend/model/` and (if Node.js is
installed) checks that the JavaScript predictions match the Python models. Runtime is about 2 minutes. Results are
reproducible (fixed random seed 42).

## What is in the models

| Task | Data | Model | Headline result (held-out) |
|---|---|---|---|
| Crop recommendation (22 crops) | Crop_recommendation.csv, 2,200 rows | Random Forest (tuned), compared with 6 other models | 99.5% test accuracy, macro-F1 0.995, correct crop in top-3 for 100% |
| Yield (tonnes/ha) | crop_yield.csv, 19,408 rows after cleaning | Gradient Boosting, compared with Random Forest, Ridge, crop-median baseline | R² 0.92 (log scale), median error 17% when predicting future years 2016-2020 |
| Fertilizer | Fertilizer_Prediction.csv, 99 rows | Logistic Regression on N, P, K | 99.9% repeated-CV accuracy (see the caveat in docs/RESULTS_SUMMARY.md) |

The web app runs these models in the browser through `frontend/model/predict.js`. The JavaScript output was checked against
the Python models on more than 6,800 test rows with zero disagreements on the predicted crop or fertilizer.
The chat/Gemini part is optional: see `docs/AI_STUDIO_PROMPT.md`.

## Data files

Used by the models: `Crop_recommendation.csv`, `crop_yield.csv`, `Fertilizer_Prediction.csv`.
Used for analysis and charts: `crop_production.csv`, `district_wise_rainfall_normal.csv`, `Crop_Season_Wise_Price_Arrival.csv` (price hint).
Collected as supporting data but **not used by any script**: `Sub_Division_Rainfall_india.csv`, `Mean_Temp_india.csv`,
`apy_total_group_report.xls`, `Cat_data_all_pub.xlsx` + `data_dictionary.csv` (Pune soil health cards).

## Honest limitations

- The crop recommendation data is small, balanced and very clean, so near-perfect accuracy is expected. Accuracy drops quickly when readings are noisy (results/crop/robustness_noise.csv).
- In the yield data, rainfall, fertilizer and pesticide are state-year values shared by every crop, so the model cannot learn their effect. The app therefore offers no rainfall/fertilizer what-if for yield.
- The fertilizer dataset has only 99 rows and its labels follow the N-P-K numbers almost exactly.
- Prices are a single 2026-27 snapshot, not a time series.
- Predictions describe patterns in past data. They are not proof of cause and effect and not a substitute for local agronomy advice.
