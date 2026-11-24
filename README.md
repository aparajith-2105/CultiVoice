# 🌾 CultiVoice

**An Explainable, Multilingual AI Assistant for Crop and Yield Recommendation**

End-semester project for *Fundamentals of AI using Agriculture DataSets* (IIT Ropar × Annam.ai).

---

## 1. The idea in one minute

**Problem.** A farmer deciding what to grow has to guess from experience. Most digital tools give a crop name with no reason,
need English, and need typing in a form.

**Solution.** CultiVoice is a small web app. The farmer gives their soil and weather values (by sliders, by voice, or by typing a sentence) and gets:

| You give | You get back |
|---|---|
| Nitrogen, Phosphorus, Potassium, soil pH, temperature, humidity, rainfall | **Top 3 crops** with a confidence score, and **why** the top crop was chosen |
| A state, crop, season and area | **Expected yield** (tonnes per hectare) and a comparison of crops in that state |
| Nitrogen, Phosphorus, Potassium | A **fertilizer** suggestion |
| Any question in plain words | An answer taken from the project's own data |

Everything works in **English, Hindi and Tamil**, supports **voice input and read-aloud**, and runs **entirely in the browser** (no backend server, no API key), so it can be hosted as a simple static website.

---

## 2. What makes it different (the three "extra" ideas)

1. **Explainable.** Every recommendation shows *why*: how each of your values compares with the crop's usual range, which values helped most, a radar chart of your field against the crop, and a "what if rainfall changes?" view.
2. **Multilingual and voice-first.** Interface, crop names and answers in three languages. Speak your values, listen to the advice.
3. **Honest.** A *Model transparency* tab shows the accuracy, the weaknesses and the limits of the data, instead of hiding them.

---

## 3. How it works

```
 Farmer's input ─────────────────────────────────────────────┐
  (sliders / voice / typed sentence / question)              │
                                                             ▼
  Datasets ──► Python (src/) ──► trained models ──► exported to JSON ──► JavaScript predictor ──► Web app
  (data/)      clean, explore,    Random Forest,     (frontend/model/)    (predict.js, runs        (frontend/)
               train, test        Gradient Boosting,                       in the browser)
                                  Logistic Regression
```

1. **Data** (`data/`): public Kaggle and government datasets.
2. **Python scripts** (`src/`): clean the data, make charts, train and compare models, save results.
3. **Export**: the final models are saved as JSON so the browser can run them. The JavaScript predictions were compared with the Python models on over 6,800 test rows with **zero disagreements**.
4. **Web app** (`frontend/`): one page with five tabs (below). No internet needed after loading.

---

## 4. The datasets

| Dataset | Source | Size | Used for |
|---|---|---|---|
| Crop recommendation | Kaggle | 2,200 rows, 22 crops, 7 soil/weather values | **Crop recommendation model** and the crop profiles |
| Crop yield (India) | Kaggle | 19,408 usable rows, 54 crops, 30 states, 1997–2020 | **Yield model** and the typical-yield answers |
| Fertilizer prediction | Kaggle | 99 rows | **Fertilizer model** |
| Season-wise price and arrival | Agmarknet | one 2026-27 snapshot | MSP and market price hint |
| Crop production, district rainfall normals | Kaggle / government | – | Analysis and charts |

Collected but **not used** by any script: sub-division rainfall, mean temperature, `apy_total_group_report.xls`, Pune soil-health cards.

---

## 5. Models and results

Every model was compared with simpler ones, tuned using only the training data, and then tested on data it had never seen.

| Task | Final model | Compared against | Result on unseen data |
|---|---|---|---|
| **Crop recommendation** (22 crops) | Random Forest (50 trees, depth 12) | Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM, Gradient Boosting | **99.5% accuracy**, correct crop in the top 3 for 100% of tests |
| **Yield** (tonnes/ha) | Gradient Boosting on log(yield) | Crop-median baseline, Ridge, Random Forest | **R² 0.92**; when predicting *future* years (2016–2020) the typical error is about 17% |
| **Fertilizer** | Logistic Regression on N, P, K | Models with extra soil/crop/climate inputs | 99.9% repeated cross-validation accuracy (very small data, see limits) |

What the models say:
- **Crop choice** depends mostly on **humidity, nitrogen and rainfall**, then potassium and phosphorus. pH and temperature matter little in this dataset.
- **Yield** is explained almost entirely by **which crop** and **which state**. Rainfall and fertilizer add very little in this data.

Good-practice steps taken: stratified train/test split, cross-validation and tuning on the training part only, a model-choice rule written down *before* seeing test results, no "answer leakage" (Production is not used to predict Yield), noise-robustness test, feature ablation, and testing on future years for yield.

Full tables are in `docs/RESULTS_SUMMARY.md`; charts are in `results/`.

---

## 6. The web app

| Tab | What it does |
|---|---|
| **Crop advisor** | Set 7 values (sliders, presets or 🎤 voice). Get top-3 crops, reasons, a radar chart, a rainfall what-if, MSP/market price hint, and 🔊 read-aloud. |
| **Yield** | Pick state, crop, season, area. Get expected yield and total production, plus a comparison of crops in that state. |
| **Fertilizer** | Enter N, P, K and get a suggested fertilizer. |
| **Model transparency** | Accuracy tables, important features, noise test, limitations. |
| **Chat** | Type or speak. Does two things (below). |

### The Chat tab (offline, English / Hindi / Tamil)

**A. Give your values in a sentence.** It finds the numbers next to keywords and asks for whatever is missing, then uses the same crop model.
> `N 90, P 42, K 43, temperature 21, humidity 82, pH 6.5, rainfall 203` → Rice, 98%

**B. Ask a question.** Answers come only from the project's own files, and say which dataset they used.

| Ask | Example |
|---|---|
| Crop conditions | "What conditions does rice need?" |
| Rankings | "Which crops need high rainfall?" |
| Comparison | "Compare rice and maize" |
| Price | "MSP of cotton" |
| Yield | "Wheat yield in Punjab" |
| Season | "Which season for maize?" |
| Fertilizer | "Fertilizer for nitrogen 20, phosphorus 10, potassium 5" |
| About the project | "How accurate is the model?", "What are the limitations?" |

Questions outside the data (pests, diseases, irrigation, subsidies, weather forecasts, sowing dates) get an honest *"I cannot answer that from my data"* instead of a guess.

---

## 7. How to run

### Option A: Use the live website (GitHub Pages)
Open the project link: **`https://<your-username>.github.io/cultivoice/`** *(replace with your own link)*.
Nothing to install. GitHub Pages serves the app over **HTTPS**, which the browser needs for the 🎤 voice input.

### Option B: Publish your own copy on GitHub Pages
The app is a plain static website (HTML + JavaScript + JSON files), so there is **no build step** and no server.
GitHub Pages can publish only from the top of a repository (or a `/docs` folder), so the app's files go at the top of a repo.

1. On github.com click **New repository**, name it `cultivoice`, make it **Public**, and create it.
2. Click **Add file → Upload files** and drag in the **contents** of the `frontend/` folder (not the folder itself):
   `index.html`, `chat_parser.js`, `chat_qa.js` and the `model/` folder. Click **Commit changes**.
3. Open **Settings → Pages**. Set Source to **Deploy from a branch**, Branch to **main**, folder to **/ (root)**, and click **Save**.
4. After 1 to 2 minutes the site is live at `https://<your-username>.github.io/cultivoice/`.
5. To update later, upload the changed files again (same names) and commit; the site refreshes by itself.

`index.html` must be at the top level of the repo, not inside a subfolder, otherwise the site shows a 404.
The `model/` folder must contain `crop_rf.json`, `yield_gbm.json`, `fert_lr.json`, `metadata.json`, `yield_defaults.json`, `metrics.json` and `predict.js`.

Only the app is published this way. The Python code, datasets and charts are not needed by the website, because the trained models are already exported to `model/` as JSON.
For the full project (code, data, results, docs), keep it in a second repository, for example `cultivoice-project`.

### Option C: Run it on your own computer
The page loads its model files with `fetch`, so it must be served over http. Double-clicking `index.html` will **not** work.
- **VS Code:** install the **Live Server** extension, right-click `frontend/index.html`, choose *Open with Live Server*.
- **Or terminal:** `cd frontend`, then `python -m http.server 8000`, then open http://localhost:8000

### Re-run the whole analysis (optional, about 2 minutes)
```
pip install -r requirements.txt
python src/run_all.py
```
This regenerates the charts in `results/`, retrains the models, re-exports them to `frontend/model/`, and (if Node.js is installed) checks that JavaScript matches Python. Results are reproducible (random seed 42). If you retrain, redeploy so the website gets the new model files.

### Run the tests (needs Node.js)
```
node tests/test_chat_parser.mjs     # 26 value-message tests
node tests/test_chat_qa.mjs         # 84 question tests
```

### If something does not work
- **Blank page, 404 or "failed to load model":** `index.html` is not at the top level of the repo, the `model/` folder is missing or incomplete, or you opened `index.html` by double-click (use Live Server instead).
- **Old version still showing after an update:** hard-refresh with Ctrl+Shift+R.
- **🎤 not working:** use Chrome or Edge on the HTTPS link; some browsers (for example Firefox) do not support voice input.

---

## 8. Folder guide

```
CultiVoice/
├── data/        the datasets used
├── src/         Python: common.py (loading/cleaning), eda.py, train_crop.py, train_yield.py,
│                train_fert.py, export_models.py, verify_export.py, run_all.py
├── models/      trained scikit-learn models (.joblib)
├── results/     charts and metric tables (eda/, crop/, yield/, fert/)
├── frontend/    the web app
│   ├── index.html        the page (5 tabs, 3 languages)
│   ├── chat_parser.js    reads the 7 values out of a message
│   ├── chat_qa.js        answers questions from the data
│   └── model/            exported models (JSON) + predict.js
├── tests/       chat tests
├── docs/        RESULTS_SUMMARY.md, VIVA_QA.md, AI_STUDIO_PROMPT.md
└── requirements.txt
```

**Tech:** Python (pandas, scikit-learn, matplotlib, seaborn) for the analysis; plain HTML, CSS and JavaScript for the app; no frameworks and no server.

---

## 9. Honest limitations

- The crop data is small, balanced and very clean, so near-perfect accuracy is expected. **Accuracy drops fast when readings are noisy** (`results/crop/robustness_noise.csv`).
- In the yield data, rainfall, fertilizer and pesticide are state-year values shared by all crops, so the model cannot learn their effect. The app therefore has no rainfall/fertilizer what-if for yield.
- The fertilizer dataset has only 99 rows and its labels follow the N-P-K numbers almost exactly.
- Fertilizer N, P, K values are on a different scale from the crop advisor's soil values.
- Prices are one 2026-27 snapshot, not a trend, and jute's market price in it looks unreliable.
- The Chat matches keywords, so unusual phrasing can be missed.
- Predictions show patterns in past data, not proven cause and effect. This is an educational project, not a substitute for local agronomy advice.

## 10. Future work

Field-level soil and weather data, price history for forecasting, testing with real farmers' inputs, a more noise-tolerant model (SVM or an ensemble), and an optional LLM (such as Gemini) for fully free-form chat (see `docs/AI_STUDIO_PROMPT.md`).
