// CultiVoice prediction engine. Pure JavaScript, no dependencies; runs in the browser or Node.
// Every function takes the exported model objects as arguments, so it works with fetch(), import, or fs.readFileSync.

const walk = (tree, x) => {
  let i = 0;
  while (tree.f[i] >= 0) {
    const t = tree.t[i];
    const goLeft = Array.isArray(t) ? t.includes(x[tree.f[i]]) : x[tree.f[i]] <= t;
    i = goLeft ? tree.l[i] : tree.r[i];
  }
  return tree.v[i];
};

// ---------- Crop recommendation (Random Forest) ----------
export function cropProbabilities(model, input) {
  const x = model.features.map((f) => Number(input[f]));
  const acc = new Array(model.classes.length).fill(0);
  for (const tree of model.trees) {
    const leaf = walk(tree, x);
    for (let k = 0; k < acc.length; k++) acc[k] += leaf[k];
  }
  return acc.map((v) => v / model.trees.length);
}

export function topCrops(model, input, k = 3) {
  const p = cropProbabilities(model, input);
  return model.classes.map((crop, i) => ({ crop, prob: p[i] })).sort((a, b) => b.prob - a.prob).slice(0, k);
}

// Why this crop? (1) is each value inside the crop's typical range (10th-90th percentile)?
// (2) how much does the crop's probability drop if that one value is replaced by the dataset average?
export function explainCrop(model, meta, input, crop) {
  const idx = model.classes.indexOf(crop);
  const base = cropProbabilities(model, input)[idx];
  return model.features.map((f) => {
    const prof = meta.profiles[crop][f];
    const v = Number(input[f]);
    const status = v >= prof.p10 && v <= prof.p90 ? "typical" : v >= prof.min && v <= prof.max ? "edge" : "unusual";
    const occluded = { ...input, [f]: meta.feature_stats[f].mean };
    return { feature: f, value: v, status, typical: [prof.p10, prof.p90], influence: base - cropProbabilities(model, occluded)[idx] };
  }).sort((a, b) => b.influence - a.influence);
}

// What if rainfall changes by pct% ? Returns the top-3 crops for each scenario.
export function rainfallWhatIf(model, input, pcts = [-40, -20, 0, 20, 40]) {
  return pcts.map((pct) => ({ pct, top: topCrops(model, { ...input, rainfall: input.rainfall * (1 + pct / 100) }, 3) }));
}

// ---------- Yield (Gradient Boosting, log scale) ----------
export function predictYield(model, input) {
  const code = (name, v) => {
    const i = model.cats[name].indexOf(v);
    if (i < 0) throw new Error(`Unknown ${name}: ${v}`);
    return i;
  };
  const x = [code("Crop", input.crop), code("Season", input.season), code("State", input.state),
    input.year, Math.log1p(input.area), input.rainfall, input.fert_per_ha, input.pest_per_ha];
  let raw = model.baseline;
  for (const tree of model.trees) raw += walk(tree, x);
  return Math.expm1(raw); // tonnes per hectare
}

// Compare the yield of every crop grown in a state, using that state's latest conditions.
export function compareCropsInState(model, defaults, state, year = 2019) {
  const d = defaults[state];
  return Object.entries(d.crops).map(([crop, info]) => ({
    crop, season: info.seasons[0],
    yield: predictYield(model, { crop, season: info.seasons[0], state, year, area: info.median_area, rainfall: d.rainfall, fert_per_ha: d.fert_per_ha, pest_per_ha: d.pest_per_ha }),
  })).sort((a, b) => b.yield - a.yield);
}

// ---------- Fertilizer (multinomial logistic regression on N, P, K) ----------
export function recommendFertilizer(model, input) {
  const z = model.features.map((f, j) => (Number(input[f]) - model.mean[j]) / model.scale[j]);
  const logits = model.coef.map((row, c) => row.reduce((s, w, j) => s + w * z[j], model.intercept[c]));
  const m = Math.max(...logits);
  const e = logits.map((l) => Math.exp(l - m));
  const s = e.reduce((a, b) => a + b, 0);
  return model.classes.map((name, i) => ({ fertilizer: name, prob: e[i] / s, npk: model.composition_NPK[name] })).sort((a, b) => b.prob - a.prob);
}

// ---------- Market price hint (Agmarknet snapshot, 2026-27) ----------
export function priceHint(meta, crop) {
  const p = meta.prices[crop];
  if (!p || p.suspect) return null;
  return { commodity: p.commodity, msp: p.msp, market: p.market, gapPct: (p.market / p.msp - 1) * 100 };
}
