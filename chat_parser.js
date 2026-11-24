// CultiVoice chat parser. Pure JavaScript, no dependencies, works offline.
// Reads a farmer's message (English, Hindi or Tamil, Latin or native digits) and extracts the seven crop-model inputs.
// It does NOT guess: only numbers that sit next to a recognised keyword or unit are used.

export const FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"];

// Latin keywords must match the whole word (so "rained" or "phone" never count). Indian-script keywords match as prefixes, which handles suffixes.
const KEYWORDS = {
  N: ["nitrogen", "nitro", "n", "नाइट्रोजन", "नाईट्रोजन", "नत्रजन", "नाइट्रोज़न", "நைட்ரஜன்", "நைட்ரஜென்", "தழைச்சத்து"],
  P: ["phosphorus", "phosphorous", "phosphate", "phosphorus", "p", "फॉस्फोरस", "फास्फोरस", "फॉस्फेट", "फास्फेट", "பாஸ்பரஸ்", "மணிச்சத்து"],
  K: ["potassium", "potash", "k", "पोटैशियम", "पोटेशियम", "पोटाश", "பொட்டாசியம்", "சாம்பல்சத்து", "சாம்பற்சத்து"],
  temperature: ["temperature", "temperatures", "temp", "तापमान", "வெப்பநிலை", "வெப்பம்"],
  humidity: ["humidity", "humid", "नमी", "आर्द्रता", "ஈரப்பதம்", "ஈரப்பதன்"],
  ph: ["ph", "पीएच", "பிஎச்", "பிஹெச்"],
  rainfall: ["rainfall", "rain", "rains", "precipitation", "बारिश", "वर्षा", "மழை", "மழைப்பொழிவு", "மழையளவு"],
};
// weak hints: a unit right after a number can identify the feature when no keyword does
const UNITS = {
  rainfall: ["mm", "millimeter", "millimeters", "millimetre", "millimetres", "मिमी", "மிமீ", "மில்லிமீட்டர்"],
  temperature: ["°", "celsius", "degree", "degrees", "deg", "डिग्री", "டிகிரி", "செல்சியஸ்"],
  humidity: ["%", "percent", "percentage", "प्रतिशत", "சதவீதம்", "சதவிகிதம்"],
};
// values the farmer could really have (hard limits). Anything outside is ignored with a warning.
const HARD = { N: [0, 500], P: [0, 500], K: [0, 500], temperature: [-20, 60], humidity: [0, 100], ph: [0, 14], rainfall: [0, 6000] };

const DIGITS = [["०", "0"], ["१", "1"], ["२", "2"], ["३", "3"], ["४", "4"], ["५", "5"], ["६", "6"], ["७", "7"], ["८", "8"], ["९", "9"],
  ["௦", "0"], ["௧", "1"], ["௨", "2"], ["௩", "3"], ["௪", "4"], ["௫", "5"], ["௬", "6"], ["௭", "7"], ["௮", "8"], ["௯", "9"]];

export function normalize(text) {
  let s = String(text).toLowerCase();
  for (const [a, b] of DIGITS) s = s.split(a).join(b);
  s = s.replace(/(\d),(\d{3})(?!\d)/g, "$1$2").replace(/(\d),(\d{1,2})(?!\d)/g, "$1.$2"); // 1,200 -> 1200 ; 6,5 -> 6.5
  s = s.replace(/(\d)([^\d\s.,])/g, "$1 $2").replace(/([^\d\s.,])(\d)/g, "$1 $2");        // 200mm -> 200 mm ; n90 -> n 90
  return s;
}

export function detectLanguage(text) {
  const ta = (text.match(/[஀-௿]/g) || []).length, hi = (text.match(/[ऀ-ॿ]/g) || []).length;
  if (!ta && !hi) return "en";
  return ta >= hi ? "ta" : "hi";
}

const matchKeyword = (tok) => {
  for (const [f, kws] of Object.entries(KEYWORDS)) for (const k of kws) {
    const latin = /^[a-z]+$/.test(k);
    if (latin ? tok === k : tok.startsWith(k)) return f;
  }
  return null;
};
const matchUnit = (tok) => { for (const [f, us] of Object.entries(UNITS)) if (us.includes(tok)) return f; return null; };

export function parseMessage(text, ranges = null) {
  let s = normalize(text);
  const values = {}, warnings = [];
  const accept = (f, v) => {
    if (!Number.isFinite(v)) return;
    const [lo, hi] = HARD[f];
    if (v < lo || v > hi) { warnings.push({ feature: f, value: v, reason: "invalid" }); return; }
    if (ranges && ranges[f] && (v < ranges[f].min || v > ranges[f].max)) warnings.push({ feature: f, value: v, reason: "outside", min: ranges[f].min, max: ranges[f].max });
    values[f] = v;
  };

  // "NPK 90-42-43" style, handled first and then blanked out
  const npk = s.match(/\bn\s*[-:/]?\s*p\s*[-:/]?\s*k\b\D{0,12}?(\d+(?:\.\d+)?)\D{1,4}?(\d+(?:\.\d+)?)\D{1,4}?(\d+(?:\.\d+)?)/);
  if (npk) { accept("N", +npk[1]); accept("P", +npk[2]); accept("K", +npk[3]); s = s.replace(npk[0], " "); }

  const raw = s.match(/\d+(?:\.\d+)?|[\p{L}\p{M}]+|[%°]/gu) || [];
  const toks = raw.map((r) => {
    if (/^\d/.test(r)) return { type: "num", v: parseFloat(r) };
    const f = matchKeyword(r); if (f) return { type: "kw", f };
    const u = matchUnit(r); if (u) return { type: "unit", f: u };
    return { type: "word" };
  });

  const taken = new Set(Object.keys(values)), usedTok = new Set(), assigned = new Map();
  const nums = toks.map((t, i) => (t.type === "num" ? i : -1)).filter((i) => i >= 0);
  const give = (i, f, j) => { assigned.set(i, f); taken.add(f); if (j != null) usedTok.add(j); accept(f, toks[i].v); };

  // pass 1: keyword just before the number ("nitrogen is 90", "N=90", "rainfall of 203")
  for (const i of nums) {
    for (let j = i - 1; j >= 0 && j >= i - 5; j--) {
      if (toks[j].type === "num") break;
      if (toks[j].type === "kw" && !usedTok.has(j) && !taken.has(toks[j].f)) { give(i, toks[j].f, j); break; }
    }
  }
  // pass 2: keyword just after the number ("203 mm rainfall", "21 degrees temperature")
  for (const i of nums) {
    if (assigned.has(i)) continue;
    for (let j = i + 1; j < toks.length && j <= i + 4; j++) {
      if (toks[j].type === "num") break;
      if (toks[j].type === "kw" && !usedTok.has(j) && !taken.has(toks[j].f)) { give(i, toks[j].f, j); break; }
    }
  }
  // pass 3: a unit right after the number ("203 mm", "21 °C", "82 %")
  for (const i of nums) {
    if (assigned.has(i)) continue;
    const nx = toks[i + 1];
    if (nx && nx.type === "unit" && !taken.has(nx.f)) give(i, nx.f, null);
  }
  return { values, warnings, lang: detectLanguage(text), numbersFound: nums.length + (npk ? 3 : 0) };
}
