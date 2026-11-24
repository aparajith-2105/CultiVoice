// CultiVoice chat Q&A. Pure JavaScript, offline. Answers typed or spoken questions (English, Hindi, Tamil)
// using ONLY the project's own data files: model/metadata.json, yield_defaults.json, fert_lr.json, metrics.json.
// No internet, no API key. If a question cannot be answered from that data it says so instead of guessing.
import { parseMessage, normalize } from "./chat_parser.js";
import { recommendFertilizer } from "./model/predict.js";

const REC_NAMES = {
  rice: ["rice", "paddy", "चावल", "धान", "நெல்", "அரிசி"], maize: ["maize", "corn", "मक्का", "மக்காச்சோளம்"],
  chickpea: ["chickpea", "chickpeas", "chana", "bengal gram", "gram", "चना", "கொண்டைக்கடலை"], kidneybeans: ["kidney bean", "kidney beans", "kidneybeans", "rajma", "राजमा", "ராஜ்மா"],
  pigeonpeas: ["pigeon pea", "pigeon peas", "pigeonpeas", "arhar", "tur", "toor", "red gram", "अरहर", "तुअर", "துவரை"], mothbeans: ["moth bean", "moth beans", "mothbeans", "moth", "मोठ", "மொத் பயறு"],
  mungbean: ["mung bean", "mungbean", "mung", "moong", "green gram", "मूंग", "பாசிப்பயறு", "பாசிப்பருப்பு"], blackgram: ["black gram", "blackgram", "urad", "urd", "उड़द", "உளுந்து"],
  lentil: ["lentil", "lentils", "masoor", "masur", "मसूर", "மசூர்"], pomegranate: ["pomegranate", "अनार", "மாதுளை"], banana: ["banana", "केला", "வாழை"], mango: ["mango", "आम", "மாம்பழம்", "மாங்காய்"],
  grapes: ["grape", "grapes", "अंगूर", "திராட்சை"], watermelon: ["watermelon", "तरबूज", "தர்பூசணி"], muskmelon: ["muskmelon", "खरबूजा", "முலாம்பழம்"], apple: ["apple", "सेब", "ஆப்பிள்"],
  orange: ["orange", "संतरा", "ஆரஞ்சு"], papaya: ["papaya", "पपीता", "பப்பாளி"], coconut: ["coconut", "नारियल", "தேங்காய்"], cotton: ["cotton", "कपास", "பருத்தி"], jute: ["jute", "जूट", "சணல்"], coffee: ["coffee", "कॉफ़ी", "कॉफी", "காபி"],
};
const REC_LABEL = {
  hi: { rice: "चावल", maize: "मक्का", chickpea: "चना", kidneybeans: "राजमा", pigeonpeas: "अरहर", mothbeans: "मोठ", mungbean: "मूंग", blackgram: "उड़द", lentil: "मसूर", pomegranate: "अनार", banana: "केला", mango: "आम", grapes: "अंगूर", watermelon: "तरबूज", muskmelon: "खरबूजा", apple: "सेब", orange: "संतरा", papaya: "पपीता", coconut: "नारियल", cotton: "कपास", jute: "जूट", coffee: "कॉफ़ी" },
  ta: { rice: "நெல்", maize: "மக்காச்சோளம்", chickpea: "கொண்டைக்கடலை", kidneybeans: "ராஜ்மா", pigeonpeas: "துவரை", mothbeans: "மொத் பயறு", mungbean: "பாசிப்பயறு", blackgram: "உளுந்து", lentil: "மசூர் பருப்பு", pomegranate: "மாதுளை", banana: "வாழை", mango: "மாம்பழம்", grapes: "திராட்சை", watermelon: "தர்பூசணி", muskmelon: "முலாம்பழம்", apple: "ஆப்பிள்", orange: "ஆரஞ்சு", papaya: "பப்பாளி", coconut: "தேங்காய்", cotton: "பருத்தி", jute: "சணல்", coffee: "காபி" },
};
// recommendation crop -> crop name used in the yield dataset
const REC_TO_YC = { rice: "Rice", maize: "Maize", chickpea: "Gram", pigeonpeas: "Arhar/Tur", mungbean: "Moong(Green Gram)", blackgram: "Urad", lentil: "Masoor", mothbeans: "Moth", cotton: "Cotton(lint)", jute: "Jute", banana: "Banana" };
// yield-only crops: [yield name, english aliases, hindi, tamil]
const YO = [
  ["Wheat", ["wheat", "gehun"], "गेहूं|गेहूँ", "கோதுமை"], ["Sugarcane", ["sugarcane", "sugar cane"], "गन्ना", "கரும்பு"], ["Groundnut", ["groundnut", "peanut", "peanuts"], "मूंगफली", "நிலக்கடலை"],
  ["Potato", ["potato", "potatoes"], "आलू", "உருளைக்கிழங்கு"], ["Onion", ["onion", "onions"], "प्याज", "வெங்காயம்"], ["Turmeric", ["turmeric"], "हल्दी", "மஞ்சள்"], ["Ginger", ["ginger"], "अदरक", "இஞ்சி"],
  ["Tapioca", ["tapioca", "cassava"], "कसावा|टैपिओका", "மரவள்ளி"], ["Bajra", ["bajra", "pearl millet"], "बाजरा", "கம்பு"], ["Jowar", ["jowar", "sorghum"], "ज्वार", "சோளம்"], ["Ragi", ["ragi", "finger millet"], "रागी", "கேழ்வரகு"],
  ["Soyabean", ["soybean", "soyabean", "soya"], "सोयाबीन", "சோயா"], ["Sunflower", ["sunflower"], "सूरजमुखी", "சூரியகாந்தி"], ["Sesamum", ["sesame", "sesamum", "til"], "तिल", "எள்"], ["Barley", ["barley"], "जौ", "பார்லி"],
  ["Rapeseed &Mustard", ["mustard", "rapeseed"], "सरसों", "கடுகு"], ["Black pepper", ["black pepper", "pepper"], "काली मिर्च", "மிளகு"], ["Dry chillies", ["chilli", "chillies", "chili", "chilies"], "मिर्च", "மிளகாய்"],
  ["Garlic", ["garlic"], "लहसुन", "பூண்டு"], ["Arecanut", ["arecanut", "areca nut", "areca"], "सुपारी", "பாக்கு"], ["Cashewnut", ["cashew", "cashewnut"], "काजू", "முந்திரி"], ["Cardamom", ["cardamom"], "इलायची", "ஏலக்காய்"],
  ["Coriander", ["coriander"], "धनिया", "கொத்தமல்லி"], ["Castor seed", ["castor"], "अरंडी", "ஆமணக்கு"], ["Tobacco", ["tobacco"], "तंबाकू|तम्बाकू", "புகையிலை"], ["Sweet potato", ["sweet potato"], "शकरकंद", "சர்க்கரைவள்ளி"],
  ["Linseed", ["linseed", "flax"], "अलसी", "ஆளி"], ["Horse-gram", ["horse gram", "horsegram"], "कुल्थी", "கொள்ளு"], ["Cowpea(Lobia)", ["cowpea", "lobia"], "लोबिया", "காராமணி"], ["Ragi", ["ragi"], "रागी", "கேழ்வரகு"],
];
const STATE_X = {
  "Tamil Nadu": ["tamilnadu", "तमिलनाडु", "தமிழ்நாடு", "தமிழகம்"], Punjab: ["पंजाब", "பஞ்சாப்"], Kerala: ["केरल", "கேரளா", "கேரளம்"], Karnataka: ["कर्नाटक", "கர்நாடகா"],
  "Andhra Pradesh": ["andhra", "आंध्र प्रदेश", "ஆந்திரா", "ஆந்திர"], Maharashtra: ["महाराष्ट्र", "மகாராஷ்டிரா"], "Uttar Pradesh": ["उत्तर प्रदेश", "உத்தர பிரதேசம்"], Bihar: ["बिहार", "பீகார்"],
  Gujarat: ["गुजरात", "குஜராத்"], Rajasthan: ["राजस्थान", "ராஜஸ்தான்"], "West Bengal": ["पश्चिम बंगाल", "மேற்கு வங்காளம்"], "Madhya Pradesh": ["मध्य प्रदेश", "மத்திய பிரதேசம்"],
  Haryana: ["हरियाणा", "ஹரியானா"], Odisha: ["orissa", "ओडिशा", "ஒடிசா"], Assam: ["असम", "அஸ்ஸாம்"], Telangana: ["तेलंगाना", "தெலங்கானா"],
};
const FN = { // short feature names
  en: { N: "Nitrogen (N)", P: "Phosphorus (P)", K: "Potassium (K)", temperature: "Temperature", humidity: "Humidity", ph: "Soil pH", rainfall: "Rainfall" },
  hi: { N: "नाइट्रोजन (N)", P: "फॉस्फोरस (P)", K: "पोटैशियम (K)", temperature: "तापमान", humidity: "नमी", ph: "मिट्टी का pH", rainfall: "वर्षा" },
  ta: { N: "நைட்ரஜன் (N)", P: "பாஸ்பரஸ் (P)", K: "பொட்டாசியம் (K)", temperature: "வெப்பநிலை", humidity: "ஈரப்பதம்", ph: "மண் pH", rainfall: "மழை" },
};
const UNIT = { N: "", P: "", K: "", temperature: " °C", humidity: " %", ph: "", rainfall: " mm" };
const FEATS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"];
// feature words: [feature, implied direction or null, keywords]
const FEATKW = [
  ["N", null, ["nitrogen", "नाइट्रोजन", "நைட்ரஜன்"]], ["P", null, ["phosphorus", "phosphorous", "phosphate", "फॉस्फोरस", "फास्फोरस", "பாஸ்பரஸ்"]], ["K", null, ["potassium", "potash", "पोटैशियम", "पोटेशियम", "पोटाश", "பொட்டாசியம்"]],
  ["temperature", null, ["temperature", "temp", "तापमान", "வெப்பநிலை"]], ["temperature", "high", ["hot", "warm", "heat", "गर्म", "வெப்பமான"]], ["temperature", "low", ["cold", "cool", "ठंड", "குளிர்"]],
  ["humidity", null, ["humidity", "humid", "नमी", "आर्द्रता", "ஈரப்பதம்"]], ["ph", null, ["ph", "पीएच", "பிஎச்"]], ["ph", "low", ["acidic", "acid", "अम्लीय", "அமிலம்"]], ["ph", "high", ["alkaline", "alkali", "basic", "क्षारीय", "காரம்"]],
  ["rainfall", null, ["rainfall", "rain", "rainy", "water", "बारिश", "वर्षा", "पानी", "மழை", "தண்ணீர்"]], ["rainfall", "low", ["drought", "dry", "सूखा", "வறட்சி"]], ["rainfall", "high", ["wet", "flood", "waterlogged"]],
];
const W = {
  high: ["high", "highest", "more", "most", "maximum", "max", "rich", "heavy", "lot", "large", "ज़्यादा", "ज्यादा", "अधिक", "सबसे ज्यादा", "ज़ादा", "அதிக", "அதிகம்", "அதிகமான"],
  low: ["low", "lowest", "less", "least", "minimum", "little", "small", "कम", "குறைவு", "குறைந்த", "குறைவான"],
  price: ["price", "pric", "msp", "rate", "cost", "market", "mandi", "sell", "भाव", "कीमत", "दाम", "मूल्य", "एमएसपी", "मंडी", "விலை", "எம்எஸ்பி", "சந்தை"],
  yield: ["yield", "production", "produce", "harvest", "tonne", "tonnes", "output", "per hectare", "उपज", "पैदावार", "उत्पादन", "उत्पादकता", "மகசூல்", "விளைச்சல்", "உற்பத்தி"],
  season: ["season", "kharif", "rabi", "zaid", "sow", "sowing", "plant", "planting", "when", "month", "सीजन", "सीज़न", "ऋतु", "खरीफ", "रबी", "बुवाई", "कब", "பருவம்", "சீசன்", "காரீப்", "ராபி", "விதைப்பு", "எப்போது"],
  fert: ["fertili", "urea", "dap", "manure", "उर्वरक", "खाद", "यूरिया", "डीएपी", "உரம்", "யூரியா", "டிஏபி"],
  scope: ["pest", "disease", "insect", "blight", "fungus", "fungicide", "weed", "irrigat", "subsid", "loan", "insurance", "forecast", "tomorrow", "organic", "कीट", "रोग", "सिंचाई", "सब्सिडी", "ऋण", "बीमा", "पूर्वानुमान", "பூச்சி", "நோய்", "பாசன", "மானியம்", "கடன்", "காப்பீடு", "முன்னறிவிப்பு"],
  acc: ["accura", "reliab", "trust", "precision", "how good", "correct", "सटीक", "भरोसा", "सही", "துல்லிய", "நம்பக"],
  imp: ["important", "factor", "most affect", "feature", "महत्वपूर्ण", "कारक", "मुख्य", "முக்கிய", "காரணி"],
  data: ["dataset", "data", "source", "kaggle", "where from", "डेटा", "स्रोत", "தரவு", "மூலம்"],
  limit: ["limit", "weakness", "drawback", "problem", "shortcoming", "सीमा", "कमी", "कमियां", "வரம்பு", "குறை"],
  how: ["algorithm", "random forest", "how does", "how do you", "how it work", "model", "एल्गोरिदम", "कैसे काम", "मॉडल", "எப்படி வேலை", "மாதிரி", "அல்காரிதம்"],
  list: ["which crops", "what crops", "list", "supported", "how many crops", "all crops", "कौन सी फसल", "कौन-सी फसल", "कितनी फसल", "फसलों की सूची", "என்ன பயிர்", "எந்த பயிர்", "எத்தனை பயிர்", "பயிர்கள் பட்டியல்"],
  help: ["help", "what can you", "what do you", "how to use", "menu", "मदद", "क्या कर सकते", "உதவி", "என்ன செய்ய", "என்ன செய்வ"],
  hello: ["hello", "hi", "hey", "namaste", "namaskar", "vanakkam", "good morning", "नमस्ते", "नमस्कार", "हेलो", "வணக்கம்", "ஹலோ"],
  thanks: ["thank", "thanks", "धन्यवाद", "शुक्रिया", "நன்றி"],
};
const isLatin = (k) => /^[a-z ]+$/.test(k);
const hasAny = (s, list) => list.some((k) => (isLatin(k) ? new RegExp("(^|[^a-z])" + k).test(s) : s.includes(k)));
const hasWord = (s, k) => (isLatin(k) ? new RegExp("(^|[^a-z])" + k + "([^a-z]|$)").test(s) : s.includes(k));

function buildEntities(ctx) {
  const ents = [];
  for (const [id, al] of Object.entries(REC_NAMES)) ents.push({ id, rec: id, yc: REC_TO_YC[id] || null, aliases: al });
  const seen = new Set(Object.values(REC_TO_YC));
  for (const [yc, en, hi, ta] of YO) { if (ents.some((e) => e.yc === yc && !e.rec)) continue; ents.push({ id: "y:" + yc, rec: null, yc, aliases: [...en, ...hi.split("|"), ta], hi: hi.split("|")[0], ta }); seen.add(yc); }
  for (const yc of ctx.meta.crops_all) if (!seen.has(yc)) { const a = yc.toLowerCase().replace(/\(.*?\)/g, "").replace(/&/g, " ").replace(/[^a-z ]/g, " ").replace(/\s+/g, " ").trim(); if (a) ents.push({ id: "y:" + yc, rec: null, yc, aliases: [a], generic: true }); }
  return ents;
}
export function cropLabel(lg, e) {
  if (e.rec) return lg === "en" ? e.rec : REC_LABEL[lg][e.rec] || e.rec;
  if (lg === "hi" && e.hi) return e.hi; if (lg === "ta" && e.ta) return e.ta; return e.yc;
}
// find crops (longest alias first, matched text is blanked so "black gram" does not also match "gram")
function findCrops(s, ents) {
  const all = []; ents.forEach((e) => e.aliases.forEach((a) => all.push([a, e]))); all.sort((x, y) => y[0].length - x[0].length);
  const out = []; let rest = s;
  for (const [a, e] of all) {
    const re = isLatin(a) ? new RegExp("(^|[^a-z])(" + a + ")(?:s|es)?(?![a-z])") : new RegExp("()(" + a + ")");
    const m = re.exec(rest); if (!m) continue;
    const st = m.index + m[1].length, en = m.index + m[0].length;
    out.push({ e, at: st }); rest = rest.slice(0, st) + " ".repeat(en - st) + rest.slice(en);
  }
  out.sort((a, b) => a.at - b.at);
  const seen = new Set(); return out.filter((o) => (seen.has(o.e.id) ? false : seen.add(o.e.id))).map((o) => o.e);
}
function findStates(s, ctx) {
  const out = [];
  for (const st of ctx.meta.states) {
    const al = [st.toLowerCase(), ...(STATE_X[st] || [])];
    if (al.some((a) => hasWord(s, a))) out.push(st);
  }
  // "Madhya Pradesh" vs "Uttar Pradesh": full names only, so no clash
  return out;
}
const fmt = (v, d = 1) => (v == null ? "-" : String(+(+v).toFixed(d)));
const stateL = (lg, st) => (lg === "en" ? st : (STATE_X[st] || []).find((x) => (lg === "hi" ? /[ऀ-ॿ]/ : /[஀-௿]/).test(x)) || st);

/* ------------------------------------------------ answer strings ------------------------------------------------ */
const S = {
  en: {
    src: "Source", cond: (c) => `Usual growing conditions for ${c}`, condNote: "From the crop recommendation dataset. “Usual” = the middle 80% of the farm records for this crop (10th to 90th percentile).",
    cols: ["Factor", "Usual range", "Average"], only: (c, f) => `${c}: ${f}`, rankTitle: (f, d) => `Crops with the ${d === "low" ? "lowest" : "highest"} average ${f}`, rankNote: "Average of the dataset's records for each crop. Showing the top 6 of 22 crops.",
    cmp: (a, b) => `${a} vs ${b}`, cmpNote: "Usual range (10th–90th percentile) from the crop recommendation dataset.",
    priceT: (c) => `Price for ${c}`, msp: "MSP (government minimum support price)", mkt: "Market price (Agmarknet snapshot, 2026-27)", unit: "₹ per quintal", above: (p) => `Market is ${p}% above MSP in this snapshot.`, below: (p) => `Market is ${p}% below MSP in this snapshot.`,
    suspect: "The market figure in the snapshot looks unreliable for this crop (a unit or entry mismatch), so I am not showing it.", noPrice: (c) => `I have no price data for ${c}. Prices in this project cover only: `, priceAll: "MSP and market price snapshot for the crops that have price data",
    yTitle: (c, st) => `Typical ${c} yield${st ? " in " + st : ""}`, yState: (v, n, a) => `Median yield about <b>${v} tonnes per hectare</b> (from ${n} records; typical area ${a} ha).`,
    yAll: (n) => `Across ${n} states in the data`, yTop: "Highest", yLow: "Lowest", yNote: "Median of past records (up to 2020). Units follow the dataset: crops like sugarcane or banana are weighed as whole produce, so do not compare tonnes across different crops.", yNo: (c, st) => `The yield data has no ${c} records for ${st}.`, yNoCrop: (c) => `The yield data has no records for ${c}.`,
    seasonT: (c) => `Seasons recorded for ${c}`, seasonLine: (s, n) => `${s}: in ${n} state${n > 1 ? "s" : ""}`, seasonNote: "These are the seasons written in the yield dataset. I do not have a sowing calendar or exact dates, so please check local advice for when to sow.",
    stateT: (st) => `Most commonly recorded crops in ${st}`, stateNote: "From the yield dataset (most records first). For advice based on your soil and weather, type your 7 values.",
    fertT: "Fertilizer types in the dataset (N-P-K %)", fertHow: "To get a suggestion, type your nutrient values, for example: <i>fertilizer for nitrogen 20, phosphorus 10, potassium 5</i>.",
    fertOut: (v) => `Suggested fertilizer: <b>${v}</b>`, fertNote: "Trained on only 99 rows, using the Kaggle fertilizer dataset's own N, P, K scale (N 4–42, P 0–42, K 0–19), which is not the same as the crop advisor's soil values.", fertWarn: (f, lo, hi) => `${f} is outside the dataset range (${lo} to ${hi}), so treat this with caution.`,
    fertNeed: (m) => `For a fertilizer suggestion I also need: ${m}.`,
    accT: "How accurate is it?", acc: (a, n, t3, cvm, cv) => `The crop model (Random Forest) got <b>${a}%</b> on ${n} unseen test records, and the right crop was in its top 3 for ${t3}% of them. But the data is small and very clean, so real fields will be less certain.`, accY: (r2, e) => `Yield model (Gradient Boosting) tested on future years (2016–2020): R² ${r2} on log yield, median error about ${e}%.`, accF: (a) => `Fertilizer model: ${a}% cross-validated accuracy, but only 99 rows with near rule-like labels.`,
    impT: "What matters most?", impC: "For crop choice (permutation importance):", impY: "For yield, crop type and state explain almost everything; rainfall and fertilizer add very little in this data.",
    dataT: "Which data does this use?", data: ["Crop recommendation data (Kaggle): 2,200 rows, 22 crops, 7 soil/climate values", "Crop yield data (Kaggle, India): 19,408 usable rows, 54 crops, 30 states, to 2020", "Fertilizer data (Kaggle): 99 rows", "Market prices: Agmarknet season-wise snapshot with MSP (2026-27)"],
    limT: "Honest limitations", lim: ["The crop data is small, balanced and very clean; real fields are messier and results get worse when inputs are noisy.", "Yield data ends in 2020 and fertilizer/pesticide/rain are state-year values shared by all crops.", "Fertilizer model uses only 99 rows.", "Prices are one snapshot, not a trend.", "Patterns in past data, not proof of cause and effect."],
    howT: "How does it work?", how: "Your 7 values go into a Random Forest of 50 trees trained on 2,200 records. It outputs a probability for each of 22 crops. The same trained model runs here in your browser with no internet. Yield uses Gradient Boosting, fertilizer uses Logistic Regression.",
    listT: "Crops I know", listA: "Crop advisor (soil and climate model, 22 crops)", listB: "Also in yield and season data", listN: "Ask for any of them, for example “rice price” or “wheat yield in Punjab”.",
    scope: "I cannot answer that from my data. I only use crop soil/climate ranges, yield by state, season records, MSP/market prices, fertilizer types and model facts. For pests, diseases, irrigation schedules, subsidies, loans or weather forecasts, please ask your local agriculture officer or Krishi Vigyan Kendra.",
    helpT: "You can ask me things like", help: ["Type your 7 soil and weather values to get crop advice", "“What conditions does rice need?”", "“Which crops need high rainfall?”", "“Compare rice and maize”", "“MSP of cotton”", "“Wheat yield in Punjab”", "“Which season for maize?”", "“Fertilizer for nitrogen 20, phosphorus 10, potassium 5”", "“How accurate is the model?”"],
    hello: "Hello! I am the CultiVoice assistant. Ask me about crops, yields, prices or fertilizer, or type your 7 soil and weather values for a recommendation.", thanks: "You are welcome!",
    noMatch: "I did not understand that. Try one of the questions below, or type your soil and weather values.", yieldOnly: (c) => `${c} is not in the soil/climate advisor (it covers 22 crops). I only have yield, season and sometimes price data for it:`,
    seeOpen: "Tip: open the Yield tab for a prediction.",
  },
  hi: {
    src: "स्रोत", cond: (c) => `${c} के लिए सामान्य परिस्थितियाँ`, condNote: "फसल सिफारिश डेटासेट से। “सामान्य” = इस फसल के बीच के 80% रिकॉर्ड (10वाँ से 90वाँ प्रतिशतक)।",
    cols: ["कारक", "सामान्य सीमा", "औसत"], only: (c, f) => `${c}: ${f}`, rankTitle: (f, d) => `${f} सबसे ${d === "low" ? "कम" : "ज़्यादा"} चाहने वाली फसलें`, rankNote: "प्रत्येक फसल के रिकॉर्ड का औसत। 22 में से शीर्ष 6 फसलें।",
    cmp: (a, b) => `${a} बनाम ${b}`, cmpNote: "फसल सिफारिश डेटासेट से सामान्य सीमा (10वाँ–90वाँ प्रतिशतक)।",
    priceT: (c) => `${c} का भाव`, msp: "एमएसपी (सरकारी न्यूनतम समर्थन मूल्य)", mkt: "बाज़ार भाव (एगमार्कनेट, 2026-27)", unit: "₹ प्रति क्विंटल", above: (p) => `इस आँकड़े में बाज़ार भाव एमएसपी से ${p}% ऊपर है।`, below: (p) => `इस आँकड़े में बाज़ार भाव एमएसपी से ${p}% नीचे है।`,
    suspect: "इस फसल का बाज़ार आँकड़ा भरोसेमंद नहीं लगता (इकाई या प्रविष्टि की गड़बड़ी), इसलिए नहीं दिखा रहा।", noPrice: (c) => `${c} का भाव डेटा मेरे पास नहीं है। भाव सिर्फ इनके हैं: `, priceAll: "जिन फसलों का भाव डेटा है उनका एमएसपी और बाज़ार भाव",
    yTitle: (c, st) => `${st ? st + " में " : ""}${c} की सामान्य उपज`, yState: (v, n, a) => `माध्य उपज लगभग <b>${v} टन प्रति हेक्टेयर</b> (${n} रिकॉर्ड से; सामान्य क्षेत्र ${a} हे.)।`,
    yAll: (n) => `डेटा के ${n} राज्यों में`, yTop: "सबसे ज़्यादा", yLow: "सबसे कम", yNote: "2020 तक के पुराने रिकॉर्ड का माध्य। इकाइयाँ डेटासेट की हैं: गन्ना, केला जैसी फसलें पूरी उपज के वज़न में हैं, इसलिए अलग-अलग फसलों के टन की तुलना न करें।", yNo: (c, st) => `उपज डेटा में ${st} के लिए ${c} का रिकॉर्ड नहीं है।`, yNoCrop: (c) => `उपज डेटा में ${c} का रिकॉर्ड नहीं है।`,
    seasonT: (c) => `${c} के दर्ज मौसम (सीज़न)`, seasonLine: (s, n) => `${s}: ${n} राज्य में`, seasonNote: "ये उपज डेटासेट में लिखे सीज़न हैं। मेरे पास बुवाई का कैलेंडर या सही तारीख नहीं है, इसलिए बुवाई का समय स्थानीय सलाह से देखें।",
    stateT: (st) => `${st} में सबसे ज़्यादा दर्ज फसलें`, stateNote: "उपज डेटासेट से (ज़्यादा रिकॉर्ड पहले)। अपनी मिट्टी और मौसम के अनुसार सलाह के लिए 7 मान लिखें।",
    fertT: "डेटासेट में उर्वरक के प्रकार (N-P-K %)", fertHow: "सुझाव के लिए पोषक मान लिखें, जैसे: <i>उर्वरक नाइट्रोजन 20, फॉस्फोरस 10, पोटैशियम 5</i>।",
    fertOut: (v) => `सुझाया गया उर्वरक: <b>${v}</b>`, fertNote: "केवल 99 पंक्तियों पर प्रशिक्षित, कैगल उर्वरक डेटा के अपने N, P, K पैमाने (N 4–42, P 0–42, K 0–19) पर; यह फसल सलाह के मिट्टी मानों जैसा नहीं है।", fertWarn: (f, lo, hi) => `${f} डेटासेट की सीमा (${lo} से ${hi}) से बाहर है, सावधानी रखें।`,
    fertNeed: (m) => `उर्वरक सुझाव के लिए मुझे ये भी चाहिए: ${m}।`,
    accT: "यह कितना सटीक है?", acc: (a, n, t3) => `फसल मॉडल (रैंडम फ़ॉरेस्ट) ने ${n} अनदेखे परीक्षण रिकॉर्ड पर <b>${a}%</b> सही दिया, और ${t3}% में सही फसल शीर्ष 3 में थी। लेकिन डेटा छोटा और बहुत साफ़ है, इसलिए असली खेतों में निश्चितता कम होगी।`, accY: (r2, e) => `उपज मॉडल (ग्रेडिएंट बूस्टिंग) भविष्य के वर्षों (2016–2020) पर: लॉग उपज पर R² ${r2}, माध्य त्रुटि लगभग ${e}%।`, accF: (a) => `उर्वरक मॉडल: ${a}% क्रॉस-वैलिडेटेड सटीकता, पर केवल 99 पंक्तियाँ और लगभग नियम जैसे लेबल।`,
    impT: "सबसे ज़्यादा क्या मायने रखता है?", impC: "फसल चुनने के लिए (परम्यूटेशन महत्व):", impY: "उपज के लिए फसल का प्रकार और राज्य लगभग सब कुछ समझाते हैं; इस डेटा में बारिश और उर्वरक बहुत कम जोड़ते हैं।",
    dataT: "यह कौन सा डेटा उपयोग करता है?", data: ["फसल सिफारिश डेटा (कैगल): 2,200 पंक्तियाँ, 22 फसलें, 7 मिट्टी/मौसम मान", "फसल उपज डेटा (कैगल, भारत): 19,408 उपयोगी पंक्तियाँ, 54 फसलें, 30 राज्य, 2020 तक", "उर्वरक डेटा (कैगल): 99 पंक्तियाँ", "बाज़ार भाव: एगमार्कनेट सीज़न-वार आँकड़े और एमएसपी (2026-27)"],
    limT: "ईमानदार सीमाएँ", lim: ["फसल डेटा छोटा, संतुलित और बहुत साफ़ है; असली खेत ज़्यादा गड़बड़ होते हैं और शोर वाले मान पर नतीजे बिगड़ते हैं।", "उपज डेटा 2020 तक है और उर्वरक/कीटनाशक/बारिश राज्य-वर्ष के मान हैं जो सभी फसलों के लिए समान हैं।", "उर्वरक मॉडल केवल 99 पंक्तियों पर है।", "भाव एक ही आँकड़ा हैं, रुझान नहीं।", "ये पुराने डेटा के पैटर्न हैं, कारण-परिणाम का प्रमाण नहीं।"],
    howT: "यह कैसे काम करता है?", how: "आपके 7 मान 2,200 रिकॉर्ड पर प्रशिक्षित 50 पेड़ों वाले रैंडम फ़ॉरेस्ट में जाते हैं। यह 22 फसलों में से हर एक की संभावना देता है। वही मॉडल आपके ब्राउज़र में बिना इंटरनेट चलता है। उपज के लिए ग्रेडिएंट बूस्टिंग और उर्वरक के लिए लॉजिस्टिक रिग्रेशन है।",
    listT: "जिन फसलों को मैं जानता हूँ", listA: "फसल सलाह (मिट्टी और जलवायु मॉडल, 22 फसलें)", listB: "उपज और सीज़न डेटा में भी", listN: "इनमें से किसी के बारे में पूछें, जैसे “चावल का भाव” या “पंजाब में गेहूं की उपज”।",
    scope: "यह मैं अपने डेटा से नहीं बता सकता। मैं केवल फसल की मिट्टी/जलवायु सीमा, राज्य-वार उपज, सीज़न रिकॉर्ड, एमएसपी/बाज़ार भाव, उर्वरक प्रकार और मॉडल की जानकारी उपयोग करता हूँ। कीट, रोग, सिंचाई, सब्सिडी, ऋण या मौसम पूर्वानुमान के लिए अपने कृषि अधिकारी या कृषि विज्ञान केंद्र से पूछें।",
    helpT: "आप मुझसे ऐसे पूछ सकते हैं", help: ["फसल सलाह के लिए अपने 7 मिट्टी और मौसम के मान लिखें", "“चावल को कैसी परिस्थितियाँ चाहिए?”", "“किन फसलों को ज़्यादा बारिश चाहिए?”", "“चावल और मक्का की तुलना करो”", "“कपास का एमएसपी”", "“पंजाब में गेहूं की उपज”", "“मक्का का सीज़न कौन सा?”", "“उर्वरक नाइट्रोजन 20, फॉस्फोरस 10, पोटैशियम 5”", "“मॉडल कितना सटीक है?”"],
    hello: "नमस्ते! मैं CultiVoice सहायक हूँ। फसल, उपज, भाव या उर्वरक के बारे में पूछें, या सलाह के लिए अपने 7 मिट्टी और मौसम के मान लिखें।", thanks: "आपका स्वागत है!",
    noMatch: "मैं समझ नहीं पाया। नीचे के प्रश्नों में से कोई आज़माएँ, या अपने मिट्टी और मौसम के मान लिखें।", yieldOnly: (c) => `${c} मिट्टी/जलवायु सलाह में नहीं है (वह 22 फसलों के लिए है)। इसके लिए मेरे पास केवल उपज, सीज़न और कभी-कभी भाव का डेटा है:`,
    seeOpen: "सुझाव: भविष्यवाणी के लिए उपज टैब खोलें।",
  },
  ta: {
    src: "மூலம்", cond: (c) => `${c} பயிருக்கான வழக்கமான நிலைகள்`, condNote: "பயிர் பரிந்துரை தரவுத்தொகுப்பிலிருந்து. “வழக்கமான” = இந்தப் பயிரின் நடுவிலுள்ள 80% பதிவுகள் (10வது முதல் 90வது சதவீதம்).",
    cols: ["காரணி", "வழக்கமான வரம்பு", "சராசரி"], only: (c, f) => `${c}: ${f}`, rankTitle: (f, d) => `${f} ${d === "low" ? "மிகக் குறைவாக" : "மிக அதிகமாக"} தேவைப்படும் பயிர்கள்`, rankNote: "ஒவ்வொரு பயிரின் பதிவுகளின் சராசரி. 22 இல் முதல் 6 பயிர்கள்.",
    cmp: (a, b) => `${a} மற்றும் ${b} ஒப்பீடு`, cmpNote: "பயிர் பரிந்துரை தரவுத்தொகுப்பிலிருந்து வழக்கமான வரம்பு (10–90 சதவீதம்).",
    priceT: (c) => `${c} விலை`, msp: "எம்எஸ்பி (அரசு குறைந்தபட்ச ஆதரவு விலை)", mkt: "சந்தை விலை (அக்மார்க்நெட், 2026-27)", unit: "₹ ஒரு குவிண்டாலுக்கு", above: (p) => `இந்தத் தரவில் சந்தை விலை எம்எஸ்பியை விட ${p}% அதிகம்.`, below: (p) => `இந்தத் தரவில் சந்தை விலை எம்எஸ்பியை விட ${p}% குறைவு.`,
    suspect: "இந்தப் பயிருக்கான சந்தை எண் நம்பகமாகத் தெரியவில்லை (அலகு அல்லது பதிவு பிழை), அதனால் காட்டவில்லை.", noPrice: (c) => `${c} க்கான விலைத் தரவு என்னிடம் இல்லை. விலைத் தரவு இவற்றுக்கு மட்டும்: `, priceAll: "விலைத் தரவு உள்ள பயிர்களின் எம்எஸ்பி மற்றும் சந்தை விலை",
    yTitle: (c, st) => `${st ? st + " இல் " : ""}${c} வழக்கமான மகசூல்`, yState: (v, n, a) => `இடைநிலை மகசூல் ஏறத்தாழ <b>ஹெக்டேருக்கு ${v} டன்</b> (${n} பதிவுகள்; வழக்கமான பரப்பு ${a} ஹெ.).`,
    yAll: (n) => `தரவில் உள்ள ${n} மாநிலங்களில்`, yTop: "அதிகம்", yLow: "குறைவு", yNote: "2020 வரையிலான பழைய பதிவுகளின் இடைநிலை. அலகுகள் தரவுத்தொகுப்பின்படி: கரும்பு, வாழை போன்றவை முழு விளைபொருள் எடையில் உள்ளன; வெவ்வேறு பயிர்களின் டன்னை ஒப்பிடாதீர்கள்.", yNo: (c, st) => `மகசூல் தரவில் ${st} க்கான ${c} பதிவு இல்லை.`, yNoCrop: (c) => `மகசூல் தரவில் ${c} பதிவு இல்லை.`,
    seasonT: (c) => `${c} க்கான பதிவான பருவங்கள்`, seasonLine: (s, n) => `${s}: ${n} மாநிலங்களில்`, seasonNote: "இவை மகசூல் தரவுத்தொகுப்பில் எழுதப்பட்ட பருவங்கள். என்னிடம் விதைப்பு நாட்காட்டி அல்லது சரியான தேதிகள் இல்லை; விதைக்கும் நேரத்திற்கு உள்ளூர் ஆலோசனையைப் பாருங்கள்.",
    stateT: (st) => `${st} இல் அதிகம் பதிவான பயிர்கள்`, stateNote: "மகசூல் தரவுத்தொகுப்பிலிருந்து (அதிக பதிவுகள் முதலில்). உங்கள் மண், வானிலைக்கான ஆலோசனைக்கு 7 மதிப்புகளை எழுதுங்கள்.",
    fertT: "தரவுத்தொகுப்பில் உள்ள உர வகைகள் (N-P-K %)", fertHow: "பரிந்துரைக்கு ஊட்டச்சத்து மதிப்புகளை எழுதுங்கள், எ.கா: <i>உரம் நைட்ரஜன் 20, பாஸ்பரஸ் 10, பொட்டாசியம் 5</i>.",
    fertOut: (v) => `பரிந்துரைக்கப்படும் உரம்: <b>${v}</b>`, fertNote: "99 வரிசைகளில் மட்டுமே பயிற்சி; கேகிள் உரத் தரவின் சொந்த N, P, K அளவுகோலில் (N 4–42, P 0–42, K 0–19); இது பயிர் ஆலோசனையின் மண் மதிப்புகளைப் போன்றதல்ல.", fertWarn: (f, lo, hi) => `${f} தரவு வரம்புக்கு (${lo} முதல் ${hi}) வெளியே உள்ளது, எச்சரிக்கையுடன் பாருங்கள்.`,
    fertNeed: (m) => `உரப் பரிந்துரைக்கு எனக்கு இவையும் தேவை: ${m}.`,
    accT: "இது எவ்வளவு துல்லியம்?", acc: (a, n, t3) => `பயிர் மாதிரி (ரேண்டம் ஃபாரஸ்ட்) பார்க்காத ${n} சோதனைப் பதிவுகளில் <b>${a}%</b> சரியாக இருந்தது; ${t3}% இல் சரியான பயிர் முதல் 3 இல் இருந்தது. ஆனால் தரவு சிறியதும் மிகத் தூய்மையானதும், எனவே உண்மை வயல்களில் உறுதி குறைவாக இருக்கும்.`, accY: (r2, e) => `மகசூல் மாதிரி (கிரேடியன்ட் பூஸ்டிங்) எதிர்கால ஆண்டுகளில் (2016–2020): பதிவு மகசூலில் R² ${r2}, இடைநிலைப் பிழை ஏறத்தாழ ${e}%.`, accF: (a) => `உர மாதிரி: ${a}% குறுக்கு சரிபார்ப்புத் துல்லியம், ஆனால் 99 வரிசைகள் மட்டுமே, விதி போன்ற லேபிள்கள்.`,
    impT: "எது மிக முக்கியம்?", impC: "பயிர் தேர்வுக்கு (பெர்முட்டேஷன் முக்கியத்துவம்):", impY: "மகசூலுக்கு பயிர் வகையும் மாநிலமும் கிட்டத்தட்ட எல்லாவற்றையும் விளக்குகின்றன; இந்தத் தரவில் மழையும் உரமும் மிகக் குறைவாகவே சேர்க்கின்றன.",
    dataT: "எந்தத் தரவைப் பயன்படுத்துகிறது?", data: ["பயிர் பரிந்துரைத் தரவு (கேகிள்): 2,200 வரிசைகள், 22 பயிர்கள், 7 மண்/வானிலை மதிப்புகள்", "பயிர் மகசூல் தரவு (கேகிள், இந்தியா): 19,408 பயன்படுத்தத்தக்க வரிசைகள், 54 பயிர்கள், 30 மாநிலங்கள், 2020 வரை", "உரத் தரவு (கேகிள்): 99 வரிசைகள்", "சந்தை விலை: அக்மார்க்நெட் பருவவாரி தரவு மற்றும் எம்எஸ்பி (2026-27)"],
    limT: "நேர்மையான வரம்புகள்", lim: ["பயிர் தரவு சிறியது, சமநிலையானது, மிகத் தூய்மையானது; உண்மை வயல்கள் குழப்பமானவை, சத்தமான மதிப்புகளில் முடிவு மோசமாகும்.", "மகசூல் தரவு 2020 வரை; உரம்/பூச்சிக்கொல்லி/மழை மாநில-ஆண்டு மதிப்புகள், எல்லாப் பயிர்களுக்கும் ஒன்றே.", "உர மாதிரி 99 வரிசைகள் மட்டுமே.", "விலைகள் ஒரே ஒரு தரவு, போக்கு அல்ல.", "இவை பழைய தரவின் வடிவங்கள், காரண-விளைவுக்கு ஆதாரம் அல்ல."],
    howT: "இது எப்படி வேலை செய்கிறது?", how: "உங்கள் 7 மதிப்புகள் 2,200 பதிவுகளில் பயிற்சி பெற்ற 50 மரங்கள் கொண்ட ரேண்டம் ஃபாரஸ்டுக்குள் செல்கின்றன. 22 பயிர்களுக்கும் நிகழ்தகவு தருகிறது. அதே மாதிரி உங்கள் உலாவியில் இணையம் இல்லாமல் ஓடுகிறது. மகசூலுக்கு கிரேடியன்ட் பூஸ்டிங், உரத்துக்கு லாஜிஸ்டிக் ரிக்ரெஷன்.",
    listT: "எனக்குத் தெரிந்த பயிர்கள்", listA: "பயிர் ஆலோசனை (மண் மற்றும் காலநிலை மாதிரி, 22 பயிர்கள்)", listB: "மகசூல் மற்றும் பருவத் தரவிலும்", listN: "இவற்றில் எதைப் பற்றியும் கேளுங்கள், எ.கா. “நெல் விலை” அல்லது “பஞ்சாபில் கோதுமை மகசூல்”.",
    scope: "இதை என் தரவிலிருந்து சொல்ல முடியாது. நான் பயிர் மண்/காலநிலை வரம்புகள், மாநிலவாரி மகசூல், பருவப் பதிவுகள், எம்எஸ்பி/சந்தை விலை, உர வகைகள், மாதிரி தகவல்களை மட்டுமே பயன்படுத்துகிறேன். பூச்சி, நோய், பாசனம், மானியம், கடன், வானிலை முன்னறிவிப்புக்கு உங்கள் வேளாண் அலுவலர் அல்லது கிருஷி விஞ்ஞான் கேந்திராவை அணுகுங்கள்.",
    helpT: "இப்படிக் கேட்கலாம்", help: ["பயிர் ஆலோசனைக்கு உங்கள் 7 மண் மற்றும் வானிலை மதிப்புகளை எழுதுங்கள்", "“நெல்லுக்கு என்ன நிலைகள் தேவை?”", "“எந்தப் பயிர்களுக்கு அதிக மழை தேவை?”", "“நெல் மற்றும் மக்காச்சோளம் ஒப்பிடு”", "“பருத்தி எம்எஸ்பி”", "“பஞ்சாபில் கோதுமை மகசூல்”", "“மக்காச்சோளம் எந்தப் பருவம்?”", "“உரம் நைட்ரஜன் 20, பாஸ்பரஸ் 10, பொட்டாசியம் 5”", "“மாதிரி எவ்வளவு துல்லியம்?”"],
    hello: "வணக்கம்! நான் CultiVoice உதவியாளர். பயிர், மகசூல், விலை, உரம் பற்றிக் கேளுங்கள்; அல்லது பரிந்துரைக்கு உங்கள் 7 மண் மற்றும் வானிலை மதிப்புகளை எழுதுங்கள்.", thanks: "நன்றி!",
    noMatch: "எனக்குப் புரியவில்லை. கீழே உள்ள கேள்விகளில் ஒன்றை முயலுங்கள், அல்லது மண், வானிலை மதிப்புகளை எழுதுங்கள்.", yieldOnly: (c) => `${c} மண்/காலநிலை ஆலோசனையில் இல்லை (அது 22 பயிர்களுக்கு). இதற்கு என்னிடம் மகசூல், பருவம், சில நேரம் விலைத் தரவு மட்டுமே உள்ளது:`,
    seeOpen: "குறிப்பு: கணிப்புக்கு மகசூல் தாவலைத் திறக்கவும்.",
  },
};
export const SUGGEST = {
  en: ["What conditions does rice need?", "Which crops need high rainfall?", "Compare rice and maize", "MSP of cotton", "Wheat yield in Punjab", "Which season for maize?", "Fertilizer for nitrogen 20, phosphorus 10, potassium 5", "How accurate is the model?", "What can you do?"],
  hi: ["चावल को कैसी परिस्थितियाँ चाहिए?", "किन फसलों को ज़्यादा बारिश चाहिए?", "चावल और मक्का की तुलना करो", "कपास का एमएसपी", "पंजाब में गेहूं की उपज", "मक्का का सीज़न कौन सा?", "उर्वरक नाइट्रोजन 20, फॉस्फोरस 10, पोटैशियम 5", "मॉडल कितना सटीक है?", "तुम क्या कर सकते हो?"],
  ta: ["நெல்லுக்கு என்ன நிலைகள் தேவை?", "எந்தப் பயிர்களுக்கு அதிக மழை தேவை?", "நெல் மற்றும் மக்காச்சோளம் ஒப்பிடு", "பருத்தி எம்எஸ்பி", "பஞ்சாபில் கோதுமை மகசூல்", "மக்காச்சோளம் எந்தப் பருவம்?", "உரம் நைட்ரஜன் 20, பாஸ்பரஸ் 10, பொட்டாசியம் 5", "மாதிரி எவ்வளவு துல்லியம்?", "நீ என்ன செய்வாய்?"],
};
const src = (lg, x) => `<div class="muted" style="font-size:12px;margin-top:6px">${S[lg].src}: ${x}</div>`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const table = (head, rows) => `<table><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</table>`;
const ul = (items) => `<ul class="why">${items.map((i) => `<li>${i}</li>`).join("")}</ul>`;

/* ------------------------------------------------ answer builders ------------------------------------------------ */
function rng(p) { return `${fmt(p.p10)} – ${fmt(p.p90)}`; }
function aCond(lg, e, feat, ctx) {
  const s = S[lg], pr = ctx.meta.profiles[e.rec], nm = cropLabel(lg, e);
  if (feat) { const p = pr[feat]; return `<b>${esc(s.only(nm, FN[lg][feat]))}</b><br>${esc(s.cols[1])}: <b>${rng(p)}${UNIT[feat]}</b> · ${esc(s.cols[2])}: ${fmt(p.mean)}${UNIT[feat]}${src(lg, "crop recommendation dataset")}`; }
  const rows = FEATS.map((f) => [esc(FN[lg][f]), `${rng(pr[f])}${UNIT[f]}`, `${fmt(pr[f].mean)}${UNIT[f]}`]);
  return `<h3>${esc(s.cond(nm))}</h3>${table(s.cols, rows)}<div class="muted" style="font-size:12.5px;margin-top:6px">${s.condNote}</div>${extraLine(lg, e, ctx)}`;
}
function extraLine(lg, e, ctx) {
  const bits = [], pr = ctx.meta.prices[e.rec];
  if (pr && !pr.suspect) bits.push(`${S[lg].msp}: ₹${fmt(pr.msp, 0)}`);
  return bits.length ? `<div class="muted" style="font-size:12.5px;margin-top:4px">${bits.join(" · ")}</div>` : "";
}
function aRank(lg, f, dir, ctx) {
  const s = S[lg], rows = Object.entries(ctx.meta.profiles).map(([c, p]) => [c, p[f].mean]).sort((a, b) => (dir === "low" ? a[1] - b[1] : b[1] - a[1])).slice(0, 6);
  return `<h3>${esc(s.rankTitle(FN[lg][f], dir))}</h3>${table([s.cols[0] === "Factor" ? "Crop" : lg === "hi" ? "फसल" : "பயிர்", s.cols[2]], rows.map(([c, v]) => [esc(cropLabel(lg, { rec: c })), `${fmt(v)}${UNIT[f]}`]))}<div class="muted" style="font-size:12.5px;margin-top:6px">${s.rankNote}</div>${src(lg, "crop recommendation dataset")}`;
}
function aCompare(lg, a, b, ctx) {
  const s = S[lg], pa = ctx.meta.profiles[a.rec], pb = ctx.meta.profiles[b.rec], na = cropLabel(lg, a), nb = cropLabel(lg, b);
  return `<h3>${esc(s.cmp(na, nb))}</h3>${table([s.cols[0], na, nb], FEATS.map((f) => [esc(FN[lg][f]), `${rng(pa[f])}${UNIT[f]}`, `${rng(pb[f])}${UNIT[f]}`]))}<div class="muted" style="font-size:12.5px;margin-top:6px">${s.cmpNote}</div>`;
}
function aPrice(lg, e, ctx) {
  const s = S[lg], nm = cropLabel(lg, e), p = e.rec ? ctx.meta.prices[e.rec] : null;
  if (!p) return `${esc(s.noPrice(nm))}${Object.keys(ctx.meta.prices).map((c) => esc(cropLabel(lg, { rec: c }))).join(", ")}.`;
  let h = `<h3>${esc(s.priceT(nm))}</h3><ul class="why"><li>${s.msp}: <b>₹${fmt(p.msp, 0)}</b> (${s.unit})</li>`;
  if (p.suspect) h += `</ul><div class="muted">${s.suspect}</div>`;
  else { const g = (p.market / p.msp - 1) * 100; h += `<li>${s.mkt}: <b>₹${fmt(p.market, 0)}</b></li></ul><div class="muted">${g >= 0 ? s.above(fmt(g, 0)) : s.below(fmt(-g, 0))}</div>`; }
  return h + src(lg, `Agmarknet + MSP · ${esc(p.commodity)}`);
}
function aPriceAll(lg, ctx) {
  const s = S[lg];
  return `<h3>${s.priceAll}</h3>${table(["", "MSP", lg === "en" ? "Market" : s.mkt.split(" (")[0]], Object.entries(ctx.meta.prices).map(([c, p]) => [esc(cropLabel(lg, { rec: c })), `₹${fmt(p.msp, 0)}`, p.suspect ? "—" : `₹${fmt(p.market, 0)}`]))}<div class="muted" style="font-size:12.5px;margin-top:6px">${s.unit}${Object.values(ctx.meta.prices).some((p) => p.suspect) ? " · " + s.suspect : ""}</div>`;
}
function yRows(ctx, yc) { const r = []; for (const [st, d] of Object.entries(ctx.yd)) { const c = d.crops && d.crops[yc]; if (c) r.push({ st, ...c }); } return r; }
function aYield(lg, e, state, ctx) {
  const s = S[lg], nm = cropLabel(lg, e);
  if (!e.yc) return esc(s.yNoCrop(nm));
  const rows = yRows(ctx, e.yc);
  if (!rows.length) return esc(s.yNoCrop(nm));
  if (state) {
    const r = rows.find((x) => x.st === state), sn = stateL(lg, state);
    if (!r) return esc(s.yNo(nm, sn));
    return `<h3>${esc(s.yTitle(nm, sn))}</h3><p>${s.yState(fmt(r.median_yield, 2), r.n, fmt(r.median_area, 0))}</p><div class="muted" style="font-size:12.5px">${s.yNote}</div><div class="muted" style="font-size:12.5px">${s.seeOpen}</div>${src(lg, "crop yield dataset")}`;
  }
  const ok = rows.filter((r) => r.n >= 5), use = (ok.length >= 4 ? ok : rows).slice().sort((a, b) => b.median_yield - a.median_yield);
  const top = use.slice(0, 3), low = use.slice(-3).reverse(), line = (r) => `${esc(stateL(lg, r.st))}: <b>${fmt(r.median_yield, 2)}</b> t/ha`;
  const same = use.length <= 3;
  return `<h3>${esc(s.yTitle(nm, ""))}</h3><div class="muted">${s.yAll(use.length)}</div><b>${s.yTop}</b>${ul(top.map(line))}${same ? "" : `<b>${s.yLow}</b>${ul(low.map(line))}`}<div class="muted" style="font-size:12.5px">${s.yNote}</div>${src(lg, "crop yield dataset")}`;
}
function aSeason(lg, e, ctx) {
  const s = S[lg], nm = cropLabel(lg, e); if (!e.yc) return esc(s.yNoCrop(nm));
  const rows = yRows(ctx, e.yc); if (!rows.length) return esc(s.yNoCrop(nm));
  const cnt = {}; rows.forEach((r) => r.seasons.forEach((x) => (cnt[x] = (cnt[x] || 0) + 1)));
  return `<h3>${esc(s.seasonT(nm))}</h3>${ul(Object.entries(cnt).sort((a, b) => b[1] - a[1]).map(([k, n]) => esc(s.seasonLine(k, n))))}<div class="muted" style="font-size:12.5px">${s.seasonNote}</div>${src(lg, "crop yield dataset")}`;
}
function aState(lg, state, ctx) {
  const s = S[lg], d = ctx.yd[state]; if (!d) return "";
  const rows = Object.entries(d.crops).sort((a, b) => b[1].n - a[1].n).slice(0, 8);
  return `<h3>${esc(s.stateT(stateL(lg, state)))}</h3>${ul(rows.map(([c, v]) => `${esc(cropLabel(lg, ctx.ents.find((e) => e.yc === c) || { yc: c }))}: <b>${fmt(v.median_yield, 2)}</b> t/ha`))}<div class="muted" style="font-size:12.5px">${s.stateNote}</div>${src(lg, "crop yield dataset")}`;
}
function aFertInfo(lg, ctx) {
  const s = S[lg];
  return `<h3>${s.fertT}</h3>${table(["", "N-P-K"], ctx.fert.classes.map((c) => [esc(c), esc(ctx.fert.composition_NPK[c])]))}<div class="muted" style="margin-top:6px">${s.fertHow}</div><div class="muted" style="font-size:12.5px;margin-top:4px">${s.fertNote}</div>${src(lg, "fertilizer dataset")}`;
}
function aFert(lg, vals, ctx) {
  const s = S[lg], map = { N: "Nitrogen", P: "Phosphorus", K: "Potassium" }, miss = ["N", "P", "K"].filter((k) => !(k in vals));
  if (miss.length) return `${esc(s.fertNeed(miss.map((k) => FN[lg][k]).join(", ")))}<br><span class="muted">${s.fertHow}</span>`;
  const input = {}; for (const k of ["N", "P", "K"]) input[map[k]] = vals[k];
  const r = recommendFertilizer(ctx.fert, input), best = r[0], warn = [];
  for (const k of ["N", "P", "K"]) { const st = ctx.meta.fert_stats[map[k]]; if (st && (vals[k] < st.min || vals[k] > st.max)) warn.push(s.fertWarn(FN[lg][k], st.min, st.max)); }
  return `<p>${s.fertOut(esc(best.fertilizer) + ` (N-P-K ${esc(best.npk)})`)}, ${fmt(best.prob * 100, 0)}%</p>${warn.map((w) => `<div class="muted">⚠ ${esc(w)}</div>`).join("")}<div class="muted" style="font-size:12.5px;margin-top:4px">${s.fertNote}</div>${src(lg, "fertilizer dataset (Logistic Regression)")}`;
}
function aModel(lg, kind, ctx) {
  const s = S[lg], m = ctx.metrics;
  if (kind === "acc") {
    const f = m.crop.final, yg = m.yield.comparison.find((r) => /Future/.test(r.Split) && r.Model === "Gradient Boosting"), fe = m.fertilizer.meta;
    return `<h3>${s.accT}</h3>${ul([s.acc(fmt(f.test_accuracy * 100), f.n_test, fmt(f.top3_accuracy * 100, 0)), yg ? s.accY(fmt(yg["R2 (log scale)"], 2), fmt(yg["Median abs % error"], 0)) : "", s.accF(fmt(fe.cv_accuracy * 100))].filter(Boolean))}${src(lg, "model metrics (results/ folder)")}`;
  }
  if (kind === "imp") return `<h3>${s.impT}</h3><div>${s.impC}</div>${ul(m.crop.importance.slice(0, 4).map((x) => `<b>${esc(FN[lg][x.feature] || x.feature)}</b>: ${fmt(x.perm_importance, 2)}`))}<div class="muted">${s.impY}</div>${src(lg, "model metrics")}`;
  if (kind === "data") return `<h3>${s.dataT}</h3>${ul(s.data.map(esc))}`;
  if (kind === "lim") return `<h3>${s.limT}</h3>${ul(s.lim.map(esc))}`;
  return `<h3>${s.howT}</h3><p>${esc(s.how)}</p>`;
}
function aList(lg, ctx) {
  const s = S[lg], rec = Object.keys(ctx.meta.profiles).map((c) => esc(cropLabel(lg, { rec: c }))).join(", "), extra = ctx.meta.crops_all.length;
  return `<h3>${s.listT}</h3><p><b>${s.listA}</b>: ${rec}.</p><p><b>${s.listB}</b>: ${extra} ${lg === "en" ? "crops (wheat, sugarcane, groundnut, potato, onion, and more)" : lg === "hi" ? "फसलें (गेहूं, गन्ना, मूंगफली, आलू, प्याज आदि)" : "பயிர்கள் (கோதுமை, கரும்பு, நிலக்கடலை, உருளைக்கிழங்கு, வெங்காயம் உள்ளிட்டவை)"}.</p><div class="muted">${s.listN}</div>`;
}

/* ------------------------------------------------ main entry ------------------------------------------------ */
// ctx = { meta, yd, fert, metrics }. Returns { html, intent } or null when the text is not a question this module can handle.
export function answerQuestion(text, lg, ctx) {
  if (!ctx.ents) ctx.ents = buildEntities(ctx);
  const raw = String(text).trim(); if (!raw) return null;
  const s = normalize(raw).replace(/[?!.,;:()"“”]/g, " ").replace(/\s+/g, " ").trim();
  const crops = findCrops(s, ctx.ents), states = findStates(s, ctx), St = S[lg];
  const featHits = []; FEATKW.forEach(([f, dir, kws]) => { if (kws.some((k) => hasWord(s, k))) featHits.push({ f, dir }); });
  const explicitFeat = featHits.find((h) => !h.dir) || featHits[0] || null;
  const dirWord = hasAny(s, W.low) && !hasAny(s, ["lowest rainfall"]) ? "low" : hasAny(s, W.high) ? "high" : null;
  const dir = (explicitFeat && explicitFeat.dir) || (hasAny(s, W.low) ? "low" : "high");
  const ret = (html, intent) => ({ html, intent });
  const parsed = parseMessage(raw, null);

  if (hasAny(s, W.scope)) return ret(`${esc(St.scope)}`, "scope");
  if (hasAny(s, W.fert)) {
    const has = ["N", "P", "K"].some((k) => k in parsed.values);
    return has ? ret(aFert(lg, parsed.values, ctx), "fert") : ret(aFertInfo(lg, ctx), "fertinfo");
  }
  if (!crops.length && !states.length) {
    if (hasAny(s, W.acc)) return ret(aModel(lg, "acc", ctx), "model");
    if (hasAny(s, W.limit)) return ret(aModel(lg, "lim", ctx), "model");
    if (hasAny(s, W.imp) && !explicitFeat) return ret(aModel(lg, "imp", ctx), "model");
    if (hasAny(s, W.data) && !hasAny(s, W.yield)) return ret(aModel(lg, "data", ctx), "model");
    if (hasAny(s, W.how)) return ret(aModel(lg, "how", ctx), "model");
  }
  if (hasAny(s, W.price)) {
    if (!crops.length) return ret(aPriceAll(lg, ctx), "price");
    return ret(crops.slice(0, 3).map((e) => aPrice(lg, e, ctx)).join("<hr>"), "price");
  }
  if (hasAny(s, W.season) && crops.length && !(explicitFeat && !hasAny(s, ["season", "kharif", "rabi", "sow", "when", "सीजन", "सीज़न", "खरीफ", "रबी", "बुवाई", "कब", "பருவம்", "சீசன்", "காரீப்", "ராபி", "விதைப்பு", "எப்போது"])))
    return ret(crops.slice(0, 3).map((e) => aSeason(lg, e, ctx)).join("<hr>"), "season");
  if (hasAny(s, W.yield) && (crops.length || states.length)) {
    if (!crops.length) return ret(aState(lg, states[0], ctx), "state");
    return ret(crops.slice(0, 3).map((e) => aYield(lg, e, states[0] || null, ctx)).join("<hr>"), "yield");
  }
  const recCrops = crops.filter((c) => c.rec);
  if (recCrops.length >= 2 && (hasAny(s, ["compar", "vs", "versus", "differ", "between", "or", "बनाम", "तुलना", "अंतर", "ஒப்பீடு", "வேறுபாடு"]) || crops.length === 2)) return ret(aCompare(lg, recCrops[0], recCrops[1], ctx), "compare");
  if (!crops.length && explicitFeat) return ret(aRank(lg, explicitFeat.f, dir, ctx), "rank");
  if (crops.length) {
    const e = crops[0];
    if (e.rec) return ret(aCond(lg, e, crops.length === 1 && explicitFeat ? explicitFeat.f : null, ctx), "cond");
    const y = aYield(lg, e, states[0] || null, ctx), se = aSeason(lg, e, ctx), nm = cropLabel(lg, e);
    return ret(`<p>${esc(St.yieldOnly(nm))}</p>${y}<hr>${se}`, "yieldonly");
  }
  if (states.length) return ret(aState(lg, states[0], ctx), "state");
  if (hasAny(s, W.list)) return ret(aList(lg, ctx), "list");
  if (hasAny(s, W.help)) return ret(`<h3>${St.helpT}</h3>${ul(St.help.map(esc))}`, "help");
  if (hasAny(s, W.thanks)) return ret(esc(St.thanks), "thanks");
  if (W.hello.some((k) => hasWord(s, k)) && s.split(" ").length <= 4) return ret(esc(St.hello), "hello");
  return null;
}
export const NO_MATCH = (lg) => S[lg].noMatch;
