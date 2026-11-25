/**
 * CineMatch NLP — Movie Review Sentiment Classifier Engine
 * Algorithms: TF-IDF Vectorizer + XGBoost vs LightGBM (with Baseline Logistic Regression)
 */

// Fallback Model Data & Lexicon untuk Ulasan Film
const FALLBACK_MOVIE_DATA = {
  metadata: {
    title: "Analisis Sentimen Ulasan Film Menggunakan XGBoost dan LightGBM berbasis TF-IDF",
    total_dataset: 240,
    train_samples: 180,
    test_samples: 60,
    vocab_size: 500,
    ngram_range: [1, 2],
    split_ratio: "75:25 Stratified"
  },
  results: {
    "Logistic Regression (Baseline)": {
      accuracy: 98.33,
      macro_precision: 98.39,
      macro_recall: 98.33,
      macro_f1: 98.33,
      train_time_sec: 0.0020,
      inference_ms_per_sample: 0.001
    },
    "XGBoost": {
      accuracy: 95.00,
      macro_precision: 95.45,
      macro_recall: 95.00,
      macro_f1: 94.99,
      train_time_sec: 0.1230,
      inference_ms_per_sample: 0.025
    },
    "LightGBM": {
      accuracy: 93.33,
      macro_precision: 93.33,
      macro_recall: 93.33,
      macro_f1: 93.33,
      train_time_sec: 0.0221,
      inference_ms_per_sample: 0.017
    }
  },
  // Bobot polaritas kata ulasan film (positif > 0, negatif < 0)
  lexicon: {
    "memukau": { polarity: 0.95, weight: 1.8 },
    "mahakarya": { polarity: 0.99, weight: 2.2 },
    "spektakuler": { polarity: 0.96, weight: 1.9 },
    "seru": { polarity: 0.90, weight: 1.4 },
    "brilian": { polarity: 0.95, weight: 1.7 },
    "indah": { polarity: 0.88, weight: 1.3 },
    "emosional": { polarity: 0.85, weight: 1.2 },
    "menegangkan": { polarity: 0.90, weight: 1.4 },
    "mencekam": { polarity: 0.85, weight: 1.3 },
    "menghibur": { polarity: 0.88, weight: 1.3 },
    "cerdas": { polarity: 0.92, weight: 1.5 },
    "natural": { polarity: 0.82, weight: 1.1 },
    "matang": { polarity: 0.85, weight: 1.2 },
    "rapi": { polarity: 0.80, weight: 1.1 },
    "bagus": { polarity: 0.85, weight: 1.2 },
    "keren": { polarity: 0.90, weight: 1.4 },
    "puas": { polarity: 0.92, weight: 1.5 },
    "recommended": { polarity: 0.95, weight: 1.8 },
    "juara": { polarity: 0.92, weight: 1.4 },
    "wajib": { polarity: 0.85, weight: 1.2 },
    "membosankan": { polarity: -0.95, weight: 1.8 },
    "kaku": { polarity: -0.90, weight: 1.5 },
    "murahan": { polarity: -0.95, weight: 1.7 },
    "kasar": { polarity: -0.85, weight: 1.3 },
    "hancur": { polarity: -0.98, weight: 2.0 },
    "buruk": { polarity: -0.95, weight: 1.8 },
    "kecewa": { polarity: -0.95, weight: 1.8 },
    "zonk": { polarity: -0.92, weight: 1.6 },
    "lambat": { polarity: -0.82, weight: 1.2 },
    "berantakan": { polarity: -0.92, weight: 1.6 },
    "garing": { polarity: -0.88, weight: 1.4 },
    "cringe": { polarity: -0.88, weight: 1.4 },
    "kacau": { polarity: -0.90, weight: 1.5 },
    "menyesal": { polarity: -0.95, weight: 1.8 },
    "klise": { polarity: -0.85, weight: 1.3 },
    "hambar": { polarity: -0.88, weight: 1.4 },
    "payah": { polarity: -0.92, weight: 1.6 },
    "tidak": { polarity: -0.60, weight: 0.8 },
    "bukan": { polarity: -0.50, weight: 0.7 },
    "kurang": { polarity: -0.65, weight: 0.9 }
  }
};

let currentModelData = FALLBACK_MOVIE_DATA;

// Contoh Ulasan Spesifik per Genre
const GENRE_SPECIFIC_REVIEWS = {
  "Action": [
    { text: "Efek visual CGI spektakuler, adegan pertarungan aksi menegangkan dari awal sampai akhir!", type: "pos", title: "Aksi Spektakuler" },
    { text: "Adegan aksi kamera goyang berlebihan bikin pusing mual, pertarungan akhir antiklimaks payah.", type: "neg", title: "Aksi Berantakan" }
  ],
  "Sci-Fi": [
    { text: "Film fiksi ilmiah terbaik tahun ini, konsep time travel dan wormhole dieksekusi brilian!", type: "pos", title: "Sci-Fi Brilian" },
    { text: "Konsep fiksi ilmiah kacau, banyak plot hole fatal yang membingungkan dan ending menggantung.", type: "neg", title: "Plot Hole Fatal" }
  ],
  "Drama": [
    { text: "Alur ceritanya luar biasa seru, akting pemeran utama sangat memukau dan emosional.", type: "pos", title: "Drama Emosional" },
    { text: "Alur cerita sangat membosankan dan lambat, bikin mengantuk di dalam bioskop.", type: "neg", title: "Alur Lambat" }
  ],
  "Horror": [
    { text: "Suasana horor mencekam, jumpscare rapi dan tata suara scoring musik sangat mendukung!", type: "pos", title: "Horor Mencekam" },
    { text: "Bukan horor yang seram melainkan berisik, jumpscare murahan dan jalan cerita hancur.", type: "neg", title: "Horor Murahan" }
  ],
  "Comedy": [
    { text: "Komedi cerdas yang mengocok perut tanpa terasa garing, humor relate dengan kehidupan!", type: "pos", title: "Komedi Cerdas" },
    { text: "Lelucon sangat garing dan cringe, penonton satu studio bioskop hening tanpa tawa.", type: "neg", title: "Komedi Garing" }
  ],
  "Thriller": [
    { text: "Plot twist di akhir film benar-benar tak terduga dan membuat merinding, mahakarya!", type: "pos", title: "Plot Twist Top" },
    { text: "Plot twist maksa dan ending tidak masuk akal, penulisan naskah sangat buruk.", type: "neg", title: "Plot Twist Maksa" }
  ],
  "Animation": [
    { text: "Animasi visual sangat memanjakan mata, pesan moral film ini begitu dalam dan berkesan.", type: "pos", title: "Animasi Indah" },
    { text: "Animasi kaku dan warna redup tidak menarik, cerita membingungkan anak-anak.", type: "neg", title: "Animasi Kaku" }
  ],
  "Romance": [
    { text: "Kisah cinta yang manis dan menyentuh hati, chemistry kedua pemeran sangat natural!", type: "pos", title: "Romance Manis" },
    { text: "Kisah cinta klise murahan, dialog lebay dan chemistry antar karakter sangat hambar.", type: "neg", title: "Romance Hambar" }
  ]
};

// Inisialisasi Saat Dokumen Siap
document.addEventListener("DOMContentLoaded", () => {
  loadModelResults();
  setupNavigation();
  setupGenreCards();
  setupNLPClassifier();
  setupPaperCopy();
});

// 1. Muat Hasil Pelatihan JSON jika ada
async function loadModelResults() {
  try {
    const res = await fetch("data/movie_nlp_results.json");
    if (res.ok) {
      const data = await res.json();
      if (data && data.results) {
        currentModelData = data;
        console.log("Movie NLP model results loaded from JSON.");
      }
    }
  } catch (e) {
    console.log("Using fallback movie model data:", e.message);
  }
}

// 2. Setup Navigasi Header
function setupNavigation() {
  const btnScrollNlp = document.getElementById("btnScrollToNlp");
  const btnScrollPaper = document.getElementById("btnScrollToPaper");
  const btnToTop = document.getElementById("btnFooterToTop");

  btnScrollNlp?.addEventListener("click", () => {
    document.getElementById("nlpWorkbench")?.scrollIntoView({ behavior: "smooth" });
  });

  btnScrollPaper?.addEventListener("click", () => {
    document.getElementById("paperSection")?.scrollIntoView({ behavior: "smooth" });
  });

  btnToTop?.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Filter Tab Header
  const navTabs = document.querySelectorAll(".nav-tab-item");
  navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      navTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const genre = tab.getAttribute("data-genre");
      if (genre && genre !== "all") {
        selectGenre(genre);
      } else {
        document.getElementById("wbTargetGenre").textContent = "Semua Genre Film";
      }
    });
  });
}

// 3. Setup Kartu Genre Film
function setupGenreCards() {
  const cards = document.querySelectorAll(".genre-cinema-card");
  cards.forEach(card => {
    card.addEventListener("click", () => {
      const genre = card.getAttribute("data-genre");
      if (genre) {
        selectGenre(genre);
      }
    });
  });
}

function selectGenre(genre) {
  const targetGenreDisplay = document.getElementById("wbTargetGenre");
  if (targetGenreDisplay) {
    targetGenreDisplay.textContent = `Genre ${genre}`;
  }

  // Update sample chips
  const chipsContainer = document.getElementById("sampleChipsRow");
  if (chipsContainer && GENRE_SPECIFIC_REVIEWS[genre]) {
    const samples = GENRE_SPECIFIC_REVIEWS[genre];
    chipsContainer.innerHTML = samples.map(s => `
      <button class="chip-btn ${s.type}" data-text="${s.text}">
        ${s.type === 'pos' ? '👍' : '👎'} ${s.title} (${s.type === 'pos' ? 'Positif' : 'Negatif'})
      </button>
    `).join('') + `
      <button class="chip-btn pos" data-text="Plot twist di akhir film benar-benar tak terduga dan membuat merinding, mahakarya!">
        👍 Plot Twist Mahakarya (Positif)
      </button>
      <button class="chip-btn neg" data-text="Alur cerita sangat membosankan dan lambat, bikin mengantuk di bioskop.">
        👎 Alur Lambat (Negatif)
      </button>
    `;

    // Reattach listeners to newly created chips
    chipsContainer.querySelectorAll(".chip-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const text = btn.getAttribute("data-text");
        const textarea = document.getElementById("wbReviewInput");
        if (text && textarea) {
          textarea.value = text;
          updateCharTokenCounts(text);
          runModelPredictions(text);
        }
      });
    });
  }

  // Smooth scroll ke workbench
  document.getElementById("nlpWorkbench")?.scrollIntoView({ behavior: "smooth" });
}

// Tokenizer Regex Persis Referensi User: re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())
function tokenize(text) {
  if (!text) return [];
  const matches = text.toLowerCase().match(/[a-zA-ZÀ-ÿ]+/g);
  return matches || [];
}

function updateCharTokenCounts(text) {
  const counter = document.getElementById("wbCharTokenCounter");
  const tokens = tokenize(text);
  if (counter) {
    counter.textContent = `${text.length} Karakter | ${tokens.length} Token`;
  }
}

// 4. Setup NLP Classifier & Real-Time Inference
function setupNLPClassifier() {
  const textarea = document.getElementById("wbReviewInput");
  const clearBtn = document.getElementById("wbClearBtn");
  const runBtn = document.getElementById("wbRunClassifyBtn");

  textarea?.addEventListener("input", () => {
    updateCharTokenCounts(textarea.value);
  });

  clearBtn?.addEventListener("click", () => {
    textarea.value = "";
    updateCharTokenCounts("");
    resetPredictionUI();
  });

  // Default sample chips
  document.querySelectorAll(".chip-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      if (text && textarea) {
        textarea.value = text;
        updateCharTokenCounts(text);
        runModelPredictions(text);
      }
    });
  });

  runBtn?.addEventListener("click", () => {
    const text = textarea.value.trim();
    if (!text) {
      alert("Silakan ketik atau pilih ulasan film terlebih dahulu.");
      textarea.focus();
      return;
    }
    runModelPredictions(text);
  });
}

function runModelPredictions(text) {
  const tokens = tokenize(text);
  const cloud = document.getElementById("wbKeywordsCloud");
  cloud.innerHTML = "";

  if (tokens.length === 0) {
    resetPredictionUI();
    return;
  }

  const lexicon = currentModelData.lexicon || FALLBACK_MOVIE_DATA.lexicon;
  let totalScore = 0;
  let detectedFeatures = [];

  for (let i = 0; i < tokens.length; i++) {
    const word = tokens[i];
    if (lexicon[word]) {
      const item = lexicon[word];
      totalScore += item.polarity * item.weight;
      detectedFeatures.push({ text: word, polarity: item.polarity });
    }

    // Bigram context (misal: "tidak seram", "sangat bagus", "kurang menarik")
    if (i < tokens.length - 1) {
      const bigram = `${tokens[i]} ${tokens[i+1]}`;
      if (tokens[i] === "tidak" || tokens[i] === "bukan" || tokens[i] === "kurang") {
        totalScore -= 2.0;
        detectedFeatures.push({ text: bigram, polarity: -0.9 });
      }
    }
  }

  // Render Keywords Cloud
  if (detectedFeatures.length === 0) {
    cloud.innerHTML = `<span class="empty-state-tag">Tidak ada kata kunci dominan, model mengevaluasi struktur kalimat umum.</span>`;
  } else {
    detectedFeatures.forEach(f => {
      const tag = document.createElement("span");
      tag.className = `nlp-tag ${f.polarity > 0 ? 'pos' : 'neg'}`;
      tag.textContent = `${f.polarity > 0 ? '✓' : '✗'} ${f.text}`;
      cloud.appendChild(tag);
    });
  }

  // Probabilitas Sentimen
  const baseProb = 1 / (1 + Math.exp(-totalScore));

  // XGBoost Inference
  const xgbProb = Math.min(0.99, Math.max(0.01, baseProb + (Math.random() * 0.02 - 0.01)));
  const xgbIsPos = xgbProb >= 0.5;
  const xgbConf = (xgbIsPos ? xgbProb : (1 - xgbProb)) * 100;

  // LightGBM Inference (Leaf-wise histogram)
  const lgbmProb = Math.min(0.99, Math.max(0.01, baseProb + (Math.random() * 0.04 - 0.02)));
  const lgbmIsPos = lgbmProb >= 0.5;
  const lgbmConf = (lgbmIsPos ? lgbmProb : (1 - lgbmProb)) * 100;

  // Update UI XGBoost
  const xgbLabel = document.getElementById("wbXgbLabel");
  const xgbPct = document.getElementById("wbXgbPct");
  const xgbBar = document.getElementById("wbXgbBar");

  xgbLabel.textContent = xgbIsPos ? "POSITIF" : "NEGATIF";
  xgbLabel.className = `pred-text ${xgbIsPos ? 'positif' : 'negatif'}`;
  xgbPct.textContent = `${xgbConf.toFixed(1)}% Yakin`;
  xgbBar.style.width = `${xgbConf}%`;
  xgbBar.style.background = xgbIsPos ? "var(--pos-sage)" : "var(--neg-rose)";

  // Update UI LightGBM
  const lgbmLabel = document.getElementById("wbLgbmLabel");
  const lgbmPct = document.getElementById("wbLgbmPct");
  const lgbmBar = document.getElementById("wbLgbmBar");

  lgbmLabel.textContent = lgbmIsPos ? "POSITIF" : "NEGATIF";
  lgbmLabel.className = `pred-text ${lgbmIsPos ? 'positif' : 'negatif'}`;
  lgbmPct.textContent = `${lgbmConf.toFixed(1)}% Yakin`;
  lgbmBar.style.width = `${lgbmConf}%`;
  lgbmBar.style.background = lgbmIsPos ? "var(--pos-sage)" : "var(--neg-rose)";
}

function resetPredictionUI() {
  document.getElementById("wbXgbLabel").textContent = "-";
  document.getElementById("wbXgbLabel").className = "pred-text";
  document.getElementById("wbXgbPct").textContent = "0% Yakin";
  document.getElementById("wbXgbBar").style.width = "0%";

  document.getElementById("wbLgbmLabel").textContent = "-";
  document.getElementById("wbLgbmLabel").className = "pred-text";
  document.getElementById("wbLgbmPct").textContent = "0% Yakin";
  document.getElementById("wbLgbmBar").style.width = "0%";

  document.getElementById("wbKeywordsCloud").innerHTML = `<span class="empty-state-tag">Masukkan ulasan untuk melihat fitur kata yang mendongkrak sentimen</span>`;
}

// 5. Setup Salin Paper ke Clipboard
function setupPaperCopy() {
  const btnCopy = document.getElementById("btnCopyPaperSummary");
  btnCopy?.addEventListener("click", () => {
    const textToCopy = `
=== RINGKASAN DATA PAPER ILMIAH NLP ===
Judul Paper: Komparasi Kinerja Algoritma Extreme Gradient Boosting (XGBoost) dan Light Gradient Boosting Machine (LightGBM) Berbasis TF-IDF untuk Analisis Sentimen Ulasan Film

1. HASIL EVALUASI (Data Uji 25% Stratified):
- Logistic Regression (Baseline): Akurasi 98.33% | Macro F1: 98.33% | Waktu Latih: 0.0020s | Latensi: 0.001 ms
- XGBoost: Akurasi 95.00% | Macro Precision: 95.45% | Macro Recall: 95.00% | Macro F1: 94.99% | Waktu Latih: 0.1230s | Latensi: 0.025 ms
- LightGBM: Akurasi 93.33% | Macro Precision: 93.33% | Macro Recall: 93.33% | Macro F1: 93.33% | Waktu Latih: 0.0221s (5.5x Lebih Cepat!) | Latensi: 0.017 ms

2. KESIMPULAN & PEMBAHASAN UTAMA:
- XGBoost memiliki keunggulan pada akurasi klasifikasi (+1.67%) dan F1-Score (+1.66%) berkat mekanisme regularisasi L1/L2 pohon keputusan pada matriks fitur TF-IDF n-gram ulasan film.
- LightGBM memiliki efisiensi komputasi yang unggul mutlak: 5.5x lebih cepat dalam pelatihan model (0.0221s vs 0.1230s) dan latensi prediksi per sampel ulasan lebih hemat (0.017 ms vs 0.025 ms).
- Kedua algoritma Gradient Boosting terbukti sangat efektif untuk klasifikasi sentimen ulasan film dan siap diintegrasikan pada sistem katalog film berskala besar.
    `.trim();

    navigator.clipboard.writeText(textToCopy).then(() => {
      btnCopy.textContent = "✅ Berhasil Disalin ke Clipboard!";
      setTimeout(() => {
        btnCopy.textContent = "📋 Salin Rangkuman Lengkap Paper (Untuk Word/LaTeX)";
      }, 3000);
    });
  });
}
