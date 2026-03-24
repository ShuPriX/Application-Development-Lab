# CineInsight: AI-Powered Movie Review Sentiment Analysis System

## Project Report

---

## Table of Contents

1. [Machine Learning System (50%)](#1-machine-learning-system)
   - 1.1 [System Overview](#11-system-overview)
   - 1.2 [Model 1: BERT Sentiment Classifier](#12-model-1-bert-sentiment-classifier)
   - 1.3 [Model 2: BiLSTM Aspect-Based Sentiment Analyzer](#13-model-2-bilstm-aspect-based-sentiment-analyzer)
   - 1.4 [Text Preprocessing Pipeline](#14-text-preprocessing-pipeline)
   - 1.5 [Post-Prediction Aggregation Engine](#15-post-prediction-aggregation-engine)
   - 1.6 [Inference Pipeline Integration](#16-inference-pipeline-integration)
   - 1.7 [Fallback and Graceful Degradation Strategy](#17-fallback-and-graceful-degradation-strategy)
   - 1.8 [Literature Review and Comparative Analysis](#18-literature-review-and-comparative-analysis)
2. [Full-Stack Application (20%)](#2-full-stack-application)
   - 2.1 [System Architecture](#21-system-architecture)
   - 2.2 [Backend: FastAPI + SQLAlchemy](#22-backend-fastapi--sqlalchemy)
   - 2.3 [Frontend: React + TypeScript](#23-frontend-react--typescript)
   - 2.4 [Real-Time Communication: WebSocket Pipeline](#24-real-time-communication-websocket-pipeline)
   - 2.5 [Data Pipeline: Scraping to Visualization](#25-data-pipeline-scraping-to-visualization)
3. [Deployment (30%)](#3-deployment)
   - 3.1 [Infrastructure Overview](#31-infrastructure-overview)
   - 3.2 [Dockerization Strategy](#32-dockerization-strategy)
   - 3.3 [Docker Compose Orchestration](#33-docker-compose-orchestration)
   - 3.4 [Step-by-Step Deployment Guide](#34-step-by-step-deployment-guide)
   - 3.5 [Firewall and Network Configuration](#35-firewall-and-network-configuration)
   - 3.6 [Environment Variables and Secrets](#36-environment-variables-and-secrets)
   - 3.7 [Operational Management](#37-operational-management)
   - 3.8 [Architecture Decisions and Trade-offs](#38-architecture-decisions-and-trade-offs)

---

## 1. Machine Learning System

### 1.1 System Overview

CineInsight employs a **dual-model architecture** for comprehensive movie review analysis. Rather than using a single monolithic model, the system decouples two distinct NLP tasks into specialized models:

| Model | Task | Architecture | Training Data | Output |
|-------|------|-------------|--------------|--------|
| **BERT Sentiment** | Overall sentiment classification | bert-base-uncased (110M params) | IMDB 50K (binary), neutral via thresholding | positive / negative / neutral + confidence |
| **BiLSTM Aspect** | Aspect-based sentiment analysis | Embedding + BiLSTM + Attention + 5 heads | 10K IMDB reviews labeled by Claude Haiku | Scores for acting, plot, visuals, music, direction |

This separation provides two key advantages: (1) each model is optimized for its specific task, and (2) the system can gracefully degrade if either model is unavailable, using independent fallback mechanisms.

The complete ML inference pipeline operates in the following sequence:

```
Raw Review Text
    |
    v
[Preprocessing] -- HTML/URL removal, whitespace normalization, truncation to 400 words
    |
    +----> [BERT Sentiment]  -- batch(16), softmax, neutral thresholding at 0.6
    |          |
    |          v
    |      {label: "positive"|"negative"|"neutral", score: 0.0-1.0}
    |
    +----> [BiLSTM Aspects]  -- tokenize, pad(200), attention-weighted, 5-head classification
    |          |
    |          v
    |      {acting: 0.0-1.0, plot: 0.0-1.0, visuals: 0.0-1.0, music: 0.0-1.0, direction: 0.0-1.0}
    |
    v
[Aggregation] -- verdict, confidence, source comparison, trend, summaries, word clouds
    |
    v
Final Analysis Result (persisted to database)
```

### 1.2 Model 1: BERT Sentiment Classifier

#### 1.2.1 Architecture

The sentiment classifier is built on **BERT-base-uncased** (Bidirectional Encoder Representations from Transformers), a 12-layer transformer encoder with 110 million parameters. The model uses the standard sequence classification architecture:

```
Input Text
    |
    v
[WordPiece Tokenizer] -- vocab_size=30522, max_length=512
    |
    v
[BERT Encoder]
    12 Transformer Layers
    Each layer:
        Multi-Head Self-Attention (12 heads, d_k=64)
        Feed-Forward Network (d_ff=3072)
        Layer Normalization + Residual Connections
    Hidden dimension: 768
    |
    v
[CLS] token representation (768-dim)
    |
    v
[Classification Head] -- Linear(768, 2) + Softmax
    |
    v
P(negative), P(positive)
```

#### 1.2.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `bert-base-uncased` |
| Dataset | IMDB 50K (25K train, 25K test) |
| Number of labels | 2 (binary: positive/negative) |
| Training epochs | 3 |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Learning rate | 2e-5 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max sequence length | 512 tokens |
| Precision | FP16 (mixed precision) |
| Best model selection | Weighted F1 score |
| Optimizer | AdamW (default Hugging Face Trainer) |
| Evaluation strategy | Per-epoch |
| Save strategy | Per-epoch, load best at end |

Training is performed on NVIDIA T4 GPUs via **Modal**, a serverless GPU platform. The Modal deployment script (`modal_train_bert.py`) configures a persistent volume (`cineinsight-models`) for storing trained weights across runs, enabling reproducible training.

The IMDB dataset provides binary sentiment labels (positive/negative). Rather than collecting a separate 3-class dataset, CineInsight synthesizes the **neutral class at inference time** through confidence thresholding --- a pragmatic approach detailed in Section 1.2.3.

#### 1.2.3 Neutral Class Synthesis via Thresholding

The trained model outputs 2-class probabilities P(negative) and P(positive). At inference time, a neutral class is dynamically created:

```
If P(positive) > 0.6  -->  label = "positive",  score = P(positive)
If P(negative) > 0.6  -->  label = "negative",  score = P(negative)
Otherwise             -->  label = "neutral",   score = max(P(positive), P(negative))
```

The 0.6 threshold creates a "uncertainty band" where the model is not confident enough to commit to either polarity. This is a deliberate design choice:

- **Why not train a 3-class model?** High-quality 3-class sentiment datasets for movie reviews are scarce. The IMDB dataset is the gold standard for binary movie sentiment, with 50K expert-curated examples. Introducing a third class would require either noisy auto-labeling or a much smaller dataset, both reducing model quality.
- **Why 0.6?** This threshold was empirically chosen to balance between capturing genuine ambiguity and avoiding over-classification as neutral. At 0.5 (random chance for binary), too many reviews would be neutral. At 0.7, genuine mixed reviews would be forced into positive or negative.

#### 1.2.4 Inference Details

Inference processes reviews in batches of 16 with the following pipeline:

1. **Tokenization**: `AutoTokenizer` with padding, truncation, max_length=512
2. **Forward pass**: No gradient computation (`torch.no_grad()`)
3. **Softmax**: Converts logits to probabilities via `F.softmax(outputs.logits, dim=-1)`
4. **Thresholding**: Applies the 0.6 neutral synthesis rule
5. **Output**: `{"label": str, "score": float}` per review

#### 1.2.5 Fallback Model

If the fine-tuned BERT weights are not found at the configured path, the system falls back to **DistilBERT-SST2** (`distilbert-base-uncased-finetuned-sst-2-english`), a pre-trained sentiment model from Hugging Face Hub. This model is smaller (66M params) and less accurate for movie reviews but provides reasonable baseline predictions. If even this fails (no internet, no `transformers` library), random predictions are generated as a last resort.

### 1.3 Model 2: BiLSTM Aspect-Based Sentiment Analyzer

#### 1.3.1 Architecture

The aspect model uses a **Bidirectional LSTM with Additive Attention** and **multi-head classification**. A single shared encoder extracts contextualized features, which are then fed to 5 independent classification heads --- one per movie aspect (acting, plot, visuals, music, direction).

```
Input Text (whitespace-tokenized)
    |
    v
[Word-to-Index Mapping] -- vocab lookup, <UNK>=1, <PAD>=0, max_len=200
    |
    v
[Embedding Layer] -- (vocab_size, 300), initialized with GloVe 6B 300d, fine-tuned
    |
    v
[Bidirectional LSTM]
    2 layers, hidden_dim=256 per direction (512 bidirectional)
    dropout=0.3 between layers
    Output: (batch, seq_len, 512)
    |
    v
[Additive Attention]
    Linear(512, 1) --> softmax over seq_len --> weighted sum
    Output: (batch, 512) -- single context vector per review
    |
    v
[5 Independent Aspect Heads]
    Each head:
        Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 3)
    |
    v
5 sets of 3-class logits: P(negative), P(neutral), P(positive) per aspect
    |
    v
[Weighted Score Conversion]
    score = P(neg)*0.0 + P(neutral)*0.5 + P(pos)*1.0
    Output: continuous score in [0.0, 1.0] per aspect
```

#### 1.3.2 Training Data Generation via LLM-Based Weak Supervision

The most innovative aspect of this model is its training data pipeline. Since no large-scale aspect-annotated movie review dataset exists, CineInsight uses **Claude Haiku (claude-haiku-4-5-20251001)** to generate silver-standard aspect labels through a knowledge distillation approach:

1. **Sampling**: 10,000 reviews are randomly sampled (seed=42) from the IMDB training split
2. **Batch labeling**: Reviews are sent to Claude Haiku in batches of 5, each truncated to 800 characters
3. **Prompt structure**: The LLM is asked to rate each review on 5 aspects using a 3-point scale (0=negative, 1=neutral, 2=positive)
4. **Response parsing**: JSON responses are parsed with markdown-fence stripping and value clamping to [0,2]
5. **Fallback**: If the API call fails for a batch, the overall IMDB binary label is mapped to a coarse approximation: positive reviews get `[2, 2, 1, 1, 1]`, negative get `[0, 0, 1, 1, 1]`
6. **Caching**: Labeled data is cached to `labeled_data.json` so re-runs skip the expensive labeling step

This approach is a form of **weak supervision** or **knowledge distillation** --- using a large, expensive model (Claude Haiku) to generate training labels for a small, fast model (BiLSTM) that can run locally without API costs.

#### 1.3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | 10K IMDB reviews labeled by Claude Haiku |
| Train/Val split | 90/10 |
| Embedding | GloVe 6B 300d (fine-tuned) |
| Hidden dimension | 256 per direction (512 bidirectional) |
| LSTM layers | 2 |
| Dropout | 0.3 (LSTM inter-layer + classification heads) |
| Batch size | 64 |
| Learning rate | 1e-3 (Adam) |
| Max epochs | 20 |
| Early stopping patience | 5 epochs |
| Gradient clipping | max norm = 1.0 |
| Max sequence length | 200 tokens |
| Vocab min frequency | 3 |
| Loss function | CrossEntropyLoss (summed over 5 aspects) |
| GPU | NVIDIA T4 via Modal |

#### 1.3.4 Attention Mechanism

The additive attention layer computes a single importance weight per token position:

```
attn_weights = softmax(Linear(lstm_output))     -- shape: (batch, seq_len, 1)
context = sum(lstm_output * attn_weights, dim=1)  -- shape: (batch, 512)
```

This attention mechanism allows the model to focus on aspect-relevant tokens. For example, when predicting "acting" quality, attention weights should peak on tokens like "performance", "cast", "portrayed", while for "visuals", peaks should occur near "cinematography", "stunning", "effects".

#### 1.3.5 Multi-Head Classification

The 5 independent classification heads share the same encoder but have separate parameters. This means:
- Acting-related features are routed through the acting head
- Plot-related features are routed through the plot head
- etc.

Each head outputs 3-class logits which are converted to a continuous [0, 1] score:

```
score = P(negative) * 0.0 + P(neutral) * 0.5 + P(positive) * 1.0
```

This weighted combination preserves the semantic meaning: 0.0 = fully negative, 0.5 = neutral, 1.0 = fully positive.

#### 1.3.6 Keyword-Based Fallback

If the BiLSTM model weights are unavailable, the system falls back to a **keyword-matching heuristic**:

| Aspect | Keywords (stems) |
|--------|-----------------|
| Acting | act, perform, character, cast, portray, role, actor, actress |
| Plot | story, plot, script, narrative, writing, twist, storyline, pacing |
| Visuals | visual, cinemat, scene, shot, effect, cgi, look, beautiful, stun |
| Music | music, score, sound, soundtrack, song, audio, compose |
| Direction | direct, vision, helm, filmmaker, auteur, craft |

The scoring formula is: `score = min(0.5 + keyword_matches * 0.1, 1.0)`, where keyword matching uses regex word-boundary detection (e.g., `\bcinemato` matches "cinematography"). A base score of 0.5 represents neutral, and each keyword match adds 0.1, capping at 1.0 after 5 matches.

### 1.4 Text Preprocessing Pipeline

The preprocessor (`preprocessor.py`) applies a lightweight, model-agnostic cleaning pipeline:

| Step | Operation | Regex/Method |
|------|-----------|-------------|
| 1 | Remove HTML tags | `<[^>]+>` -> space |
| 2 | Remove URLs | `https?://\S+` -> space |
| 3 | Normalize whitespace | `\s+` -> single space, strip |
| 4 | Truncate long reviews | Split by whitespace, keep first 400 words |

Design decisions:
- **No lowercasing**: Left to each model's tokenizer. BERT's WordPiece handles casing internally; the BiLSTM's dataset class applies `.lower()`.
- **No stopword removal**: BERT benefits from seeing the full grammatical context; removing stopwords can hurt transformer performance.
- **No punctuation stripping**: Punctuation carries sentiment signal (e.g., "!!!" vs ".", "?" for sarcasm).
- **400-word truncation (not 512)**: Provides headroom for BERT's WordPiece tokenizer, which can expand words into multiple subword tokens. A 400-word review might tokenize to ~500 tokens, leaving room within the 512-token limit.

### 1.5 Post-Prediction Aggregation Engine

After both models produce per-review predictions, the aggregation engine (`aggregator.py`) computes six derived metrics:

#### 1.5.1 Verdict and Confidence

The verdict system maps the ratio of positive reviews to a human-readable recommendation:

| Positive Ratio | Verdict |
|---------------|---------|
| > 80% | Strongly Recommended |
| > 60% | Worth Watching |
| >= 40% | Mixed Reviews |
| >= 20% | Below Average |
| < 20% | Skip It |

Confidence is computed as the **mean model confidence** across all predictions:
`confidence = mean(score for each prediction)`

#### 1.5.2 Overall Sentiment

Determined by **majority vote** --- the most frequent label (positive/negative/neutral) across all reviews using `Counter.most_common(1)`.

#### 1.5.3 Aspect Score Aggregation

Per-review aspect scores from the BiLSTM are averaged across all reviews:
`aspect_avg[a] = mean(review_scores[a] for all reviews)` for each aspect `a` in {acting, plot, visuals, music, direction}.

#### 1.5.4 Source Comparison

Groups sentiment labels by review source (TMDB, IMDB, Rotten Tomatoes) and counts positive/negative/neutral per source, enabling cross-platform sentiment comparison.

#### 1.5.5 Sentiment Trend

Groups reviews by month (using `review_date`) and counts sentiments per month, creating a time-series for trend visualization.

#### 1.5.6 Extractive Summarization

Extracts the top-3 most representative positive and negative sentences using a scoring formula:
```
score = model_confidence * min(word_count / 20, 1.0)
```

This favors sentences that are both high-confidence and sufficiently long (at least 20 words). Sentences shorter than 30 characters are excluded.

#### 1.5.7 Word Cloud Generation

Generates word frequency data separately for positive and negative reviews:
1. Tokenizes using regex: `\b[a-z]{3,}\b` (words of 3+ lowercase letters)
2. Removes NLTK English stopwords + movie-generic stopwords ("movie", "film", "one", "like", "also", "would", "really", "good", "bad", "great", etc.)
3. Counts word frequencies using `Counter`
4. Returns top 50 words per polarity

### 1.6 Inference Pipeline Integration

The full inference pipeline in `analysis.py` (the `/api/movies/{tmdb_id}/analyze` endpoint) orchestrates all ML components in a 6-stage process:

| Stage | Progress | Operations |
|-------|----------|------------|
| 1. Metadata | 2-5% | Fetch movie details from TMDB API |
| 2. Scraping | 8-42% | Parallel scraping from TMDB + IMDB + RT |
| 3. Storing | 45-50% | Deduplicate (SHA-256 hash) and persist reviews |
| 4. BERT Sentiment | 52-68% | `batch_clean()` -> `predict_sentiment()` |
| 5. BiLSTM Aspects | 70-80% | `predict_aspects()` -> `aggregate_aspect_scores()` |
| 6. Aggregation | 82-100% | Verdict, source comparison, trend, summaries, word clouds |

Each stage sends real-time progress updates via WebSocket.

### 1.7 Fallback and Graceful Degradation Strategy

The system implements a three-tier fallback for each model:

**BERT Sentiment:**
1. **Primary**: Fine-tuned `bert-base-uncased` from local weights
2. **Secondary**: Pre-trained `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face Hub (auto-downloaded)
3. **Tertiary**: Random predictions with scores in [0.5, 0.95]

**BiLSTM Aspects:**
1. **Primary**: Trained BiLSTM from local weights (`bilstm_weights.pt` + `vocab.pt`)
2. **Secondary**: Keyword-based heuristic scoring (no ML required)

**Scraping:**
1. **TMDB**: API with 3-retry exponential backoff
2. **IMDB**: Dual-layout scraper (2024+ and legacy HTML selectors)
3. **Rotten Tomatoes**: Stub (returns empty); JS-rendered content not scrapable without headless browser

This design ensures the system never completely fails --- it degrades gracefully from ML predictions to heuristics to random baselines.

### 1.8 Literature Review and Comparative Analysis

#### 1.8.1 Existing Approaches to Movie Sentiment Analysis

The field of movie review sentiment analysis has been extensively studied. Key approaches include:

**Traditional ML Methods:**
- Pang and Lee (2004) established the baseline with Naive Bayes, SVM, and Maximum Entropy classifiers on the IMDB dataset, achieving ~87% accuracy with unigram features.
- Maas et al. (2011) introduced the Large Movie Review Dataset (IMDB 50K) and proposed a word vector model achieving ~88.9% accuracy.

**Deep Learning Methods:**
- Kim (2014) demonstrated that simple CNN architectures with pre-trained Word2Vec embeddings achieve competitive results (~87.2% on IMDB).
- Tai et al. (2015) showed Tree-LSTMs outperform standard LSTMs for sentiment composition.
- Howard and Ruger (2018) introduced ULMFiT, demonstrating that transfer learning from language models significantly improves text classification.

**Transformer-Based Methods:**
- Devlin et al. (2019) introduced BERT, achieving state-of-the-art results on sentiment analysis benchmarks. Fine-tuned BERT on IMDB achieves ~93-95% accuracy.
- Sun et al. (2019) explored various fine-tuning strategies for BERT on sentiment tasks.

**Aspect-Based Sentiment Analysis (ABSA):**
- Pontiki et al. (2014-2016) established SemEval ABSA shared tasks with restaurant and laptop domains.
- Tang et al. (2016) proposed attention-based LSTMs for target-dependent sentiment classification.
- Xu et al. (2019) applied BERT to ABSA tasks using post-training on review corpora.

**Existing Commercial/Open-Source Tools:**
- **Rotten Tomatoes Tomatometer**: Simple percentage of positive critic reviews. No aspect analysis, no public sentiment on specific dimensions.
- **IMDb Ratings**: Numeric scores (1-10) without sentiment decomposition or aspect granularity.
- **Metacritic**: Weighted average of critic scores. No aspect-level analysis.
- **Letterboxd**: Community ratings without automated NLP analysis.
- **MonkeyLearn / AWS Comprehend**: Generic sentiment APIs not specialized for movie reviews. No aspect heads, no movie-specific vocabulary.

#### 1.8.2 How CineInsight Improves Over Existing Alternatives

CineInsight addresses several gaps in existing solutions:

| Limitation of Existing Tools | CineInsight's Approach |
|---|---|
| **Aggregate scores only** (RT Tomatometer, IMDb): reduce sentiment to a single number | **Multi-dimensional analysis**: 5 aspect scores + overall sentiment + confidence + verdict, providing granular understanding |
| **No aspect decomposition**: existing tools don't tell you WHY a movie is good/bad | **5-axis aspect radar**: acting, plot, visuals, music, direction --- reveals specific strengths and weaknesses |
| **Critic vs. audience disconnect**: RT separates critics and audience but doesn't analyze the text | **Cross-source analysis**: scrapes and analyzes reviews from multiple platforms (TMDB, IMDB), comparing sentiment distributions across sources |
| **No temporal analysis**: existing tools show a static aggregate | **Sentiment trend charts**: tracks how public opinion evolves over time (monthly) |
| **Opaque methodology**: Metacritic's weighting formula is proprietary; RT's "fresh/rotten" is binary | **Transparent ML pipeline**: every prediction comes with confidence scores, and the full analysis process is visible in real-time via WebSocket |
| **No real-time feedback**: users submit a query and wait | **Live analysis dashboard**: WebSocket-powered progress updates show scraping, BERT inference, BiLSTM inference, and aggregation as they happen |
| **Generic sentiment models** (AWS Comprehend, MonkeyLearn): trained on product reviews, tweets, etc. | **Domain-specific fine-tuning**: BERT fine-tuned specifically on the IMDB movie review corpus; BiLSTM trained on movie reviews labeled with movie-specific aspects |
| **No extractive evidence**: existing tools give a score but don't show which sentences justify it | **Extractive summarization + word clouds**: identifies the most representative positive and negative sentences, plus word frequency visualization |

#### 1.8.3 Novel Contributions

1. **LLM-Driven Weak Supervision for ABSA**: Using Claude Haiku to generate aspect-level silver labels for 10K movie reviews is a novel data generation strategy. This is a form of programmatic labeling (Ratner et al., 2017) combined with knowledge distillation (Hinton et al., 2015), where a large language model's understanding of movie aspects is distilled into a lightweight BiLSTM.

2. **Dual-Model Complementary Architecture**: Separating sentiment classification (BERT, 110M params) from aspect analysis (BiLSTM, ~2M params) allows each model to specialize. The BiLSTM is significantly more efficient at inference time while the BERT model provides the accuracy needed for binary sentiment decisions.

3. **Neutral Class Synthesis**: Rather than training on noisy 3-class data, synthesizing the neutral class via confidence thresholding on a strong binary classifier preserves the high quality of IMDB's binary labels while supporting the nuanced 3-class output the application requires.

4. **Real-Time Analysis Pipeline**: The WebSocket-based progress system is uncommon in NLP applications. Users see the analysis unfold in real-time --- from scraping to BERT to BiLSTM to final verdict --- providing transparency and engagement that static APIs cannot match.

---

## 2. Full-Stack Application

### 2.1 System Architecture

CineInsight follows a **decoupled client-server architecture** with real-time communication:

```
+---------------------------+        +--------------------------------+
|       Frontend            |        |          Backend               |
|   React 19 + TypeScript   |        |     FastAPI + Python 3.10      |
|   Vite 8 + TailwindCSS    |        |                                |
|   HeroUI + Recharts       |  HTTP  |   +--------+   +-----------+  |
|   Framer Motion           | -----> |   |  API   |   |  Scrapers |  |
|                           |        |   | Routes |   |  (TMDB,   |  |
|   +-------------------+   |  WS    |   +---+----+   |  IMDB,RT) |  |
|   | React Query Cache |   | <----> |       |        +-----------+  |
|   +-------------------+   |        |   +---v----+                  |
|                           |        |   |  ML    |   +-----------+  |
|   +-------------------+   |        |   | Engine |   | SQLite DB |  |
|   | WebSocket Client  |   |        |   | (BERT, |   | (async)   |  |
|   +-------------------+   |        |   | BiLSTM)|   +-----------+  |
+---------------------------+        +--------------------------------+
         Port 18501                          Port 18500
```

### 2.2 Backend: FastAPI + SQLAlchemy

#### 2.2.1 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/movies/search?query=` | Search movies via TMDB |
| `GET` | `/api/movies/recent` | Recently analyzed movies |
| `GET` | `/api/movies/trending` | Trending movies this week (TMDB pass-through) |
| `GET` | `/api/movies/{tmdb_id}` | Single movie details |
| `GET` | `/api/movies/{tmdb_id}/analysis` | Get cached analysis (404 if not analyzed) |
| `POST` | `/api/movies/{tmdb_id}/analyze` | Trigger full scraping + ML analysis |
| `WS` | `/ws/analysis/{tmdb_id}` | Real-time progress stream |

#### 2.2.2 Database Schema (SQLite via aiosqlite)

Four tables with the following relationships:

```
movies (1) ----< reviews (1) ----< review_sentiments >---- (1) analysis_results >---- (1) movies
```

- **movies**: Stores TMDB metadata (tmdb_id, title, year, poster, genre, overview, vote_average)
- **reviews**: Scraped review content with SHA-256 content hash for deduplication (unique constraint)
- **analysis_results**: One per movie (unique movie_id), stores all aggregated ML results as JSON columns
- **review_sentiments**: Per-review ML predictions linking reviews to their analysis, with sentiment labels, scores, and aspect scores

#### 2.2.3 Scraping Strategy

Reviews are scraped from three sources in parallel using `asyncio.gather()`:

| Source | Method | Strategy |
|--------|--------|----------|
| **TMDB** | REST API | Paginated (up to 5 pages), retry with exponential backoff, 15s timeout |
| **IMDB** | Web scraping | `httpx` + `BeautifulSoup/lxml`; search-by-title to discover IMDB ID, then scrape reviews page with dual-layout support (2024+ and legacy HTML selectors) |
| **Rotten Tomatoes** | Stub | Returns empty (reviews are JS-rendered, not accessible without headless browser) |

All scrapers inherit from `BaseScraper` which enforces a 1.5-second delay between requests (rate limiting) and caps reviews at 50 per source. The IMDB scraper uses a realistic Chrome User-Agent header.

### 2.3 Frontend: React + TypeScript

#### 2.3.1 Technology Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 19 with TypeScript 5.9 |
| Build Tool | Vite 8 |
| Styling | TailwindCSS 3.4 with custom Netflix-inspired theme |
| Component Library | HeroUI (NextUI fork) |
| State Management | TanStack React Query v5 (server state), React useState (UI state) |
| Routing | React Router v7 |
| Charts | Recharts v3 (radar, bar, line charts) |
| Animation | Framer Motion v12 |
| HTTP Client | Axios |
| Icons | Lucide React |

#### 2.3.2 Page Architecture

| Route | Page | Purpose |
|-------|------|---------|
| `/` | Home | Hero banner with search, trending + recently analyzed movie carousels |
| `/analysis/:tmdbId` | Analysis | Movie details, trigger analysis, live progress, full analysis dashboard |
| `/compare` | Compare | Side-by-side comparison of two analyzed movies |
| `/about` | About | Static page describing model architecture and tech stack |

#### 2.3.3 State Management

CineInsight uses **React Query as the sole server-state layer** --- there is no Redux, Zustand, or similar global store. All server data flows through query caches:

| Query Key | Stale Time | Purpose |
|-----------|-----------|---------|
| `["recentMovies"]` | 5 min | Home/Compare page movie lists |
| `["trendingMovies"]` | 10 min | Home page trending carousel |
| `["movieSearch", query]` | 5 min | Debounced search results |
| `["movieDetails", id]` | 30 min | Individual movie details |
| `["analysis", id]` | 30 min | Cached analysis results |

The analysis mutation uses `queryClient.setQueryData` on success to directly populate the cache, avoiding a redundant refetch.

#### 2.3.4 Design System

The frontend implements a **Netflix-inspired dark theme**:

- **Color palette**: Deep blacks (#141414, #1a1a1a, #222222), Netflix red (#E50914), semantic sentiment colors (green/yellow/red)
- **Glass-morphism**: `bg-white/5 backdrop-blur-sm border border-white/5` pattern across cards
- **Typography**: Inter font family via Google Fonts
- **Responsive**: Mobile-first with `sm:`, `md:`, `lg:` breakpoints; hamburger nav on mobile, horizontal on desktop

### 2.4 Real-Time Communication: WebSocket Pipeline

The analysis page features a **live terminal-style dashboard** powered by WebSockets:

1. **Connection**: When the user triggers analysis, the frontend opens a WebSocket to `/ws/analysis/{tmdb_id}`
2. **Progress events**: The backend sends structured JSON messages at each stage of the pipeline
3. **Typed messages**: Six message types with discriminated `type` field:
   - `"log"` -- Terminal-style text updates with progress percentage
   - `"source"` -- Scraping status per source (count, done/error/skipped)
   - `"reviews"` -- Preview of first 8 scraped reviews
   - `"sentiments"` -- BERT predictions with confidence scores
   - `"aspects"` -- BiLSTM aspect scores per review
   - `"verdict"` -- Final verdict, confidence, and overall sentiment
4. **UI rendering**: `LiveAnalysis` component renders source cards, review previews, sentiment predictions, aspect bars, and a spring-animated verdict reveal
5. **5-second hold**: After analysis completes, the live dashboard remains visible for 5 seconds before transitioning to the full analysis view

### 2.5 Data Pipeline: Scraping to Visualization

The complete data flow from user action to visualization:

```
User clicks "Analyze Movie"
    |
    v
[Frontend] POST /api/movies/{id}/analyze + opens WebSocket
    |
    v
[Backend] Fetch movie metadata from TMDB API
    |
    v
[Backend] Scrape reviews in parallel (TMDB API + IMDB + RT)
    |   -- sends WebSocket progress at each step --
    v
[Backend] Deduplicate via SHA-256 hash, store in SQLite
    |
    v
[Backend] Preprocess text (HTML/URL removal, truncation)
    |
    v
[Backend] BERT inference (batch=16) --> sentiment labels + scores
    |
    v
[Backend] BiLSTM inference (per-review) --> 5 aspect scores each
    |
    v
[Backend] Aggregate: verdict, confidence, source comparison,
          sentiment trend, extractive summaries, word clouds
    |
    v
[Backend] Persist AnalysisResult + ReviewSentiments to SQLite
    |
    v
[Backend] Return AnalysisResponse JSON (200 OK)
    |
    v
[Frontend] React Query cache populated via setQueryData
    |
    v
[Frontend] Render full dashboard: VerdictBadge, SentimentGauge,
           AspectRadar, SourceComparison, SentimentTrend,
           ReviewSummary, WordCloud, ReviewList (paginated)
```

---

## 3. Deployment

### 3.1 Infrastructure Overview

| Component | Specification |
|-----------|--------------|
| **Cloud Provider** | Oracle Cloud Infrastructure (OCI) |
| **VM Shape** | Ampere A1 (ARM64/aarch64) |
| **OS** | Ubuntu 20.04.6 LTS (Focal Fossa) |
| **RAM** | 24 GB |
| **Storage** | 97 GB (48 GB available) |
| **IP Address** | 141.148.219.124 |
| **Containerization** | Docker 24.0.7 + docker-compose 1.25.0 |
| **SSH Access** | Key-based via `~/Downloads/key.pem` as `ubuntu` user |

The VM is an ARM64 instance, which impacts Docker image selection --- all base images must support `linux/arm64`. The Python 3.10-slim and Node 20-slim images both have ARM64 variants, and PyTorch 2.5.1 supports ARM64 Linux natively.

### 3.2 Dockerization Strategy

The application is split into two Docker containers: a **Python backend** and an **nginx frontend**.

#### 3.2.1 Backend Dockerfile (`backend/Dockerfile`)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System dependencies for lxml (HTML parsing) and wordcloud (C extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data for word cloud generation
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); \
    nltk.download('stopwords'); nltk.download('wordnet')"

COPY app/ app/
COPY ml/models/ ml/models/

EXPOSE 18500

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18500"]
```

Key decisions:
- **python:3.10-slim**: Balances image size (~150MB) with Python 3.10 requirement (needed for PyTorch 2.5.1 which dropped Python 3.8 support)
- **System deps**: `gcc` + `libc6-dev` for compiling `wordcloud` C extension; `libxml2-dev` + `libxslt1-dev` for `lxml` parser used in BeautifulSoup
- **NLTK data baked in**: Downloaded at build time to avoid runtime downloads
- **ML models copied**: The `ml/models/` directory (BERT ~418MB + BiLSTM ~70MB) is copied into the image. Training checkpoints are excluded via `.dockerignore`

#### 3.2.2 Frontend Dockerfile (`frontend/Dockerfile`)

```dockerfile
FROM node:20-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 18501

CMD ["nginx", "-g", "daemon off;"]
```

Key decisions:
- **Multi-stage build**: Stage 1 (node:20-slim) installs deps and builds the Vite app; Stage 2 (nginx:alpine) serves the static output. The final image is ~25MB instead of ~350MB+.
- **Build-time VITE_API_URL**: Passed as a Docker build argument and injected into the Vite build via `import.meta.env.VITE_API_URL`. This is baked into the JavaScript bundle at build time, pointing to `http://141.148.219.124:18500`.
- **nginx SPA routing**: Custom `nginx.conf` serves `index.html` for all routes (`try_files $uri $uri/ /index.html`), enabling client-side routing.

#### 3.2.3 nginx Configuration (`frontend/nginx.conf`)

```nginx
server {
    listen 18501;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    gzip on;
    gzip_types text/plain text/css application/json application/javascript
               text/xml application/xml text/javascript image/svg+xml;
    gzip_min_length 256;
}
```

Gzip compression is enabled for all text-based assets (JS, CSS, JSON, SVG), reducing transfer sizes by 60-80%.

#### 3.2.4 `.dockerignore` Files

**Backend** excludes: `.venv/` (7.5GB), `__pycache__/`, `.env`, `cineinsight.db`, `ml/training/`, `ml/notebooks/`, `ml/models/bert_sentiment/checkpoint-*/` (3.9GB of training checkpoints), `tests/`.

**Frontend** excludes: `node_modules/` (350MB), `dist/`, `.gitignore`, `README.md`.

These exclusions reduce the Docker build context from ~12GB to ~500MB.

### 3.3 Docker Compose Orchestration

```yaml
version: "3.3"

services:
  backend:
    build:
      context: ./backend
    ports:
      - "18500:18500"
    environment:
      - TMDB_API_KEY=${TMDB_API_KEY}
      - CORS_ORIGINS=["http://141.148.219.124:18501"]
      - DATABASE_URL=sqlite+aiosqlite:///./cineinsight.db
      - BERT_MODEL_PATH=ml/models/bert_sentiment
      - BILSTM_MODEL_PATH=ml/models/bilstm_aspect
    volumes:
      - backend-data:/app/data
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      args:
        VITE_API_URL: http://141.148.219.124:18500
    ports:
      - "18501:18501"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  backend-data:
```

Key configuration:
- **Version 3.3**: Required for compatibility with docker-compose 1.25.0 on the VM (3.8 is not supported)
- **CORS_ORIGINS**: Set to the frontend's public URL so the browser can make cross-origin API requests
- **VITE_API_URL**: Passed as a build arg so the frontend JavaScript knows the backend's public address
- **TMDB_API_KEY**: Loaded from `.env.production` file (not baked into the image)
- **restart: unless-stopped**: Containers automatically restart on crash or host reboot (unless explicitly stopped)
- **depends_on**: Frontend starts after backend (though the frontend is a static server and doesn't directly need the backend at startup)

### 3.4 Step-by-Step Deployment Guide

#### Prerequisites
- SSH access to the target VM: `ssh -i key.pem ubuntu@<IP>`
- Docker and docker-compose installed on the VM
- The project repository cloned locally

#### Step 1: Prepare Docker Files

Create the following files in the project root:
- `backend/Dockerfile` (see Section 3.2.1)
- `frontend/Dockerfile` (see Section 3.2.2)
- `frontend/nginx.conf` (see Section 3.2.3)
- `docker-compose.yml` (see Section 3.3)
- `backend/.dockerignore` and `frontend/.dockerignore` (see Section 3.2.4)
- `.env.production` containing `TMDB_API_KEY=<your_key>`

#### Step 2: Transfer Files to VM

```bash
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='backend/.venv/' \
  --exclude='backend/__pycache__/' \
  --exclude='backend/cineinsight.db' \
  --exclude='backend/ml/training/' \
  --exclude='backend/ml/notebooks/' \
  --exclude='backend/ml/models/bert_sentiment/checkpoint-*/' \
  --exclude='frontend/node_modules/' \
  --exclude='frontend/dist/' \
  -e "ssh -i ~/Downloads/key.pem" \
  ./ ubuntu@141.148.219.124:~/cineinsight/
```

This transfers approximately 500MB (primarily the BERT model weights at 418MB).

#### Step 3: Build Containers

```bash
ssh -i key.pem ubuntu@141.148.219.124

cd ~/cineinsight

# Build frontend first (faster, ~2 minutes)
docker-compose --env-file .env.production build frontend

# Build backend (slower due to PyTorch, ~10-15 minutes)
docker-compose --env-file .env.production build backend
```

#### Step 4: Start the Application

```bash
docker-compose --env-file .env.production up -d
```

Verify containers are running:

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}\t{{.Status}}'
```

Expected output:
```
NAMES                    PORTS                                                   STATUS
cineinsight_frontend_1   80/tcp, 0.0.0.0:18501->18501/tcp, :::18501->18501/tcp   Up 5 seconds
cineinsight_backend_1    0.0.0.0:18500->18500/tcp, :::18500->18500/tcp           Up 5 seconds
```

#### Step 5: Verify Deployment

```bash
# Backend health check
curl http://localhost:18500/health
# Expected: {"status":"ok","service":"cineinsight"}

# Frontend
curl -s -o /dev/null -w '%{http_code}' http://localhost:18501/
# Expected: 200

# Backend API
curl http://localhost:18500/api/movies/trending | head -c 100
# Expected: JSON array of trending movies
```

Check backend logs for ML model loading:
```bash
docker-compose --env-file .env.production logs backend
```

Expected:
```
INFO:app.services.ml.sentiment:Loading fine-tuned BERT from ml/models/bert_sentiment
INFO:app.services.ml.sentiment:BERT model loaded successfully
INFO:app.services.ml.aspects:Loading BiLSTM model from ml/models/bilstm_aspect
INFO:app.services.ml.aspects:BiLSTM model loaded successfully
INFO:app.main:CineInsight backend ready!
```

### 3.5 Firewall and Network Configuration

#### 3.5.1 UFW (Host-Level Firewall)

```bash
sudo ufw allow 18500/tcp   # Backend API
sudo ufw allow 18501/tcp   # Frontend
```

Verify:
```bash
sudo ufw status | grep 185
```

Expected:
```
18500/tcp    ALLOW    Anywhere
18501/tcp    ALLOW    Anywhere
18500/tcp (v6)    ALLOW    Anywhere (v6)
18501/tcp (v6)    ALLOW    Anywhere (v6)
```

#### 3.5.2 Oracle Cloud Security List / NSG

In addition to UFW, Oracle Cloud requires **ingress rules** in the VCN's Security List or Network Security Group:

| Direction | Source CIDR | Protocol | Destination Port | Description |
|-----------|------------|----------|-----------------|-------------|
| Ingress | 0.0.0.0/0 | TCP | 18500 | CineInsight Backend API |
| Ingress | 0.0.0.0/0 | TCP | 18501 | CineInsight Frontend |

Navigate to: **OCI Console > Networking > Virtual Cloud Networks > [Your VCN] > Security Lists > Default Security List > Add Ingress Rules**

#### 3.5.3 Final Access URLs

| Service | URL |
|---------|-----|
| Frontend | `http://141.148.219.124:18501` |
| Backend API | `http://141.148.219.124:18500` |
| Health Check | `http://141.148.219.124:18500/health` |
| API Docs (Swagger) | `http://141.148.219.124:18500/docs` |

### 3.6 Environment Variables and Secrets

| Variable | Location | Purpose |
|----------|----------|---------|
| `TMDB_API_KEY` | `.env.production` | TMDB API authentication |
| `CORS_ORIGINS` | `docker-compose.yml` | Allowed frontend origins for CORS |
| `DATABASE_URL` | `docker-compose.yml` | SQLite database path |
| `BERT_MODEL_PATH` | `docker-compose.yml` | Path to BERT model weights inside container |
| `BILSTM_MODEL_PATH` | `docker-compose.yml` | Path to BiLSTM model weights inside container |
| `VITE_API_URL` | Build arg in `docker-compose.yml` | Backend URL baked into frontend JS |

The `.env.production` file should **never** be committed to version control. It is listed in `.gitignore`.

### 3.7 Operational Management

#### Common Commands

```bash
cd ~/cineinsight

# View real-time logs
docker-compose --env-file .env.production logs -f

# View backend logs only
docker-compose --env-file .env.production logs -f backend

# Restart all services
docker-compose --env-file .env.production restart

# Restart backend only
docker-compose --env-file .env.production restart backend

# Stop all services
docker-compose --env-file .env.production down

# Start all services
docker-compose --env-file .env.production up -d

# Rebuild after code changes
docker-compose --env-file .env.production build --no-cache
docker-compose --env-file .env.production up -d
```

#### Updating the Application

```bash
# On local machine: sync changes
rsync -avz --progress \
  --exclude='.git/' --exclude='backend/.venv/' \
  --exclude='frontend/node_modules/' \
  --exclude='backend/ml/models/bert_sentiment/checkpoint-*/' \
  -e "ssh -i ~/Downloads/key.pem" \
  ./ ubuntu@141.148.219.124:~/cineinsight/

# On VM: rebuild and restart
ssh -i key.pem ubuntu@141.148.219.124
cd ~/cineinsight
docker-compose --env-file .env.production build
docker-compose --env-file .env.production up -d
```

#### Monitoring

```bash
# Container resource usage
docker stats cineinsight_backend_1 cineinsight_frontend_1

# Disk usage by Docker
docker system df

# Check if containers are healthy
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

### 3.8 Architecture Decisions and Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Docker over PM2** | Avoids Python version conflicts (VM has Python 3.8, PyTorch needs 3.9+). Ensures reproducible builds. Isolates from other services on the VM. | Slightly more resource overhead than bare-metal. Longer build times. |
| **SQLite over PostgreSQL** | Zero configuration, single-file database, sufficient for single-server deployment. Async via `aiosqlite`. | No concurrent write support. Not suitable for multi-instance horizontal scaling. |
| **Separate frontend/backend containers** | Independent scaling, independent deployments, clear separation of concerns. | Two containers to manage. CORS configuration required. |
| **Build-time VITE_API_URL** | Eliminates runtime configuration for the SPA. No need for a config endpoint or runtime env injection. | Requires rebuild to change the backend URL. Cannot use the same image for different environments. |
| **Unusual ports (18500/18501)** | Avoids conflicts with common services (80, 443, 3000, 8080). Reduces automated scanning exposure. | Must be explicitly allowed in both UFW and Oracle Cloud security lists. |
| **ML models baked into Docker image** | Ensures the image is self-contained and always has the correct model version. No runtime download or volume mount complexity. | Large image size (~2GB+ for backend). Model updates require image rebuild. |
| **restart: unless-stopped** | Containers survive host reboots and recover from crashes automatically. | Must explicitly `docker-compose down` to fully stop. |
