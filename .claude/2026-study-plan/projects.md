# Portfolio Projects (P0–P5) + Reproductions

**The bar (from the research):** a **live demo URL** + Docker + tests/CI + a README framed **Problem → Approach → Result → Impact** with **real numbers**. Reviewers spend <2 min — pin your best three. Every project ships a **case study** and a **LinkedIn post**.

**Your signature move:** every project shows the **testing/evaluation** you did. That's the part others skip and clients fear most — and it's your ex-SDET moat.

**Anchor trio (pin these):** **P3 (RAG API)** + **P4 (LLM-eval pipeline)** + best of **P1/P5**.

---

## Repo hygiene (do once, W1) — this is itself a hireability signal

- `pyproject.toml` with pinned deps (add `torch`, `transformers`, `datasets`, `fastapi`, `uvicorn`, `ragas`, `langchain`/`llama-index`, `pytest`).
- Package layout: `src/<pkg>/…`, `tests/`, `notebooks/` (exploration only — production code lives in `src/`).
- `Dockerfile` + `.dockerignore`; a `Makefile`/`justfile` (`make train`, `make serve`, `make test`, `make eval`).
- **GitHub Actions**: lint (ruff) + `pytest` on push. Later add an **"LLM eval" job** (P4) — this is the wow.
- A repo README that states *what you're building and why* (production-grade, scalable ML). Each project gets its own subfolder + README.
- `.env.example` for keys; never commit secrets.

> Frame this in a post: *"I set up my ML repo like a production service, not a notebook dump — here's the skeleton and why each piece earns its place."*

---

## P0 — Anomaly / Clustering Demo  ·  *classic-ML, easy win*
- **Feeds from:** Ch 7, 8. **Effort:** ~8–12 hrs (W4–W6, around the SAA sprint).
- **Problem:** given an unlabeled real-ish dataset (e.g. network/transaction/sensor logs, or a public fraud/telemetry set), find natural segments and flag anomalies.
- **Build:** EDA → PCA/UMAP for structure → k-means + DBSCAN segmentation → GMM/IsolationForest anomaly scoring → threshold → a small **Streamlit** app where a user uploads rows and sees cluster + anomaly score.
- **Success metrics:** silhouette score for clustering; on a labeled subset, precision/recall of anomaly flags; a clear "here's what an operator would do with this."
- **Deploy:** Streamlit Community Cloud / HF Space.
- **Wedge angle:** add a tiny test suite validating the scoring function on synthetic outliers.
- **Post:** "I let an algorithm find the groups in messy, unlabeled data with no idea what it'd surface — here's what it actually found." (anomalies for [domain])

---

## P1 — Transfer-Learning Image Classifier  ·  *CV, deployed*
- **Feeds from:** Ch 12. **Effort:** ~15 hrs (W9–W10).
- **Problem:** a **fine-grained** classification task with **few images/class** (e.g. plant disease, defect detection, food-101 subset, a niche you can self-collect). Real/messy data > clean Kaggle.
- **Build:** pretrained backbone (ResNet/EfficientNet/ViT) → freeze → train head → unfreeze + differential-LR fine-tune → augmentation. **Gradio** UI: upload an image → top-3 + probabilities.
- **Success metrics:** accuracy vs. a from-scratch baseline; per-class accuracy; confusion matrix; **3–5 honest failure cases** with commentary.
- **Deploy:** HF Space (Gradio).
- **Wedge angle:** an eval script + a "model card" reporting where it fails and the data it should *not* be trusted on.
- **Post:** "With ~30 images per class, a frozen pretrained model embarrassed my from-scratch one — then I went hunting for where it still breaks."

---

## P2 — Forecasting / Predictive Maintenance  ·  *the classic-ML hedge*
- **Feeds from:** Ch 13 (+6). **Effort:** ~15–25 hrs (W11–W12).
- **Problem:** predict a business-relevant outcome from time-series — demand forecasting, equipment failure, energy load, churn timing. Pick something a manager would *pay* to know.
- **Build:** windowed features → **XGBoost baseline** → LSTM comparison → honest backtest. Wrap the winner in a **FastAPI** `/predict` endpoint.
- **Success metrics:** MAE/RMSE (or precision/recall for failure) vs. a naive baseline; a cost/impact framing ("catching failures N days earlier ≈ $X saved").
- **Deploy:** FastAPI on Railway/Render + a tiny UI or a documented API.
- **Wedge angle:** input-validation + schema tests; a data-drift check stub.
- **Post:** "I was sure the LSTM would win. The boring gradient-boosted trees won, and it wasn't close." (forecasting [X])

---

## P3 — Domain-Specific RAG API  ·  ⭐ **ANCHOR** ·  *the 2026 golden project*
- **Feeds from:** Ch 14, 15. **Effort:** ~30 hrs (W15–W17).
- **Problem:** accurate, cited Q&A over a **specific corpus** — pick a niche with real pain: legal contracts, medical guidelines, a framework's docs, or internal-wiki-style content. Niche > generic chatbot.
- **Build:**
  - Ingestion: loaders → **smart chunking** (structure-aware, not naive fixed-size) → embeddings → **vector DB** (Chroma/FAISS/Qdrant).
  - Retrieval: hybrid **dense + BM25** → **cross-encoder reranking** → context assembly with **citations**.
  - Generation: a strong API model or a local one; **grounded** answers with source spans.
  - Serve: **FastAPI + Docker**; `/query` with streaming; a minimal chat UI.
- **Success metrics (this is the differentiator):** **RAGAS** faithfulness, answer-relevancy, context-precision/recall; retrieval hit-rate@k; p50/p95 latency; cost/query. Show a **before/after** when you add reranking.
- **Deploy:** Docker → Railway/Render/Fly; demo UI on HF Space.
- **Wedge angle:** this is where P4 attaches — a full eval harness, not vibes.
- **Post:** "My RAG demo answered everything confidently. Then I checked how many answers the documents actually supported." (faithfulness 0.62 → 0.89 after reranking, for [domain])

---

## P4 — LLM Evaluation & Observability Pipeline  ·  ⭐ **ANCHOR** ·  *your signature*
- **Feeds from:** Ch 15, App B, LoRA repro. **Effort:** ~20–25 hrs (W18–W20).
- **Problem:** most LLM apps ship with zero regression protection. You build the layer that catches quality drops **before users do** — the exact thing your SDET brain is built for.
- **Build (over P3):**
  - A **golden dataset** of question/expected-behavior pairs.
  - **RAGAS + DeepEval** metric suite; **LLM-as-judge** with calibrated rubrics.
  - **Tracing/observability**: LangSmith or **Phoenix/Arize** — see every retrieval + generation.
  - **"LLM CI"**: a **GitHub Action** that runs the eval suite on every change and **fails the build** if faithfulness/relevancy drop below thresholds.
  - Drift/quality dashboard.
- **Success metrics:** the suite catches an injected regression (demonstrate it); latency of the eval run; % of hallucinations caught.
- **Deploy:** the CI gate lives in the P3 repo; a public dashboard/report page.
- **Post (signature):** "The least glamorous question in LLM apps, and the one I can't stop thinking about: how do you *know* the output is right?" *(This post is your brand thesis — the eval obsession you actually have. Make it excellent, and keep it genuine rather than salesy.)*

---

## P5 — Agentic Workflow **or** MLOps Service  ·  *round out the story*
- **Feeds from:** Ch 17, App B. **Effort:** ~25 hrs (W21). **Pick one:**
- **(a) Agentic workflow** — a **LangGraph** tool-using agent (e.g. a data-analyst that writes+runs Python, or a research/DB agent) with **self-correction, evals, and cost/latency guardrails**. Shows "systems that *act*," the fastest-growing category.
  - **Metrics:** task success rate; tool-call accuracy; cost/task; guardrail catches.
  - **Post:** "I gave a model some tools and let it loose. It confidently did the wrong thing — so I spent a week on the guardrails."
- **(b) MLOps service** — one of your models behind **Docker + CI + drift monitoring** (Evidently/Grafana) + **alerting** + a model registry.
  - **Metrics:** deploy pipeline time; drift detection lead time; uptime.
  - **Post:** "Nobody tells you a model has quietly gone stale — so I built the monitoring that notices before the client does."
- **Recommendation:** (a) for the "wow", (b) to hammer the reliability wedge. If undecided, do (a) — agents are hotter for inbound.

---

## Reproductions (packaged, with writeups)

Reproductions build reputation **because almost no one does them** — but the value is in the **packaging**: annotated code + a "paper→code" writeup + *why it matters*. Model your writeups on **The Annotated Transformer** / **labml.ai**. Do a *few deep*, not many shallow.

### R1 — makemore  ·  ~8–15 hrs (W7–W8)
Char-level language model from scratch (Karpathy). On-ramp to autograd/backprop/embeddings. Writeup: "I taught a tiny neural net to invent names one character at a time — watching gibberish become almost-real words was unreasonably fun." *(Optional if time-pressed — nanoGPT subsumes it.)*

### R2 — nanoGPT / build-nanogpt  ·  ⭐ ~15–30 hrs (W13–W15)
Reproduce a **GPT-2-class** model (~300-line model + ~300-line training loop; scale down to your hardware — full 124M wants an 8×A100 node, but it runs small). **Highest-signal reproduction.** Writeup: "I rebuilt GPT-2 from scratch just to understand attention — the idea underneath is almost embarrassingly simple."

### R3 — LoRA / QLoRA fine-tune  ·  ⭐ ~10–20 hrs (W18)
From-scratch LoRA adapter, then the **PEFT** production path; QLoRA runs on a **free Colab T4**. **Most directly hireable** — fine-tuning is a premium gig. Writeup: "When RAG isn't enough: fine-tuning an LLM cheaply with QLoRA." Doubles as a skill feeding P4.

### (Stretch) R4 — tiny DDPM diffusion  ·  ~15–30 hrs
Small denoising diffusion model on MNIST/CIFAR (+ DDIM/CFG). Strongest *generative* signal, eye-catching samples. Only if ahead.

> **Credibility upgrade:** the ML Reproducibility Challenge is now an **official NeurIPS 2026 track** (TMLR/ReScience publication). If a reproduction goes deep, consider submitting — the highest-prestige way to turn it into a citable credential.

---

## Presentation checklist (every project)

- [ ] Live demo URL (top of README, with a GIF/screenshot).
- [ ] README: **Problem → Approach → Result → Impact**, with numbers + a failure section.
- [ ] Dockerfile + `make`/`just` commands that actually run.
- [ ] `tests/` + green CI badge.
- [ ] An eval/metrics section (your wedge).
- [ ] A LinkedIn post + (for anchors) a short blog/case study.
- [ ] Pinned on GitHub profile if it's a top-3.
