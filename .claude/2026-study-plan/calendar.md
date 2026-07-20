# Week-by-Week Calendar (Jul 18 → Dec 30, 2026)

**Budget:** 12–15 hrs/week. Each week has **Core** (must-do) and **Stretch** (if ahead).
**Legend:** 📘 book · 🔬 reproduction · 🛠️ project · ☁️ cert · 📣 post/brand.
**Golden rule:** behind? drop Stretch. Never sacrifice the anchor projects (P3, P4), SAA, or Ch 15.

> Dates are 7-day blocks from Jul 18. Slide them to your real weeks; the *sequence* matters more than exact dates.

---

## BLOCK A — Setup + SAA cert + Part I finish (W1–W6 · Jul 18 – Aug 28)

> During the SAA sprint, book work = the **light, non-DL Part-I chapters (7, 8)** — easy to do in small chunks around cert study. Save your high-focus time for the demanding DL chapters after the exam.

### W1 · Jul 18–24 — Foundations & ignition
- **Core:**
  - 📘 Finish Ch 10 loose ends (close out `ch10_test2` labs 10–12: Optuna, save/load, compile).
  - 🛠️ Turn the repo production-minded: add `torch`, `transformers`, `datasets` to `pyproject.toml`; create `src/` package layout, a `Makefile`/`justfile`, and a `README` that states what you're building. (See `projects.md` §Repo hygiene.)
  - 📣 Rewrite LinkedIn profile as a landing page (headline, About, Featured — make it sound like a person, not a résumé). Ship **post #1**: a small metrics gotcha from finishing Ch 10 — "a bug that made my model look worse than it was, and what it taught me about trusting metrics." (Observation, not a plan-declaration. See `brand.md`.)
  - ☁️ Pick SAA resources; take a **diagnostic** Tutorials Dojo timed test to find weak domains.
- **Stretch:** Skim Ch 9 (Scikit-Learn MLP intro) as a 30-min bridge to the PyTorch chapters.
- **Deliverable:** repo restructured + profile live + SAA baseline score.

### W2 · Jul 25–31 — SAA: Secure Architectures (30%)
- **Core:** ☁️ SAA Domain 1 (IAM, KMS, encryption, security groups, VPC security) — the biggest domain. 📘 Ch 7 (Dimensionality Reduction) light: PCA intuition + one hands-on PCA notebook.
- **Stretch:** TD cheat-sheet pass on VPC/networking.
- **Deliverable:** Domain-1 practice set ≥ 70%.

### W3 · Aug 1–7 — SAA: Resilient + High-Performing (26% + 24%)
- **Core:** ☁️ Domains 2–3 (multi-AZ, ELB/ASG, RDS/Aurora, S3 tiers, caching, decoupling with SQS/SNS). 📘 Ch 8 (Unsupervised) light: k-means + DBSCAN + Gaussian mixtures for anomaly detection.
- **Stretch:** 📣 post #2 — "I compressed a photo with nothing but linear algebra, then watched it fall apart one dimension at a time" (PCA, from Ch 7).
- **Deliverable:** Domains 2–3 practice set ≥ 70%.

### W4 · Aug 8–14 — SAA: Cost + weak-area drilling
- **Core:** ☁️ Domain 4 (cost) + re-drill weakest domain from diagnostics. Start full-length TD practice exams. 🛠️ Start **P0 — Anomaly/Clustering demo** (uses Ch 8): pick a real-ish dataset, cluster + flag anomalies.
- **Deliverable:** first full TD exam done + reviewed (expect ~65–75%; TD runs harder than real).

### W5 · Aug 15–21 — SAA: exam-ready
- **Core:** ☁️ TD full exams until you hit **≥ 80% on fresh sets** (your go/no-go). Review every wrong answer. 🛠️ Finish + **deploy P0** (Streamlit/HF Space).
- **Deliverable:** ≥ 80% fresh TD + P0 live URL.

### W6 · Aug 22–28 — SIT SAA + reset
- **Core:** ☁️ **Sit AWS SAA-C03** (book it Tue–Thu). 📣 post #3 — "The handful of AWS ideas that actually changed how I'd deploy a model" (from six weeks of cert study; the pass is a footnote, not the headline). 🛠️ Write P0 case study; pin it. 📘 Begin Ch 11 reading.
- **Deliverable:** **SAA passed** 🎉 · P0 case study published.

---

## BLOCK B — Core DL + first projects + cert decision (W7–W12 · Aug 29 – Oct 9)

### W7 · Aug 29–Sep 4 — Ch 11 (Training DNNs) + makemore
- **Core:** 📘 Ch 11 trimmed core labs (init, activations, BatchNorm/LayerNorm, optimizers, LR schedules, dropout — the "practical guidelines" recipe). 🔬 Start **makemore** (char-level LM from scratch, Karpathy).
- **Stretch:** Ch 11 gradient-clipping + max-norm labs.
- **Deliverable:** a reusable `train()`/`build_mlp()` helper + makemore bigram/MLP stages working.

### W8 · Sep 5–11 — Finish Ch 11 + makemore
- **Core:** 📘 Wrap Ch 11 (assemble the "default recipe" pipeline). 🔬 Finish makemore (through the MLP/WaveNet stage). 📣 post #4 — "I taught a tiny neural net to invent names one character at a time — watching gibberish become almost-real words was unreasonably fun."
- **Deliverable:** makemore repo + annotated README (mini "paper→code" writeup).

### W9 · Sep 12–18 — Ch 12 (CNNs) + ⚠️ CERT DECISION
- **Core:** 📘 Ch 12 trimmed (conv/pool, the standard recipe, transfer learning, one classic arch e.g. ResNet block). 🛠️ Start **P1 — Transfer-learning image classifier** (fine-grained dataset, pretrained backbone). ☁️ **DECISION CHECKPOINT** — run the framework in `certs.md` §3: race MLA-C01 (sit by Sep 26) or target MLA-C02 (early 2027)? *Default: C02, keep momentum.*
- **Deliverable:** cert decision made & written down · P1 training working.

### W10 · Sep 19–25 — Finish Ch 12 + deploy P1
- **Core:** 📘 Finish Ch 12. 🛠️ **Deploy P1** to a HF Space with a Gradio UI; add an eval report (per-class accuracy, confusion matrix, a few failure cases). 📣 post #5 — P1: "with ~30 images per class, a frozen pretrained model embarrassed my from-scratch one — then I went hunting for where it still breaks."
- **If racing MLA-C01:** this + next week become cert-heavy; sit by **Sep 26**. Otherwise proceed.
- **Deliverable:** P1 live URL + case study.

### W11 · Sep 26–Oct 2 — Ch 13 (Sequences/Time-series)
- **Core:** 📘 Ch 13 trimmed (RNN/LSTM basics, forecasting a time series, the windowing/data-prep patterns; ARMA baseline). 🛠️ Start **P2 — Forecasting / predictive-maintenance** (the classic-ML hedge; XGBoost baseline vs. an LSTM).
- **Stretch:** multivariate / multi-step forecasting lab.
- **Deliverable:** P2 baseline + LSTM comparison notebook.

### W12 · Oct 3–9 — Deploy P2 (hedge project)
- **Core:** 🛠️ **Deploy P2** as a small FastAPI service with a metrics writeup (MAE/RMSE vs. baseline, honest failure analysis). 📣 post #6 — "I was sure the LSTM would win. The boring gradient-boosted trees won, and it wasn't close."
- **Deliverable:** P2 live + case study. **Block B done: 3 deployed things (P0, P1, P2) + 1 reproduction.**

---

## BLOCK C — Transformers/LLMs + anchor projects (W13–W19 · Oct 10 – Nov 27)

> This block is the heart of your specialization. Protect it.

### W13 · Oct 10–16 — Ch 14 (NLP + Attention) + nanoGPT start
- **Core:** 📘 Ch 14 trimmed (embeddings, a char-RNN, the attention mechanism, HF tokenizers). 🔬 Start **nanoGPT / build-nanogpt** (reproduce a GPT-2-class model; scale down to your hardware).
- **Deliverable:** attention understood cold + nanoGPT training loop running.

### W14 · Oct 17–23 — Ch 15 (Transformers) deep + nanoGPT
- **Core:** 📘 Ch 15 part 1 (the original Transformer: positional encoding, multi-head attention, build an encoder-decoder). 🔬 nanoGPT continues.
- **Deliverable:** you can draw the Transformer block from memory · nanoGPT generating text.

### W15 · Oct 24–30 — Finish nanoGPT + start P3
- **Core:** 🔬 Finish nanoGPT; write the "paper→code" annotated writeup. 📣 post #7 — "I rebuilt GPT-2 from scratch just to understand attention — the idea underneath is almost embarrassingly simple." 🛠️ Start **P3 — Domain-specific RAG API** (the anchor): pick a niche corpus (legal/medical/dev-docs).
- **Deliverable:** nanoGPT writeup published + P3 ingestion/chunking pipeline.

### W16 · Oct 31–Nov 6 — Ch 15 (LLMs) + P3 build
- **Core:** 📘 Ch 15 part 2 (encoder/decoder-only models, SFT/RLHF/DPO concepts, chatbot system, MCP — skim the tooling, grasp the concepts). 🛠️ P3 core: vector DB + retrieval + **reranking** + a generation endpoint.
- **Deliverable:** P3 answers questions end-to-end (rough).

### W17 · Nov 7–13 — Finish + deploy P3 (anchor)
- **Core:** 🛠️ Add **RAGAS eval**, wrap in **FastAPI + Docker**, deploy. Quantify: faithfulness, answer-relevancy, retrieval hit-rate, latency/cost. 📣 post #8 — P3: "my RAG demo answered everything confidently — then I checked how many answers the documents actually supported."
- **Deliverable:** **P3 live URL + eval dashboard + case study.** This is your flagship.

### W18 · Nov 14–20 — App B (Quantization) + LoRA repro
- **Core:** 📘 Appendix B (mixed precision, quantization, QAT/PTQ) — feeds the inference/MLOps wedge. 🔬 **LoRA/QLoRA fine-tune** reproduction (from-scratch adapter → PEFT; runs on free Colab T4). 🛠️ Start **P4 — LLM eval & observability pipeline** over P3.
- **Deliverable:** a fine-tuned adapter + P4 tracing scaffold.

### W19 · Nov 21–27 — P4 (your wedge) build
- **Core:** 🛠️ P4: RAGAS/DeepEval test suite + LangSmith (or Phoenix) tracing over the RAG API; a golden-dataset regression test; an "LLM CI" GitHub Action that fails the build if faithfulness drops. 📣 post #9 — "The least glamorous question in LLM apps, and the one I can't stop thinking about: how do you *know* the output is right?" **← your signature post.**
- **Deliverable:** P4 running as a CI gate on P3.

---

## BLOCK D — Agentic/MLOps + consolidate + position (W20–W24 · Nov 28 – Dec 30)

### W20 · Nov 28–Dec 4 — Ch 16 + Ch 17 + finish P4
- **Core:** 📘 Ch 16 trimmed (ViT + CLIP — multimodal intuition) + Ch 17 (speeding up Transformers / inference optimization — pairs with App B). 🛠️ Finish + polish P4; publish its case study.
- **Stretch:** a tiny CLIP zero-shot demo.
- **Deliverable:** P4 case study live.

### W21 · Dec 5–11 — P5 (agentic OR MLOps)
- **Core:** 🛠️ **P5** — pick the one that best rounds out your story:
  - **(a) Agentic workflow** — LangGraph tool-using agent (writes+runs Python / queries a DB) with evals + cost/latency guardrails, or
  - **(b) MLOps service** — one of your models behind Docker + CI + drift monitoring (Evidently/Grafana) + alerting.
  - *Recommendation: (a) if you want the "systems that act" wow; (b) if you want to hammer the reliability wedge harder.*
  - Deploy it. 📣 post #10 — P5: "I gave a model some tools and let it loose — it confidently did the wrong thing, so I spent a week on guardrails."
- **Deliverable:** P5 live + case study.

### W22 · Dec 12–18 — Ch 18/19 (light) + buffer
- **Core:** 📘 Ch 18 (autoencoders/GANs/**diffusion** — concepts + optional tiny DDPM demo for a generative signal) light; Ch 19 (RL) **skim only**. This week absorbs any slippage from Blocks B–C.
- **Stretch:** a small diffusion sample grid → 📣 post #11 — "I watched a model turn pure static into a picture, step by step. Still feels like magic."
- **Deliverable:** caught up; DL arc conceptually complete.

### W23 · Dec 19–25 — Consolidate the portfolio
- **Core:** Pin your **3 anchors** (P3 RAG API, P4 eval pipeline, + best of P1/P5). Overhaul each README to **Problem → Approach → Result → Impact** with numbers. Build/polish a one-page **portfolio site** (GitHub Pages) linking demos, repros, and posts. Publish the **LoRA** writeup if not already.
- **Deliverable:** 3 pinned, deployed, quantified projects + portfolio site.

### W24 · Dec 26–30 — Position for 2027
- **Core:** Create **Braintrust** + a **niched Upwork** profile (positioning from `brand.md`); write 3–5 quantified case studies into the profiles. 📣 post #12 — "Six months, a lot of broken training runs, and a few things I'm genuinely proud of — what building in public actually taught me." (Freelance availability as a quiet closing line, not the headline.) Draft the **MLA-C02 Q1-2027 prep plan** and a freelance-launch checklist.
- **Deliverable:** freelance profiles live · retrospective posted · 2027 runway set.

---

## Cadence summary

| Type | Count over 24 wks | Notes |
|---|---|---|
| 📘 Chapters | Ch 10 finish, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18(light), 19(skim), App B | 4–6 hrs each, trimmed |
| 🔬 Reproductions | makemore, nanoGPT, LoRA/QLoRA | packaged with writeups |
| 🛠️ Projects | P0–P5 (6) | P3 + P4 are the anchors; P0/P2/P5 flexible |
| ☁️ Certs | SAA-C03 (Aug) | ML cert decided in Sept |
| 📣 Posts | ~12 milestone posts + weekly small ones | batch weekly |

## If you fall behind (triage order — cut from the bottom)
1. Ch 19 (RL) — skip entirely.
2. Ch 18 diffusion demo → concept-only.
3. P5 → defer to Jan.
4. Ch 16 CLIP demo → concept-only.
5. makemore → fold into nanoGPT.
6. P0 or P2 → keep one, not both.

**Never cut:** SAA, Ch 11–12, Ch 15, nanoGPT, **P3 (RAG API), P4 (eval pipeline)**, the weekly posts.
