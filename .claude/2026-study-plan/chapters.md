# Chapter Guide — Trimmed to 4–6 hrs, Concept-First

Your old exercise sets were too long (Ch 11's `test1.md` alone is 12 parts / ~40 labs). This is the **trimmed spine**: for each chapter, the concepts you *must* own, a **core lab set that fits 4–6 hrs**, what to **skip/skim**, and the **project/reproduction it feeds**.

**When you reach a new chapter, ask me:** *"Generate the trimmed 4–6 hr lab set for Ch N in the ch10_test2 style."* I'll produce a tight, loose-task lab file like your existing ones — but capped.

**The anti-rabbit-hole rule:** if a "go further" / "wonder" prompt eats >20 min, write the question down and move on. Depth is bought in projects, not in chapter completionism.

---

## Part I — the last two chapters (do these *light*, during the SAA weeks)

### Ch 7 — Dimensionality Reduction  → feeds **P0**
- **Own:** curse of dimensionality; PCA (variance preservation, explained-variance ratio, choosing #components); when to reach for it (visualization, compression, speed-up before a model).
- **Core labs (~4 hrs):** (1) PCA on a real dataset; plot cumulative explained variance; pick #dims for 95%. (2) Use PCA to compress + reconstruct an image; eyeball the loss. (3) Run t-SNE/UMAP on the same data for a 2-D cluster picture.
- **Skip:** LLE, random projection internals, kernel-PCA math.
- **Feeds:** P0 (dimensionality reduction as preprocessing before clustering/anomaly detection).

### Ch 8 — Unsupervised Learning  → feeds **P0**
- **Own:** k-means (+ how to pick k), DBSCAN (density, no k needed), Gaussian mixtures for **anomaly detection**; semi-supervised labeling.
- **Core labs (~5 hrs):** (1) k-means + elbow/silhouette. (2) DBSCAN on non-spherical data; compare to k-means. (3) GMM anomaly detection: fit, score, threshold, flag outliers. (4) Use clustering to auto-label a few points, then train a classifier (semi-supervised).
- **Skip:** Bayesian GMM depth, exhaustive "other algorithms" tour.
- **Feeds:** **P0 — Anomaly/Clustering demo.**

### (Recommended insert) Ch 6 — Ensemble Learning / Boosting  → strengthens **P2**
You skipped this, but `xgboost` is already in your deps and **gradient boosting is the workhorse of freelance tabular work.** Do a **3–4 hr express pass** before P2: random forests, gradient boosting, **XGBoost/HistGradientBoosting**, early stopping, feature importance, a quick stacking demo. High ROI, low cost.

---

## Part II — the deep-learning arc (your main line)

### Ch 10 — Building NNs with PyTorch  → foundation (nearly done)
- **Status:** `ch10_test1` + `ch10_test2` mostly complete. **Finish only** labs 10–12: Optuna tuning, save/load (state_dict + safetensors), and `torch.compile`/TorchScript.
- **Own (checkpoint):** tensors/autograd, `nn.Module`, DataLoaders, custom/non-sequential models, multi-input/output, the training/eval loop. This is your bedrock — everything else assumes it.

### Ch 11 — Training Deep Neural Networks  → feeds **makemore**
Your `ch11_test1.md` is the comprehensive version — **do only this core subset:**
- **Own:** vanishing/exploding gradients; **He/Glorot init**; activation trade-offs (ReLU→GELU/SiLU); **BatchNorm vs LayerNorm** (and *why transformers use LN*); Adam/AdamW; **LR schedules** (1cycle, cosine); **dropout**; the chapter's "practical defaults" recipe.
- **Core labs (~6 hrs):**
  1. Lab 1.2 — watch activation variance drift, then fix it with He/Glorot init.
  2. Lab 2.2 + 2.8 — plot all activations; then hold everything fixed and swap only the activation (bake-off).
  3. Lab 3.1 + 3.4 — BatchNorm as first layer; then the train/eval-mode bug (feel it break).
  4. Lab 4.2 — LayerNorm vs BatchNorm at tiny batch size (*why transformers pick LN*).
  5. Lab 8.4 — implement Adam by hand once; then trust `AdamW`.
  6. Lab 10.1 + 10.7 — plot the schedules; run 1cycle vs constant (super-convergence).
  7. Lab 11.3 + 11.4 — dropout dial + Monte-Carlo dropout (uncertainty).
  8. Lab 12.1 — assemble the default recipe into one reusable pipeline.
- **Skip/skim:** orthogonal init, SELU self-normalization deep dive, Shampoo, sparse-model pruning, max-norm, warm-restart nuances, per-optimizer-by-hand marathon.
- **Feeds:** the `train()` recipe you'll reuse all plan long; **makemore**.

### Ch 12 — Deep Computer Vision (CNNs)  → feeds **P1**
Your `ch12_test1.md` is comprehensive — **core subset:**
- **Own:** convolution + pooling mechanics and **output-shape math**; the conv→ReLU→pool→dense recipe; **residual blocks / ResNet**; **transfer learning** (freeze → fine-tune, differential LRs); data augmentation.
- **Core labs (~6 hrs):**
  1. Part 1.2 + 2.1–2.2 — hand-built filters; then a real conv layer; predict output H×W before running.
  2. Part 3.1 + 3.4 — max vs avg pooling; global average pooling.
  3. Part 4.1–4.3 — build + train the standard-recipe CNN; find the flatten dim the hard way.
  4. Part 5.5 + 7.1 — residual block, then assemble **ResNet-34** from the recipe.
  5. Part 8.1–8.2 — **transfer learning**: swap head, freeze, fine-tune, differential LRs. *(This is P1.)*
  6. Part 5.3 — data augmentation pipeline.
- **Skim (concept-only, no full runs):** Inception/Xception/SENet internals, object detection/YOLO/NMS, segmentation/FCN, RevNets, dilated/transposed conv. Know *what they're for*, not every API.
- **Feeds:** **P1 — Transfer-learning image classifier.**

### Ch 13 — Sequences with RNNs & CNNs  → feeds **P2**
- **Own:** RNN/LSTM cells; the **time-series windowing/data-prep** pattern; forecasting baselines (naive, ARMA) vs. learned models; multi-step & multivariate framing.
- **Core labs (~5 hrs):** (1) prepare a windowed dataset from a raw series. (2) naive + linear baselines. (3) a simple RNN, then an LSTM; compare. (4) forecast several steps ahead.
- **Skip:** deep RNN zoo, seq2seq internals (you'll get sequence modeling properly via Transformers).
- **Feeds:** **P2 — Forecasting / predictive maintenance** (the classic-ML hedge).

### Ch 14 — NLP with RNNs & Attention  → feeds **nanoGPT + P3**
- **Own:** **embeddings**; char-RNN text generation; **the attention mechanism** (this is the hinge to Transformers); HF **tokenizers**; using pretrained models + the `pipeline` API.
- **Core labs (~5 hrs):** (1) train a char-RNN to generate text. (2) implement scaled-dot-product attention by hand. (3) tokenize with HF; run a pretrained sentiment model via `pipeline`.
- **Skip:** beam search internals, full encoder-decoder MT build (Ch 15 does Transformers properly).
- **Feeds:** attention intuition for **nanoGPT**; HF fluency for **P3**.

### Ch 15 — Transformers for NLP & Chatbots  ⭐ **the most valuable chapter** → feeds **nanoGPT + P3**
Give this **~8 hrs across two weeks** — do not trim the concepts, only the tooling breadth.
- **Own:** positional encoding; **multi-head attention**; encoder-only (BERT) vs decoder-only (GPT) vs encoder-decoder; pretraining vs fine-tuning; **SFT / RLHF / DPO** (concepts + when each); in-context / few-shot; turning an LLM into a chatbot; **RAG** and **MCP** (concepts).
- **Core labs (~8 hrs):**
  1. Build multi-head attention + a Transformer block from scratch (dovetails with nanoGPT).
  2. Load a small pretrained LLM (e.g. via HF); generate; do a QA prompt.
  3. Concept lab: write your own one-paragraph explanation of SFT vs RLHF vs DPO and when to use each (this shows up in client conversations).
  4. Stand up a minimal RAG loop (embed → retrieve → stuff → generate) — the seed of **P3**.
- **Skim:** training MT from scratch, exhaustive model zoo, library-specific TRL details (grasp, don't grind).
- **Feeds:** **nanoGPT** (architecture) + **P3 RAG API** (application).

### Ch 16 — Vision & Multimodal Transformers  → optional demo
- **Own:** **ViT** (images as patches → tokens); **CLIP** (contrastive text+image, zero-shot); the idea behind DETR/DINO.
- **Core labs (~4 hrs):** (1) run a pretrained ViT classifier. (2) CLIP **zero-shot** classification on your own images + a tiny text-image retrieval demo.
- **Skip:** Swin/Perceiver/Flamingo/BLIP internals (concept-only).
- **Feeds:** optional multimodal mini-demo; rounds out the "modern architectures" story.

### Ch 17 — Speeding Up Transformers  → feeds **P4/P5 (inference wedge)**
- **Own:** why inference is the cost center; KV-cache; attention-efficiency ideas (flash/linear attention at a concept level); batching; the levers you actually pull to cut latency/cost.
- **Core labs (~4 hrs):** (1) benchmark generation latency; add KV-cache / batching; measure. (2) pair with **Appendix B** quantization to show a real speed/quality trade-off.
- **Feeds:** the latency/cost numbers in **P3/P4/P5** case studies (your reliability+efficiency wedge).

### Ch 18 — Autoencoders, GANs, Diffusion  → optional generative demo
- **Own (light):** autoencoders (+ anomaly detection callback to Ch 8); the VAE idea; GAN vs **diffusion** (diffusion won — know why).
- **Core labs (~4 hrs, optional):** (1) a stacked autoencoder for reconstruction/anomaly. (2) *stretch:* a tiny DDPM on MNIST for a generative sample grid (eye-catching post).
- **Skip:** GAN-training-stability deep dive, discrete VAEs.
- **Feeds:** optional diffusion demo (a strong 2026 generative signal if you have time).

### Ch 19 — Reinforcement Learning  → **skim only**
- **Own (read, don't build):** what RL is; policy gradients vs value-based (Q-learning/DQN); where RL actually shows up (RLHF, control). Lowest freelance ROI — 1–2 hrs of reading, no project.

### Appendix A — Autodiff  → optional (30 min read)
You already have autograd intuition from Ch 10 and makemore. Skim if curious.

### Appendix B — Mixed Precision & Quantization  ⭐ → feeds **P4/P5**
Punches above its weight for your MLOps/inference wedge.
- **Own:** fp16/bf16/int8/int4; mixed-precision training; **PTQ vs QAT**; quantizing an LLM (bitsandbytes); the accuracy/latency/memory trade-off.
- **Core labs (~4 hrs):** (1) quantize a trained model (PTQ); measure size + latency + accuracy delta. (2) load a pre-quantized LLM and benchmark it. Turn the numbers into a post.
- **Feeds:** the efficiency story in **P4/P5**; "I made the model 4× smaller for a 1% accuracy hit" is a great client-facing line.

---

## Chapter → Artifact map (the through-line)

| Chapter(s) | Artifact it feeds |
|---|---|
| Ch 7, 8 (+6) | **P0** anomaly/clustering demo · sharpens **P2** |
| Ch 11 | **makemore** reproduction · your reusable `train()` |
| Ch 12 | **P1** image classifier |
| Ch 13 (+6) | **P2** forecasting/predictive-maintenance (hedge) |
| Ch 14, 15 | **nanoGPT** reproduction · **P3** RAG API |
| Ch 15, App B, fine-tuning | **LoRA/QLoRA** reproduction · **P4** eval pipeline |
| Ch 16 | optional multimodal demo |
| Ch 17, App B | latency/cost numbers in **P3/P4/P5** |
| Ch 18 | optional diffusion demo |
