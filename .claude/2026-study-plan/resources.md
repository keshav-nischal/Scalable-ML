# Resource Stack (with When-to-Use)

The 2026 consensus is to **layer a stack on Géron's fundamentals**, not chase one more course. Standardize on **PyTorch + Hugging Face** — your Géron PyTorch edition transfers directly to everything below. **Beware the three traps: tutorial hell, no original projects, never deploying.** Build early, build your own, deploy at least once.

---

## Core learning stack

| Resource | When to use | Link |
|---|---|---|
| **Karpathy — Neural Networks: Zero to Hero** (+ makemore, nanoGPT, nanochat) | Right after Géron's DL chapters — true from-scratch intuition for backprop + attention; the "why" under the library calls | https://karpathy.ai/zero-to-hero.html |
| **fast.ai — Practical Deep Learning** | Fast, code-first breadth pass (CV/NLP/tabular); the DL course people actually finish; use for momentum | https://course.fast.ai/ |
| **Hugging Face LLM Course** | Your primary applied LLM/NLP track once attention clicks; runs on the real production Hub | https://huggingface.co/learn |
| **DeepLearning.AI short courses** | Targeted 1–2 hr deep dives on RAG, fine-tuning, agents, eval, deployment — design-pattern supplements as you build each project | https://www.deeplearning.ai/short-courses/ |
| **Made With ML (Anyscale)** | Close the deployment/production gap — gentlest MLOps on-ramp (Git, CLI, REST, CI/CD, monitoring) | https://madewithml.com/courses/mlops/ |
| **MLOps Zoomcamp (DataTalks.Club)** | Free alternative/complement to Made With ML; project-based MLOps | https://github.com/DataTalksClub/mlops-zoomcamp |
| **Raschka — Build an LLM from Scratch** | Rigorous, book-paced complement to Karpathy if you prefer structured written treatment | https://sebastianraschka.com/llms-from-scratch/ |
| **Alammar & Grootendorst — Hands-On Large Language Models** | Applied LLM companion — using/fine-tuning LLMs + RAG/search; bridge to production | https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/ |
| **KDnuggets — LLM Engineer Roadmap 2026** | Sequencing checklist: foundations → prompting/tools → retrieval → fine-tuning/alignment → serving/ops | https://www.kdnuggets.com/the-roadmap-to-becoming-an-llm-engineer-in-2026 |

---

## Reproduction targets (ranked, with effort)

| # | Target | Effort | Why it's a strong signal | Link |
|---|---|---|---|---|
| 1 | **makemore** | 8–15 h | Best on-ramp; autograd/backprop/embeddings by hand | https://github.com/karpathy/makemore |
| 2 | **nanoGPT / build-nanogpt** ⭐ | 15–30 h | Highest-signal; legible GPT-2 architecture in a few hundred lines | https://github.com/karpathy/build-nanogpt |
| 3 | **LoRA/QLoRA fine-tune** ⭐ | 10–20 h | Most directly hireable; QLoRA runs on free Colab T4 | https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms |
| 4 | **Small DDPM diffusion** (stretch) | 15–30 h | Strongest generative signal; eye-catching samples | https://www.codersarts.com/post/how-to-build-a-diffusion-model-from-scratch-in-pytorch-ddpm-ddim-classifier-free-guidance |
| 5 | **ViT → small CLIP** (stretch) | 10–20 h + 15–25 h | Multimodal backbone of 2026 | https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/ |

**Writeup models (imitate these):** [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) · [labml.ai annotated implementations](https://nn.labml.ai/index.html).
**Credential upgrade:** [ML Reproducibility Challenge — official NeurIPS 2026 track](https://blog.neurips.cc/2026/05/04/mlrc-2026-reproducibility-as-an-official-track-at-neurips/) (TMLR/ReScience publication).

---

## Project / production toolchain (2026 essentials)

- **Modeling:** PyTorch, Hugging Face (Transformers, Datasets, **PEFT**), scikit-learn, **XGBoost**.
- **LLM apps:** LangChain / LlamaIndex, **LangGraph** (agents), vector DBs (**Chroma / FAISS / Qdrant**), rerankers (cross-encoders / Cohere rerank).
- **Evaluation (your wedge):** **RAGAS**, **DeepEval**, **LangSmith**, **Phoenix/Arize**, LLM-as-judge.
- **Serving/ops:** **FastAPI**, **Docker**, **GitHub Actions**, Railway/Render/Fly, **vLLM** (serving), **bitsandbytes** (quantization), Evidently/Grafana (monitoring).
- **Demos:** Gradio, Streamlit, Hugging Face Spaces.
- **Optional-but-valuable (don't front-load):** DSPy, GraphRAG.

---

## AWS cert resources

**SAA-C03:** [Exam guide PDF](https://d1.awsstatic.com/training-and-certification/docs-sa-assoc/AWS-Certified-Solutions-Architect-Associate_Exam-Guide.pdf) · [Tutorials Dojo practice exams](https://portal.tutorialsdojo.com/courses/aws-certified-solutions-architect-associate-practice-exams/) · [TD cheat sheets](https://tutorialsdojo.com/aws-cheat-sheets/) · [Maarek Udemy](https://www.udemy.com/course/aws-certified-solutions-architect-associate-saa-c03/).

**MLA-C01/C02:** [Official ML Engineer cert page](https://aws.amazon.com/certification/certified-machine-learning-engineer-associate/) · [MLA-C01 exam guide](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-01/machine-learning-engineer-associate-01.html) · [Skill Builder MLA-C01 prep](https://skillbuilder.aws/category/exam-prep/machine-learning-engineer-associate-MLA-C01) · [Maarek & Kane MLA-C01](https://www.udemy.com/course/aws-certified-machine-learning-engineer-associate-mla-c01/) · [Tutorials Dojo MLA-C01](https://tutorialsdojo.com/aws-certified-machine-learning-engineer-associate-mla-c01-exam-guide/) · [MLA-C02 update blog](https://aws.amazon.com/blogs/training-and-certification/updates-to-aws-certified-machine-learning-engineer-associate-mla-c02/).

---

## Freelance platforms

- **Braintrust** — https://www.usebraintrust.com (flat 15% client fee, keep 100%). Best home base.
- **Upwork** — https://www.upwork.com (largest; win by niching, not generic AI/ML bidding).
- **Gun.io** — https://www.gun.io · **Toptal** — https://www.toptal.com (curated, higher rates, heavier vetting).

---

## Communities to stay current

- r/MachineLearning, r/LocalLLaMA, r/learnmachinelearning · Hacker News · Papers with Code · Latent Space + Sebastian Raschka's *Ahead of AI* newsletters · key practitioners on X/LinkedIn.
