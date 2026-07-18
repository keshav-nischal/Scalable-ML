# Decision Brief: 6-Month ML Study + Freelance Plan (Jul–Dec 2026)

*Grounding research for an SDET at Optmyzr (B.Tech, GSoC 2024 contributor) working through Géron's Hands-On ML (Scikit-Learn + PyTorch, 2025). Goal: ML excellence, public reputation, and USD freelance income in 2027. Budget ~2–3 hrs/day.*

---

## 1. Freelance ML Market 2026 — State, Direction, Earning Path, Platforms

**State: booming aggregate demand, bifurcated reward structure.** Freelance AI skill demand grew ~109% YoY — roughly 4x faster than other in-demand skills ([Botpool, State of AI Freelancing 2026](https://www.botpool.ai/blog/state-of-ai-freelancing-trends-and-data); the 109% figure traces to [Upwork's In-Demand Skills 2026](https://investors.upwork.com/news-releases/news-release-details/upworks-demand-skills-2026-demand-top-ai-skills-more-doubles-ai) and is independently confirmed). Upwork's AI Gross Services Volume is >$300M (+50% YoY). But the generic "AI & Machine Learning" category on open marketplaces is contested — **winning requires niching, not generic bidding.**

**Direction: the money is in GenAI application work, not classic ML demos.** Fastest-growing niches: AI video (+329%), AI integration (+178%), data annotation (+154%), AI image gen (+95%), AI chatbot dev (+71%) ([Upwork In-Demand Skills 2026](https://investors.upwork.com/news-releases/news-release-details/upworks-demand-skills-2026-demand-top-ai-skills-more-doubles-ai); restated by Botpool). **Caveat (verification-flagged):** these are *growth rates off small bases*, and the "classic tabular ML is commoditizing" claim is NOT supported by the underlying data — BLS-based outlooks project ML-engineer roles growing ~22% over the decade and classic-ML skills appear in ~77% of 2026 data-scientist postings ([365 Data Science](https://365datascience.com/career-advice/career-guides/machine-learning-engineer-job-outlook-2025/)). **Treat classic ML as slower-growing off a large base, not dead. A hybrid portfolio (GenAI + solid classic ML) is the evidence-backed hedge.**

**Realistic earning path.** Verified rate anchors ([Upwork cost-to-hire](https://www.upwork.com/hire/machine-learning-experts/cost/); [Second Talent](https://www.secondtalent.com/resources/freelance-ml-engineer-hourly-rate-us/); [Jobbers](https://www.jobbers.io/the-global-freelance-hourly-rate-index-2026-real-rates-by-skill-country-and-experience-level/)):
- ML engineer median ~$100/hr; typical $50–$200 (beginner $50–80, intermediate $80–120, advanced $120–200+).
- Specialists reach $100–$300/hr; regulated verticals pay most (fintech AI ~$120–300, healthcare AI ~$100–250 — directionally confirmed; some sources place senior/regulated even higher, so these anchor *low* if anything).
- For a strong India-based freelancer billing USD, geographic arbitrage is very favorable (a $50 gig ≈ ₹4,100+) ([Infinity](https://www.infinityapp.in/blog/how-to-use-ai)).

**Project/outcome pricing beats hourly** — but use *verification-corrected* tiers, NOT the raw research numbers (the original figures were mis-sourced; see §9). Independently-verified tiers ([Biztoolkit](https://www.biztoolkit.co/post/freelance-chatbot-developer-rates-in-2026-ai-pricing); [Kellton](https://www.kellton.com/kellton-tech-blog/custom-ai-chatbot-development-llm-rag); [Chipp](https://chipp.ai/blog/ai-chatbot-pricing-guide-how-much-charge/)):
- Simple/FAQ bot: $500–$2,000
- Custom bot with integrations: $2,000–$8,000
- Full RAG/knowledge-base build (production): $15,000–$45,000; mid-complexity $75k–$120k; true enterprise $300k+
- **Maintenance/service retainers: $500–$3,000/month** (do NOT count LLM API + infra operating spend as your income — that was the research's key error).

**Best platforms.** Curated networks beat open marketplaces for a strong intermediate ([Jobbers platforms guide](https://www.jobbers.io/best-platforms-to-hire-ai-ml-freelancers-in-2026-complete-guide/)):
- **Braintrust** — most freelancer-favorable: flat 15% *client-side* fee, talent keeps 100% ([usebraintrust.com/payments](https://www.usebraintrust.com/payments) — confirmed). *Note: the marketplace is usebraintrust.com, not "braintrust.dev".*
- **Gun.io** — ~$100–$200/hr (low end slightly optimistic per some sources).
- **Toptal** — heavy, undisclosed markup; a $150/hr client rate may net ~$70–$100 (exact figure unverifiable — Toptal bars rate disclosure).
- **Upwork** — largest, but the generic AI/ML pool is contested; use it niched.

**What clients buy:** LLM/RAG chatbots on company knowledge bases (LangChain + vector DB like Pinecone/Chroma + GPT-4o/Claude-class models), AI agents/multi-agent systems, and LLM integration ([Upwork RAG developers](https://www.upwork.com/hire/rag-developers/)). Clients want **proof-of-results over certifications** — 3–5 quantified case studies ("cut review time 70%"); 69% of companies emphasize data-based selection criteria ([Resumly](https://www.resumly.ai/blog/freelance-portfolio-that-wins-for-software-engineers-in-2026)).

---

## 2. High-Demand / Low-Supply Skills (Ranked) — and What to Avoid

The consistent signal is a **"production gap"**: pay clusters where models become reliable, evaluated, and cheap in production, not around theory or demos ([mshojaei77 market analysis](https://mshojaei77.github.io/market_analysis.html); [Futureproofing.dev](https://www.futureproofing.dev/resources/ai-talent-gap/ai-engineer-demand-2026)). Overall AI-engineer demand exceeds supply ~3.2:1.

**TARGET (scarce, high-premium) — ranked:**

1. **LLM evaluation & observability** — clearest line between demo-builders and production engineers; on only ~1/3 of resumes. Tools: RAGAS, DeepEval, LangSmith, LLM-as-judge, golden datasets. *(The ~1/3 figure is single-source/low-confidence — see §9 — but the underrepresentation trend is well-supported.)*
2. **LLMOps / MLOps** — 25–40% salary premium; genuine bottleneck (85%+ of ML projects never reach production). Pipelines, CI/CD for ML, monitoring, drift, model registries, cloud, GPU cost optimization ([Acceler8 Talent](https://www.acceler8talent.com/resources/blog/the-most-in-demand-machine-learning-roles-in-2026--managing-the-ai-talent-frontier/); [Kore1](https://www.kore1.com/mlops-engineer-salary-guide/)).
3. **Production-grade agentic systems** — fastest-growing (+280% YoY postings), still immature/under-supplied; differentiator is reliability + evals + cost control, not just frameworks (LangGraph, MCP) ([jobsbyculture](https://jobsbyculture.com/blog/agentic-ai-hiring-boom-2026)).
4. **RAG-at-scale** — basic RAG is now *table stakes*; value moved to retrieval quality, reranking, eval, latency/cost ([Dextralabs Production RAG 2025](https://dextralabs.com/blog/production-rag-in-2025-evaluation-cicd-observability/)).
5. **Inference optimization / quantization** — scarcest, $300K+ comp; AWQ/GPTQ, 8/4-bit, vLLM serving. Higher-hardware; a stretch niche.
6. **Data engineering for ML & AI governance/security** — quieter but durable under-supplied foundations; AI projects stall on data pipelines, not models.
7. **Fine-tuning (LoRA/QLoRA, SFT, DPO)** — high-value but *bounded* — ~70% of production problems solved by RAG/prompting first. A specialized second tool, not a primary bet ([BigData Boutique](https://bigdataboutique.com/blog/fine-tuning-llms-when-rag-isnt-enough)).

**Your unique wedge:** SDET/testing + GSoC background maps *directly* onto #1 and #2 — "the ML engineer who actually tests, validates, and hardens models for production." This is positioning few competitors can claim. Lean into it.

**AVOID as portfolio centerpiece (saturated/low-signal):**
- From-scratch CNN on MNIST, generic Kaggle notebooks, theory-only tabular regression, pure prompt-only work.
- Basic RAG/LangChain *as your selling point* (necessary to have, useless as a differentiator — ~70% of resumes list it; PyTorch/Python are ~98% universal baselines).
- **Nuance (verification):** Titanic/Iris/MNIST are best described as *low-return uses of scarce build time*, not "actively harmful" — the strong "net-negative" framing overstates the evidence ([Statology explicitly refutes it](https://www.statology.org/what-titanic-iris-and-house-prices-say-about-your-portfolio/)). Acceptable as unpinned learning exercises; never as headline projects.

---

## 3. Portfolio Strategy — What Makes a Project High-Signal + Ranked Ideas

**What makes a project high-signal** (well-corroborated across sources):
- **A live, publicly deployed demo URL** (Hugging Face Space / Streamlit Cloud / Railway) is the dividing line between "data science notebook" and "ML engineering" ([Let's Data Science](https://letsdatascience.com/blog/the-ml-portfolio-that-actually-gets-you-hired-in-2026)).
- **End-to-end packaging:** Dockerfile, requirements/Poetry, tests/, CI (GitHub Actions).
- **Real/messy self-collected data** beats a clean Kaggle CSV — proves you can do pipeline "dirty work."
- **Business framing + honest metrics:** structure as Problem → Approach → Result → Impact with before/after numbers ("latency 640ms → 95ms"), including failure analysis.
- **Three strong deployed projects beat ten notebooks.** Reviewers spend <2 min; pin your best, write a README that states the business problem first.

*Caveat: the "70% of ML roles require end-to-end/MLOps" stat is unsupported (see §9) — real posting data clusters at CI/CD ~34%, monitoring ~42%, cloud ~55%. The qualitative "deploy, don't stop at notebooks" guidance is robust; the specific 70% is not.*

**Ranked project ideas** (demand × demonstrability × foundation-building):

| # | Project | Rationale | Rough effort |
|---|---------|-----------|--------------|
| 1 | **Domain-specific RAG API** (legal/medical/product docs) — vector DB + chunking + reranking + RAGAS eval, behind FastAPI + Docker, deployed | The 2026 "Golden Project"; hits the #1 client-demand pattern *and* your eval wedge | ~15–30 hrs |
| 2 | **LLM eval & observability pipeline** (RAGAS/DeepEval/LangSmith tracing over #1) | Separates junior from senior signal; directly showcases your SDET moat | ~10–20 hrs |
| 3 | **Tool-using / agentic workflow** (LangGraph Data-Analyst agent that writes+runs Python, self-corrects) | Fastest-growing category; shows systems that *act*, not just chat | ~20–35 hrs |
| 4 | **MLOps service with drift monitoring** (Docker + CI + Grafana/observability + drift alerts) | Proves production-readiness (models decay); the durable premium skill | ~20–30 hrs |
| 5 | **Predictive maintenance on self-scraped/sensor time-series** (XGBoost/LSTM) | Classic ML on a real business problem managers actually pay for; hedges the GenAI-only risk | ~15–25 hrs |
| 6 | **LoRA/QLoRA fine-tune on a domain corpus** (from-scratch adapter + PEFT production path) | Highly hireable PEFT skill; runs on free Colab T4 | ~10–20 hrs |
| 7 | **Real-time inference microservice + latency optimization** (load test, before/after benchmark) | Demonstrates the engineering discipline clients pay for | ~10–20 hrs |
| 8 | **Text-to-SQL over a real database** (schema injection + safe execution) | Concrete, business-useful LLM app with clear success metrics | ~10–15 hrs |

Recommended anchor trio: **#1 (RAG API) + #2 (eval on it) + #4 (MLOps/monitoring)** — covers the exact breadth clients screen for and foregrounds your testing differentiation.

---

## 4. Paper-Reproduction Targets (Ranked)

Reproductions build reputation *because almost no one does them* — but the payoff comes from **packaging** (annotated code + a clear paper-to-code writeup + why it matters), not the reproduction alone. A cutting-edge reproduction with no explanation can read as novelty-chasing ([InterviewNode](https://www.interviewnode.com/post/ml-engineer-portfolio-projects-that-will-get-you-hired-in-2025)). Do a *small number of deep* reproductions. Model writeups on [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) / [labml.ai](https://nn.labml.ai/index.html).

Karpathy's nano-series is the canonical proof this works (nanoGPT ~61k, micrograd ~17k, nanochat now ~56k GitHub stars — the research's "43k" for nanochat was a stale January figure; all other counts confirmed) ([nanoGPT](https://github.com/karpathy/nanogpt); [nanochat](https://github.com/karpathy/nanochat)).

**Ranked sequence:**

1. **makemore** (char-level LM from scratch) — best on-ramp; autograd/backprop/embeddings. **~8–15 hrs.** ([karpathy](https://github.com/karpathy/makemore))
2. **nanoGPT / build-nanogpt** (reproduce GPT-2-class) — highest-signal, most legible architecture; ~300-line model + ~300-line training loop. Full 124M reproduction realistically wants an 8×A100 node (~4 days) but scales down to single-GPU/CPU. **~15–30 hrs** following the video+code. ([build-nanogpt](https://github.com/karpathy/build-nanogpt))
3. **LoRA/QLoRA fine-tune** (from-scratch adapter, then PEFT) — most directly hireable; QLoRA runs on free Colab T4. **~10–20 hrs.** ([Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms))
4. **Small DDPM diffusion model** (+ DDIM/CFG on MNIST/CIFAR-10, U-Net) — strongest *generative* signal (diffusion dominates), eye-catching samples. **~15–30 hrs.** ([Codersarts](https://www.codersarts.com/post/how-to-build-a-diffusion-model-from-scratch-in-pytorch-ddpm-ddim-classifier-free-guidance))
5. **ViT → small CLIP** — vision transformers + multimodal alignment (backbone of 2026 multimodal). **~10–20 hrs ViT; +15–25 hrs CLIP.** ([ViT tutorial](https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/))
6. **From-scratch RAG system** (systems reproduction, not a paper) — hybrid dense+BM25 + cross-encoder reranking, quantified (NDCG@5). Most hiring-aligned. **~15–30 hrs.**

*Skip as a centerpiece:* DCGAN/MNIST — good for GAN fundamentals but a weaker 2026 signal since diffusion overtook GANs.

**Formal credibility upgrade:** the [ML Reproducibility Challenge is now an official NeurIPS 2026 track](https://blog.neurips.cc/2026/05/04/mlrc-2026-reproducibility-as-an-official-track-at-neurips/) with TMLR/ReScience publication — the highest-prestige way to convert a reproduction into a citable credential.

---

## 5. AWS MLA-C01 — Verified Facts, Hours, Resources

**Verified exam facts** (confirmed against two official AWS properties):
- **65 questions** (50 scored + 15 unscored), **130 minutes**, **$150 USD**, pass **720/1000**, valid **3 years** ([AWS official page](https://aws.amazon.com/certification/certified-machine-learning-engineer-associate/); [exam guide](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-01/machine-learning-engineer-associate-01.html)).
- New question types: multiple response, ordering, matching.

**Domain weights** (confirmed exactly):
- Data Preparation for ML — **28%**
- ML Model Development — **26%**
- ML Solution Monitoring, Maintenance & Security — **24%**
- Deployment & Orchestration of ML Workflows — **22%**

*Correction to research framing:* the exam is NOT "heavily MLOps, not model theory." The two largest domains (Data Prep 28% + Model Dev 26% = 54%) are data/modeling. It's production-*aware* but balanced (28/26/24/22). **Study effort should be spread fairly evenly with a slight tilt to data prep**, not concentrated on deployment.

Content is **SageMaker-centric** (Data Wrangler, Feature Store, Pipelines, Model Monitor, Clarify, JumpStart, endpoint types) + supporting services (Glue, Athena, Kinesis, Step Functions, CodePipeline, CloudWatch) + IAM security + cost optimization ([Tutorials Dojo](https://tutorialsdojo.com/aws-certified-machine-learning-engineer-associate-mla-c01-exam-guide/)).

**Study hours:** AWS-familiar but new to SageMaker → **~80–120 hrs (~8–12 weeks at ~10 hrs/wk)**. This is sound, even mildly conservative ([examcert](https://www.examcert.app/blog/aws-mla-c01-worth-it/)).

**⚠️ CRITICAL TIMING:** MLA-C01 is being retired. **Last day to take MLA-C01 in English is Sept 28, 2026**; MLA-C02 beta registration opens Sept 1, 2026 ([AWS blog — confirmed](https://aws.amazon.com/blogs/training-and-certification/updates-to-aws-certified-machine-learning-engineer-associate-mla-c02/)). **From July 2026, if the plan wants the current exam, it must book on/before Sept 28, 2026.** Otherwise plan for MLA-C02 (beta Sept 29+; adds GenAI/agentic/Bedrock topics; standard version early 2027). This is a hard constraint on the 6-month timeline — MLA-C01 must be front-loaded to Aug–Sept.

**Resources + sequence:**
1. [Official Exam Guide](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-01/machine-learning-engineer-associate-01.html) — read first, map every topic to it.
2. [AWS Skill Builder MLA-C01 Prep](https://skillbuilder.aws/category/exam-prep/machine-learning-engineer-associate-MLA-C01) — free Standard plan; Enhanced adds Builder Labs (hands-on SageMaker) + pretest.
3. [Maarek & Kane Udemy course](https://www.udemy.com/course/aws-certified-machine-learning-engineer-associate-mla-c01/) — conceptual coverage (watch 1.25–1.5x; verbose).
4. [Tutorials Dojo practice exams](https://tutorialsdojo.com/aws-certified-machine-learning-engineer-associate-mla-c01-exam-guide/) — closest to real difficulty; drill weak areas.
5. AWS Official Practice Question Set — calibrate late; aim for consistent ~80%+ before booking. *(Note: no dedicated Adrian Cantrill MLA-C01 course exists as of 2025–26.)*

---

## 6. AWS SAA-C03 — Verified Facts, Revision Hours, Resources

**Verified facts** (confirmed; SAA-C03 remains current in 2026, **no SAA-C04 announced** — SEO blogs claiming one are false):
- **65 questions** (50 scored + 15 unscored), **130 minutes**, **$150 USD**, pass **720/1000**, **compensatory scoring** (pass overall, not per-domain; no wrong-answer penalty — answer everything) ([AWS page](https://aws.amazon.com/certification/certified-solutions-architect-associate/); [exam guide](https://docs.aws.amazon.com/aws-certification/latest/solutions-architect-associate-03/solutions-architect-associate-03.html)).

**Domain weights:** Design Secure Architectures **30%** (largest — prioritize IAM/KMS/encryption/security groups), Resilient **26%**, High-Performing **24%**, Cost-Optimized **20%**.

**Revision hours:** for a genuine *refresh* of prior mastery, **~20–40 hrs over 2–4 weeks**. **⚠️ Caveat (verification):** this only holds if it's truly revision. If it's effectively a *first pass*, independent sources cluster at **60–80 hrs** for AWS-familiar candidates ([CBT Nuggets](https://www.cbtnuggets.com/how-long-to-study/saa-c03); [DiviTrain](https://www.divitrain.com/blogs/it-certifications/how-long-to-study-for-aws-certified-solutions-architect-associate-realistic-timelines)). **Budget the higher figure as a buffer if unsure.**

**Revision resources** (practice-test-driven loop):
1. [Official Exam Guide PDF](https://d1.awsstatic.com/training-and-certification/docs-sa-assoc/AWS-Certified-Solutions-Architect-Associate_Exam-Guide.pdf) — confirm nothing drifted.
2. [Tutorials Dojo practice exams](https://portal.tutorialsdojo.com/courses/aws-certified-solutions-architect-associate-practice-exams/) — closest-to-real; core of the loop. **TD scores run ~8–13 pts harder than the real exam** — aim ~80%+ on fresh sets as go/no-go.
3. [TD free cheat sheets](https://tutorialsdojo.com/aws-cheat-sheets/) — fast service-comparison refreshers.
4. [Maarek practice tests](https://www.udemy.com/course/aws-certified-solutions-architect-associate-saa-c03/) — strong second question bank.

**SAA ↔ MLA synergy:** SAA supplies IAM/VPC/S3/cost/networking foundations that MLA's SageMaker pipelines assume — the overlap reduces MLA study load. Doing SAA revision *first* (Aug) then MLA (Aug–Sept) is a sensible order given the Sept 28 MLA-C01 deadline.

---

## 7. Personal-Brand / LinkedIn Playbook

**Content types that work:**
- **First-person "How I" case studies with quantified outcomes** out-perform generic AI explainers and self-select qualified inbound (content-led outreach converts 4–6x cold outreach) ([ExpertsHub](https://expertshub.ai/blog/linkedin-strategies-for-ai-freelancers/)).
- **PDF carousels/document posts** are the highest-engagement format (drive saves + reposts, the strongest reach signals). Text needs a strong first-150-char hook; keep external links in the *first comment*; **avoid polls and "Agree? comment" bait** (spam-classified) ([Dataslayer](https://www.dataslayer.ai/blog/linkedin-algorithm-february-2026-whats-working-now); [Buffer](https://buffer.com/resources/how-often-to-post-on-linkedin/)). *(One source-to-source tension: video engagement figures differ — treat video as secondary, carousels primary.)*

**Sustainable cadence (fits 2–3 hrs/day study budget):**
- **2–3 posts/week + daily thoughtful comments** (LinkedIn counts comment impressions — comments act as micro-posts). The 1→2–5 posts/week jump is the biggest marginal gain.
- **One 90-min weekly batch block** turns the week's learning into all posts. Consistency for years beats intensity for months.
- 80/20 rule: 80% engaging with others, 20% original posting.

**Profile as a landing page, not a résumé:** headline formula "I help [teams] ship reliable ML | MLOps + Model Testing | ex-SDET, GSoC Contributor | Open to freelance"; Featured section = 3 pinned proofs (best repo, a reproduction writeup, a quantified project); About opens with a client pain and ends with a discovery-call CTA.

**Leverage GSoC + SDET (your two biggest differentiators):**
- Label it **"Google Summer of Code 2024 Contributor, [Org Name]"** (spell out; org as affiliation, never "Google" as employer). Public merged PRs = instant click-through trust ([Bloom](https://blog.bloomhq.ai/how-to-leverage-open-source-to-land-job/)).
- Position SDET as your moat: **"the ML engineer who tests, evaluates, and hardens models for production."** Reliability/eval is exactly what freelance ML buyers fear getting wrong — and few competitors can claim it. This ties your brand to the scarce skills in §2.
- Paper reproductions double as carousel series + lessons-learned posts + portfolio pieces — high authority per unit of effort.

**Content engine:** 3 pillars (Technical depth / Learning-in-public / Freelance+industry takes) × 10 angles (Tip, Stat, Step, Lesson, Benefit, Reason, Mistake, Example, Question, Personal Story) = 30 seeds. Jot one seed per study session ([The Data Writer](https://thedatawriter.substack.com/p/use-this-prompt-to-generate-30-data)).

---

## 8. Complementary Resources to Géron (Curated, with When-to-Use)

The 2026 consensus is to layer a *stack* on Géron's fundamentals, not one course ([DataCamp](https://www.datacamp.com/blog/best-ai-courses); [KDnuggets LLM roadmap](https://www.kdnuggets.com/the-roadmap-to-becoming-an-llm-engineer-in-2026)). Standardize on **PyTorch + Hugging Face** — the Géron PyTorch edition transfers directly to every downstream resource.

| Resource | When to use |
|----------|-------------|
| [Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) (+ makemore, nanoGPT, nanochat, microgpt) | Immediately after Géron's DL chapters — builds true from-scratch intuition for backprop and attention; the "why" under the library calls |
| [fast.ai — Practical Deep Learning](https://course.fast.ai/) | Fast code-first breadth pass (CV/NLP/tabular); the DL course people actually finish; use for momentum |
| [Hugging Face LLM Course](https://huggingface.co/learn) | Primary applied LLM/NLP track once attention "clicks"; runs on the real production Hub |
| [DeepLearning.AI short courses](https://www.deeplearning.ai/short-courses/) | Targeted 1–2 hr deep dives on RAG, fine-tuning, agents, eval, deployment — design-pattern supplements |
| [Made With ML (Anyscale)](https://madewithml.com/courses/mlops/) | Close the deployment/production gap; gentlest MLOps on-ramp (Git, CLI, REST, CI/CD, monitoring). Free alt: MLOps Zoomcamp (DataTalks.Club) |
| [Raschka — Build an LLM from Scratch](https://sebastianraschka.com/llms-from-scratch/) | Book-paced rigorous complement to Karpathy if you prefer structured written treatment |
| [Alammar & Grootendorst — Hands-On LLMs](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/) | Applied LLM companion — using/fine-tuning LLMs + RAG/search, bridge to production |
| [KDnuggets LLM Engineer Roadmap 2026](https://www.kdnuggets.com/the-roadmap-to-becoming-an-llm-engineer-in-2026) | Sequencing checklist: foundations → prompting/tools → retrieval → fine-tuning/alignment → serving/ops; essential-vs-optional tooling |

**Highest career-leverage Géron chapters:** transformers/attention → deployment/serving. LLM fine-tuning carries ~+40–60% salary premium, MLOps +25–40% ([Signify](https://www.signifytechnology.com/news/machine-learning-engineer-salary-benchmarks-us-market-2025-2026/)). **Don't stop at CNNs.** Essential 2026 toolchain: PyTorch, HF (Transformers/Datasets/PEFT), vector DBs (FAISS/Chroma), LoRA/DPO, serving/monitoring. Optional-but-valuable (don't front-load): DSPy, GraphRAG, Ragas/Phoenix.

**Avoid the three self-learner traps:** tutorial hell, no original projects, never deploying past a notebook. Build early, build your own, deploy at least once.

---

## 9. Contested / Uncertain Points — Do NOT Over-Rely On These

Flagged by adversarial verification. Use these caveats to keep the plan robust:

1. **"Classic tabular/predictive ML is commoditizing" — NOT SUPPORTED.** The growth-rate data (AI video +329%, etc.) is real but measures *relative growth off small bases*, not absolute demand, and says nothing about classic ML declining. BLS-based outlooks show classic ML growing ~22%/decade off a large base. **Keep a hybrid portfolio; don't bet against classic ML.**

2. **The "70% of ML roles require end-to-end/MLOps" stat — REFUTED / mis-attributed.** Real posting data: CI/CD ~34%, monitoring ~42%, cloud ~55%, explicit MLOps ~14% ([axialsearch](https://axialsearch.com/insights/ai-ml-engineering-jobs/)). The qualitative "deploy your projects" advice is sound; the 70% figure is not — don't quote it.

3. **Upwork reply-rate "saturation" numbers (7.21% vs 7.45%) — single-vendor, weak.** From GigRadar (a bid-automation vendor) on self-selected data; measures *reply* rate, not *win* rate; the AI/ML-vs-mean gap is a trivial 0.24pp. The *directional* thesis (niche > generic bidding) is reasonable and backed by Upwork demand data, but don't treat the percentages as fact ([verification](https://gigradar.io/blog/upwork-market-report-2026)).

4. **Résumé-prevalence percentages (RAG ~70%, LangChain ~68%, MLOps ~42%, eval ~1/3, Python/PyTorch ~98%) — single self-published source (~100 profiles), unverifiable.** Also mis-bundled (PyTorch is ~85%, not 98%). The *qualitative* thesis (baseline vs differentiator) is well-supported; cite percentages as "illustrative, single-source" only.

5. **Project/retainer pricing tiers — the raw research numbers were mis-sourced.** "$750–$1,500 RAG builds," "$5k–$8k chatbots," and "$3k–$8k/mo retainers" are NOT in the cited Kellton article, and the retainer figure conflated LLM API + infra *operating cost* with freelancer income. Use the corrected tiers in §1. "2–3 retainers = $9k–$24k/mo" also has an arithmetic slip (should be $6k–$24k). **Plan revenue on the verified tiers, not the raw claim.**

6. **MLA-C01 "heavily MLOps, not model theory" — overstated.** Balanced 28/26/24/22; the two largest domains are data prep + model dev. Study evenly.

7. **SAA-C03 20–40 hr revision — optimistic.** Only holds for a genuine refresh; a first pass is ~60–80 hrs. Buffer accordingly.

8. **Toptal "$150 nets ~$70" — unverifiable.** Toptal bars rate disclosure; estimates split between ~$70 (pessimistic) and ~$100 (50%-markup reading). Braintrust's 15%/100% model is confirmed and the safer recommendation.

9. **nanochat star count "~43k" is stale** (now ~56k) — minor; doesn't change the thesis. Also, nanoGPT's "modest hardware" claim: full GPT-2 124M reproduction realistically wants an 8×A100 node, though it scales down.

10. **Edge/on-device ML and multimodal VLM niches — thin evidence** (trend-based, low-confidence). Watch, don't bet the plan on them.

**Well-verified and safe to build on:** all AWS exam facts (MLA-C01 and SAA-C03 numbers, domains, timing/retirement dates), the Karpathy nano-series as reproduction path (star counts + $73/3hr nanochat cost confirmed), rate anchors ($100/hr median, specialist premiums), Braintrust's fee model, and the qualitative portfolio/positioning guidance (deploy, niche, prove outcomes, lead with GSoC + SDET).