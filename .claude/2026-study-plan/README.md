# ML Study & Freelance Plan — Jul 18 → Dec 30, 2026

> **Purpose:** eliminate decision fatigue. Open this file. Do the week. Ship. Post. Repeat.
> **This year (2026) = hustle & build.** Next year (2027) = earn.

---

## 0. The one-paragraph strategy

You are not becoming "a machine learning engineer" (saturated). You are becoming **the ML engineer who ships *reliable, tested, production* GenAI systems** — and can prove it with deployed apps, reproductions, and quantified case studies. Your SDET background and GSoC record are not a detour from ML; they are your **moat**. The two scarcest, highest-paid skills in the 2026 market — **LLM evaluation/observability** and **MLOps/production reliability** — are exactly what a strong tester-turned-ML-engineer can own. This plan aims every chapter, project, and post at that position.

**One-line positioning (use it everywhere):**
> *"I help teams ship LLM/ML systems that actually work in production — I build them, then I test, evaluate, and harden them. ex-SDET · GSoC 2024 contributor."*

---

## 1. Why this direction (the decision you delegated)

You asked me to choose your specialization to optimize for: your history, the timeline, real-problem contribution, future-proofing against AI automation, and income. The research (see [`research-brief.md`](./research-brief.md)) points to one answer for *your* profile:

| Your criterion | Why applied-GenAI + eval/MLOps wins |
|---|---|
| **Leverages your history** | SDET (testing, validation, automation) + GSoC (open source) map *directly* onto the #1 and #2 low-supply skills: LLM eval/observability and MLOps. Almost no one else can claim "the ML engineer who tests." |
| **Fits the timeline** | You're already in PyTorch/DL chapters. GenAI app work builds straight on that and reaches portfolio-grade in ~5 months. |
| **Real-problem contribution** | Clients pay *now* for RAG over their docs, agents, and LLM integration — concrete business problems, not demos. |
| **Future-proof vs. AI automation** | The least-automatable ML work is **judgment work**: evaluation, adversarial testing, reliability, production integration, problem framing. Commodity model-training and basic prompting are the *most* automatable. Your testing instinct is the hedge. |
| **Income odds** | Highest-demand freelance niche (LLM apps) × highest-premium scarce skills (eval, MLOps) × a differentiator few can copy. |

**The hedge (important):** verification flagged that "classic ML is dying" is *false* — it grows ~22%/decade off a huge base. So you keep **solid classic-ML + CV competence** (one forecasting/predictive project, keep XGBoost sharp). GenAI is the spearhead; classic ML is the shield. Don't bet against either.

---

## 2. The locked decisions

- **Specialization:** Applied LLM/RAG/agent engineering + **LLM-eval/testing/MLOps wedge**, hedged with classic ML/CV.
- **Time budget:** **12–15 hrs/week** (~2 hrs weekdays + more on weekends). Buffer weeks built in.
- **Book:** Géron, *Hands-On ML with Scikit-Learn & PyTorch* (2025) — Part II (DL, Ch 9→19) + the last two of Part I (**Ch 7 Dimensionality Reduction, Ch 8 Unsupervised**) + **Appendix B (Mixed Precision & Quantization)**. Exercises trimmed to **4–6 hrs/chapter, concept-first** (see [`chapters.md`](./chapters.md)).
- **Certs:** **AWS SAA-C03 (revision) in August.** ML-cert (MLA-C01 vs MLA-C02) **decided in September** via a framework (see [`certs.md`](./certs.md)). ⚠️ Hard constraint: **MLA-C01 retires Sept 28, 2026.**
- **Output cadence:** ~2–3 LinkedIn posts/week, batched weekly. Every project/reproduction ships a post + a case study.

> **You can override any of these.** The one most worth revisiting is the September cert call — the framework is in `certs.md`.

---

## 3. How the pieces fit

```
BOOK CHAPTER  ──teaches──▶  A PROJECT or PAPER REPRODUCTION  ──produces──▶  A DEPLOYED DEMO + CASE STUDY + LINKEDIN POST
     │                                    │                                              │
 (4–6 hrs, trimmed)              (the portfolio)                            (the reputation / inbound leads)
```

Nothing is learned in a vacuum. Every chapter has a downstream artifact. The rule from the research: **a live demo URL is the line between "notebook person" and "ML engineer."** Deploy everything. Three deployed projects beat ten notebooks.

---

## 4. The map (24 weeks, 4 blocks)

| Block | Weeks | Dates | Theme | Big deliverables |
|---|---|---|---|---|
| **A** | W1–6 | Jul 18 – Aug 28 | Setup + **SAA cert** + Part I finish | AWS SAA passed · repo/brand set up · P0 anomaly demo |
| **B** | W7–12 | Aug 29 – Oct 9 | Core DL + first projects + **cert decision** | makemore repro · P1 image classifier · P2 forecasting (hedge) |
| **C** | W13–19 | Oct 10 – Nov 27 | Transformers/LLMs + **anchor projects** | nanoGPT repro · **P3 RAG API (anchor)** · **P4 LLM-eval pipeline (your wedge)** |
| **D** | W20–24 | Nov 28 – Dec 30 | Agentic/MLOps + consolidate + position | P5 agent/MLOps · LoRA repro · portfolio + freelance profiles live |

Full week-by-week detail: **[`calendar.md`](./calendar.md)**.

---

## 5. What "done" looks like on Dec 30, 2026

- ✅ **1 cert in hand** (AWS SAA-C03) + a concrete plan/booking for the AWS ML Engineer cert.
- ✅ **3 pinned, deployed, quantified portfolio projects** — anchored by a production **RAG API** and an **LLM-eval/observability pipeline** (your differentiator).
- ✅ **2 packaged paper/model reproductions** with writeups (nanoGPT + LoRA/QLoRA).
- ✅ **Book Part II core done** (Ch 10–17 + App B), Ch 7–8, transformers/LLMs genuinely understood.
- ✅ **A LinkedIn presence that self-selects inbound** — ~50+ posts, profile as a landing page, GSoC + SDET foregrounded.
- ✅ **Freelance profiles live** (Braintrust + a niched Upwork), ready to take work in Q1 2027.

---

## 6. Operating rules (read once, live by them)

1. **Ship > perfect.** A deployed 85%-good project beats a local 99% one. Deploy, then improve.
2. **Deploy every project** (Hugging Face Space / Streamlit Cloud / Railway). No demo URL = didn't happen.
3. **Quantify everything.** "Cut retrieval latency 640ms → 95ms," "raised answer-faithfulness 0.62 → 0.89 (RAGAS)." Numbers are the currency of credibility.
4. **Lead with the wedge.** In every writeup, show the *testing/eval* you did. That's the part others skip and clients fear most.
5. **Trim ruthlessly.** 4–6 hrs/chapter. When a lab turns into a rabbit hole, note the question and move on. Depth comes from projects, not from exhausting a chapter.
6. **Core vs. stretch.** Each week has a core (must-do) and stretch. Behind? Drop stretch, never the anchor projects, SAA, or Ch 15 (transformers/LLMs).
7. **Post from the work, not in addition to it.** Every session leaves a breadcrumb (a screenshot, a metric, a gotcha) → that's your next post. See [`brand.md`](./brand.md).
8. **Protect momentum over completeness.** Missing Ch 19 (RL) is fine. Missing the RAG project is not.

---

## 7. The files in this plan

| File | What it's for |
|---|---|
| [`calendar.md`](./calendar.md) | **Your daily driver.** Week-by-week: chapter, project, cert, hours, deliverable. |
| [`checkpoints.md`](./checkpoints.md) | **The governance layer.** Pre-decided persist-vs-pivot rules at W6/W12/W19/W24 + 2027. When to keep going, when to adjust, when (rarely) to stop. |
| [`chapters.md`](./chapters.md) | Per-chapter: core concepts + a trimmed 4–6 hr lab set + what to skip + which project it feeds. |
| [`projects.md`](./projects.md) | Full specs for P0–P5: problem, stack, scope, success metrics, deployment, the post it generates. |
| [`certs.md`](./certs.md) | AWS SAA-C03 revision plan + the September ML-cert decision framework. Verified exam facts. |
| [`brand.md`](./brand.md) | LinkedIn/reputation playbook: content engine, cadence, profile-as-landing-page, GSoC/SDET positioning. |
| [`resources.md`](./resources.md) | Curated resource stack (with when-to-use) + ranked reproduction targets. |
| [`research-brief.md`](./research-brief.md) | The full cited market research this plan is built on (with verification caveats). |

---

## 8. Weekly ritual (Sunday, ~30 min)

1. Open `calendar.md` → read next week's row.
2. Skim the chapter's section in `chapters.md`; if it's a new chapter, generate/trim its lab set (ask me: *"make the trimmed 4–6 hr lab set for Ch N"*).
3. Check the project's spec in `projects.md`; write this week's one concrete deliverable on a sticky note.
4. Batch last week's breadcrumbs into 2–3 posts (`brand.md`).
5. Move the previous week's row to "done." Momentum is the whole game.
6. **60-sec pulse:** log two Y/N — did I hit my hours? did I ship something? (Feeds the checkpoints so they hold no surprises.)

**At the end of each block (W6, W12, W19, W24)** run the ~45-min checkpoint in [`checkpoints.md`](./checkpoints.md). That's the *only* time you're allowed to change strategy — between checkpoints you execute and park any doubts. This is what makes full commitment safe.

---

*Built from deep research (25 research agents, 8 market angles, adversarial verification). Sources and caveats live in [`research-brief.md`](./research-brief.md). Note: `.claude/` is git-ignored — if you want this plan in version control, either move it out of `.claude/` or add a `!/.claude/2026-study-plan/` negation to `.gitignore`.*
