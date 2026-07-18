# AWS Certifications Plan

**Reality check first:** for freelance ML income, **proof-of-results (deployed projects + quantified case studies) outweighs certs.** Certs are a *credibility supplement* and a forcing function for cloud fluency — not the main event. So: one quick cert now (SAA), ML cert decided in September without derailing your portfolio.

All exam facts below were **verified against official AWS sources** during the research pass.

---

## 1. AWS Certified Solutions Architect – Associate (SAA-C03) — August

**Status:** current in 2026; **no SAA-C04 announced** (blogs claiming one are wrong). Do this as **revision**.

### Verified exam facts
| | |
|---|---|
| Questions | **65** (50 scored + 15 unscored) |
| Time | **130 minutes** |
| Cost | **$150 USD** |
| Pass | **720 / 1000** (scaled) |
| Scoring | **Compensatory** — pass overall, no per-domain minimum, no wrong-answer penalty → **answer everything** |
| Validity | 3 years |

### Domain weights (study in this priority)
1. **Design Secure Architectures — 30%** (biggest: IAM, KMS, encryption, security groups, least privilege)
2. Design Resilient Architectures — 26% (multi-AZ, ELB/ASG, decoupling, backups)
3. Design High-Performing Architectures — 24% (caching, right-sizing, S3/EBS/EFS, read replicas)
4. Design Cost-Optimized Architectures — 20% (instance purchasing, S3 tiers, budgets)

### Hours — ⚠️ be honest with yourself
- **True refresh of prior mastery:** ~20–40 hrs over 2–4 weeks.
- **If it's effectively a first real pass:** budget **60–80 hrs** (independent sources cluster here for AWS-familiar candidates). The plan reserves W2–W5 (~40 hrs) — **if your diagnostic is weak, borrow hours from stretch items or push the exam to early September.**

### Revision loop (practice-test-driven)
1. **[Official Exam Guide PDF](https://d1.awsstatic.com/training-and-certification/docs-sa-assoc/AWS-Certified-Solutions-Architect-Associate_Exam-Guide.pdf)** — confirm nothing drifted.
2. **[Tutorials Dojo practice exams](https://portal.tutorialsdojo.com/courses/aws-certified-solutions-architect-associate-practice-exams/)** — the core of your loop. **TD runs ~8–13 pts harder than the real exam**; aim **≥ 80% on fresh sets** as your go/no-go.
3. **[TD free cheat sheets](https://tutorialsdojo.com/aws-cheat-sheets/)** — fast service-comparison refreshers.
4. **[Maarek SAA course/practice tests](https://www.udemy.com/course/aws-certified-solutions-architect-associate-saa-c03/)** — a strong second question bank; watch only weak-topic videos at 1.5×.

### The plan
- **W1:** diagnostic TD test → identify weak domains.
- **W2–W5:** one domain focus/week (Secure → Resilient/High-Perf → Cost) + rolling full-length TD exams.
- **W5:** hit ≥ 80% fresh → **book the exam.**
- **W6:** **sit it.** Post the win + your revision loop.

**Bonus:** SAA's IAM/VPC/S3/networking/cost foundations directly reduce the ML-cert study load later.

---

## 2. AWS ML Engineer – Associate — ⚠️ **DECIDE IN SEPTEMBER (W9)**

You chose to make this call in September once SAA is done and you see your pace. Here's everything you need to decide fast.

### The hard constraint
- **MLA-C01 is being retired.** **Last day to sit MLA-C01 (English): Sept 28, 2026.** (verified: official AWS Training & Certification blog)
- **MLA-C02** (successor): beta registration **Sept 1, 2026**; beta delivery **Sept 29, 2026+**; standard version **early 2027**. **MLA-C02 adds generative-AI / agentic / Bedrock topics** — i.e. it's *more aligned with your LLM-app direction.*

### Verified MLA-C01 facts (if you race it)
| | |
|---|---|
| Questions / time / cost | **65 / 130 min / $150** · pass **720/1000** · valid 3 yrs |
| Domain 1 — **Data Prep for ML** | **28%** |
| Domain 2 — **ML Model Development** | **26%** |
| Domain 3 — Deployment & Orchestration | **22%** |
| Domain 4 — Monitoring, Maintenance & Security | **24%** |

> **Correction to a common myth:** it is **not** "all MLOps, no modeling." The two biggest domains (Data Prep 28% + Model Dev 26% = 54%) are data/modeling. It's SageMaker-centric (Data Wrangler, Feature Store, Pipelines, Model Monitor, Clarify, JumpStart, endpoints) + Glue/Athena/Kinesis/Step Functions/CloudWatch + IAM + cost. **Study evenly, slight tilt to data prep.**

- **Hours:** AWS-familiar but new to SageMaker → **~80–120 hrs (~8–12 wks @ ~10 hrs/wk).**

### The September decision framework
Answer these in W9:

1. **Did SAA go smoothly and are you energized?** (not burned out)
2. **Can you commit ~90 focused cert hours in Sep 1–26 *without* killing the RAG project (P3) and Ch 15?**
3. **Do you want a cert-in-hand in 2026 more than you want a GenAI-aligned cert?**

- **If yes / yes / yes → RACE MLA-C01.** Use the compressed plan below; sit by **Sept 26** (48-hr buffer before the Sept 28 cutoff). Accept that Aug–Sept is cert-heavy and P3 shifts to October.
- **Otherwise → TARGET MLA-C02** (recommended default). Keep Sept for Ch 15 + P3. Prep MLA-C02 in **Jan–Feb 2027** as your "earning year" opener — fresher, GenAI-aligned, and it doesn't cannibalize the portfolio that actually drives income.

> **My recommendation:** default to **MLA-C02 in early 2027** unless the SAA sprint finished with time to spare and real appetite. Racing a *retiring* exam at the cost of your flagship RAG project + LLM chapter is a poor trade for someone whose income thesis is "deployed LLM systems + eval." You still get the AWS ML Engineer Associate cert — the better version.

### Compressed MLA-C01 race plan (only if you choose to race)
- **Sep 1–7:** [Official exam guide](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-01/machine-learning-engineer-associate-01.html) → map every topic; [AWS Skill Builder MLA-C01 prep](https://skillbuilder.aws/category/exam-prep/machine-learning-engineer-associate-MLA-C01) (Enhanced plan = Builder Labs, hands-on SageMaker).
- **Sep 8–18:** [Maarek & Kane Udemy course](https://www.udemy.com/course/aws-certified-machine-learning-engineer-associate-mla-c01/) at 1.25–1.5× (verbose) + SageMaker labs on the two big domains (Data Prep, Model Dev).
- **Sep 19–25:** [Tutorials Dojo MLA-C01 practice exams](https://tutorialsdojo.com/aws-certified-machine-learning-engineer-associate-mla-c01-exam-guide/) → drill to ~80%+; AWS Official Practice Question Set to calibrate.
- **Sep 26:** **sit it.** *(No dedicated Adrian Cantrill MLA-C01 course exists as of 2025–26.)*
- During this race, **P3 slides to October** and book work pauses to Ch 15 reading only. Adjust `calendar.md` Block B/C accordingly.

### MLA-C02 path (recommended default)
- 2026: nothing — protect the portfolio.
- **Jan–Feb 2027:** ~80–120 hrs; same SageMaker core **plus** the new GenAI/Bedrock/agentic content (which by then you'll half-know from P3/P4/P5). Sit the standard exam when available. This becomes your first 2027 credential right as you go to market.

---

## Cert summary

| Cert | When | Hours | Status |
|---|---|---|---|
| **SAA-C03** | Aug 2026 (W2–W6) | 40 (buffer 60–80) | **Do it** |
| **AWS ML Engineer** | Decide W9 | 80–120 | **C01 race by Sep 28** *or* **C02 in early 2027** (recommended) |
