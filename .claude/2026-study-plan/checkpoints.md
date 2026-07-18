# Decision Checkpoints — When to Persist vs. Pivot

> **Why this file exists:** committing fully is only safe if you know in advance the exact conditions under which you'd stop. This file pre-decides them. It turns "should I keep going?" from a daily anxiety into a scheduled, evidence-based review. Write the rules now, while you're calm and objective — so future-you, tired and doubting on a Tuesday, doesn't get to renegotiate the strategy alone.

---

## The one rule that makes the rest work

**You may only change strategy at a checkpoint — never between them.**

Between checkpoints, the answer to *"should I switch to X?"* is always: **no. Write it in the parking lot (below) and keep executing.** Thrash — restarting, chasing shiny new things, re-litigating settled decisions — is the single biggest threat to someone with your time budget. The compounding you're betting on only happens if you stay in the lane long enough for it to kick in. This rule is the guardrail.

---

## Three kinds of failure (diagnose before you act)

When something feels wrong, it's almost always one of these three — and they demand **opposite responses.** Misdiagnosing is how people quit a working strategy or cling to a broken one.

| Failure mode | Looks like | Correct response | Wrong response |
|---|---|---|---|
| **1. Execution** | You're not doing the work — missed hours, nothing shipping, posts stopped. | Fix the *inputs*: re-cut the time budget to what's real, protect a study block, shrink scope. **Extend the timeline.** | Pivoting strategy. You haven't tested anything yet — there's nothing to conclude. |
| **2. Output/skill** | You're doing the work, but projects aren't deploying / metrics are weak / posts are flat. | Fix the *craft*: one focused sprint on the specific blocker (a deployment, a content format). Adjust *tactics*. | Concluding "ML freelancing doesn't work." A stuck deploy is a skill gap, not a market verdict. |
| **3. Market** | Portfolio is complete, deployed, quantified — and after *sustained real outreach*, the market gives ~zero signal. | **This is the only mode that justifies rethinking the strategy.** | Reaching this conclusion early. You cannot diagnose market failure until you've cleared modes 1 and 2 *and* actually gone to market. |

**Rule of thumb:** in 2026, essentially every problem you hit is mode 1 or 2. Mode 3 cannot be honestly diagnosed until ~mid-2027, because it requires a finished portfolio *plus* a real sales push. Don't let a mode-1 problem (you got busy) masquerade as a mode-3 verdict (the world doesn't want this).

---

## Two lightweight pulses (so checkpoints hold no surprises)

**Weekly (60 seconds, part of the Sunday ritual):** answer two yes/no questions and log them.
- Did I hit roughly my study hours this week? (Y/N)
- Did I ship *something* — code, a deploy, a post? (Y/N)

Two N's in a row on either line = an early execution wobble. Fix the schedule *this* week; don't wait for the checkpoint.

**Monthly (30 min):** skim the last 4 weekly logs. Are execution and posting trending up, flat, or down? Note it. No decisions — just awareness so the quarterly checkpoint is a formality, not a shock.

---

## Track these (you can't check what you don't measure)

Keep a dead-simple log (a `progress.md` or a spreadsheet). Split into **execution metrics** (fully in your control — be strict) and **market metrics** (partly out of your control — read *trends*, not single data points).

- **Execution:** weeks you hit hours · artifacts shipped · artifacts *deployed* (live URL) · posts published.
- **Market:** post impressions/reactions/comments (trend over months) · profile views (and *who* — recruiters? founders? ML people?) · any DM/comment/inbound from someone who could hire you.

---

## The checkpoints

Each maps to the end of a block in `calendar.md`. At each: pull your numbers, classify 🟢/🟡/🔴, diagnose the failure mode if not 🟢, apply the pre-decided action, and **write one paragraph** recording what you decided and why (so you can't silently rewrite history later).

> Market thresholds below are **starting defaults** — calibrate to your own baseline. The execution thresholds are the strict ones; those you control.

### ✅ Checkpoint 1 — End of Block A (~W6, Aug 28) · *"Does the machine run?"*
Pure execution test. **No market signal is expected yet** — don't even look for it.

| | Condition | Action |
|---|---|---|
| 🟢 | SAA passed (or booked, hitting ≥80% on fresh TD sets) · repo restructured · **P0 deployed (live URL)** · profile live · ≥3 posts out · hit hours ≥4 of 6 weeks | Continue to Block B as written. |
| 🟡 | SAA slipping but recoverable · P0 built but not deployed · posts sporadic | **Protect the deploy habit and the posting habit above all** — those are the skills, not the study. Tighten scope; add nothing new. |
| 🔴 | Didn't sit/pass SAA **and** didn't deploy P0 **and** posted <2× **and** missed hours most weeks | **Mode 1 (execution).** 12–15 hrs/week was likely unrealistic against your life right now. Re-cut to 8–10 hrs, stretch the calendar by 4–6 weeks, and fix the *environment* (a fixed, protected block). **Do not touch the strategy.** |

### ✅ Checkpoint 2 — End of Block B (~W12, Oct 9) · *"Can I ship, and is anyone noticing?"*
Execution proven + first *weak* market pulse. Cert decision must be locked.

| | Condition | Action |
|---|---|---|
| 🟢 | SAA done · makemore + P1 + P2 all **deployed with metrics** · cert decision made & written · posting 2–3×/wk · engagement *trending up* and ≥1 post clearly beat your baseline | Continue to Block C — the anchor block. |
| 🟡 | Shipping but stuck *local* (not deploying) · OR posting but flat for 6 wks · OR skipped the cert decision | Diagnose: deploy-blocked → **Mode 2**, one focused sprint to take *one* thing fully to production. Posts flat → content problem, apply `brand.md` fixes (carousels, hooks, comment 5×/day) — *not* a strategy change. Make the cert call now. |
| 🔴 | <2 of {makemore, P1, P2} deployed **and** posting collapsed | **Mode 1 again.** Same as CP1: re-cut hours, extend timeline, fix environment. Still no portfolio to test the market with — no strategy conclusion is available. |

### ⭐ Checkpoint 3 — End of Block C (~W19, Nov 27) · *"Do the anchors exist?"* — THE BIG ONE
This is make-or-break for the plan's core promise: *can you build and prove production LLM systems?* P3 (RAG API) and P4 (eval pipeline) are the proof.

| | Condition | Action |
|---|---|---|
| 🟢 | **P3 + P4 deployed with quantified metrics** · nanoGPT writeup published · LoRA done · engagement trending up | Continue to Block D. You now have a portfolio that proves the thesis. |
| 🟡 | One anchor deployed, the other stuck · OR deployed but eval is shallow / metrics thin | **Priority-protection moment.** Cut *everything* else — P5, Ch 18/19, extra posts — and finish the anchors. Per the calendar triage, the anchors are non-negotiable. |
| 🔴 | Neither anchor deployed by end of November | **Mode 1 or 2 — decide which.** If it's hours (Mode 1): extend into January; the targets stay. If the work is genuinely beyond current reach in the time (Mode 2): **scope down, don't abandon** — a simpler *deployed* RAG + a lighter *working* eval harness proves the thesis and beats two unfinished ambitious ones. The pivot here is *scope*, never *direction*. |

### ✅ Checkpoint 4 — End of Block D (~W24, Dec 30) · *"Am I positioned to sell?"*
Positioning + first real market read. **Paid work by Dec 30 is a bonus, not the bar** — the goal was always income in *2027*.

| | Condition | Action |
|---|---|---|
| 🟢 | 3 pinned, deployed, quantified projects · portfolio site live · freelance profiles live · retrospective posted · *any* early signal (relevant profile views, a DM, an inbound question) | You're on the runway. Move to the 2027 launch (below). |
| 🟡 | Portfolio there, but profiles not set up / zero market signal | Set up the profiles — it's mechanical, do it. Begin light warm-up outbound. Zero inbound now is **normal** (audience lags months); not a red flag. |
| 🔴 | Portfolio incomplete **and** no market presence | **Almost always Mode 1 (timeline).** Roll the unfinished pieces into a Jan–Feb finishing sprint *before* going to market. The plan needed more calendar, not a new plan. |

### 🎯 Checkpoint 5 — The 2027 Income Test (~Q2 2027) · *"Does the market pay?"* — the REAL thesis test
Only here do you judge the **strategy** itself. Preconditions: a complete, deployed, quantified portfolio **and** a *genuine* sustained sales push — 8–12 weeks of real effort (Braintrust/Upwork applications, niched pitches, consistent content), not a half-hearted week.

| Signal | Meaning | Action |
|---|---|---|
| 🟢 **Go** | Discovery calls happening · responses to applications · first contract(s) landed, even small | **The thesis is validated.** Continue: raise rates, productize (the "LLM eval audit"), stack retainers. This is the 2027 earning year firing. |
| 🟡 **Fix sales, not skill** | Getting interviews/calls but not *closing* | **Mode 2, sales edition** — positioning/pitch/pricing/niche problem, and it's coachable. Get feedback on your pitch and profile; sharpen the niche; adjust rates. You're at the 1-yard line — don't walk off the field. |
| 🔴 **Legitimate pivot signal** | After a *real* sustained push with a complete portfolio: essentially zero interest — no responses, no calls, no relevant engagement | **Now — and only now — reconsidering the freelance-services thesis is supported.** Even here the downside is protected: convert the same skill into a **full-time ML role** (the floor this whole plan was built on), or aim the identical portfolio at a **product** instead of services. You lose nothing you built. |

---

## Kill criteria — the honest bar for "the strategy is wrong"

Abandoning or majorly re-routing the *strategy* is justified **only when all three hold** — and by construction that can't happen before ~mid-2027:

1. **You executed** — shipped the artifacts, posted consistently (rules out Mode 1).
2. **You ran a real market test** — a sustained outreach push with a complete, deployed, quantified portfolio (rules out "never gave it a chance").
3. **The market returned ~nothing** over that window (Mode 3 confirmed).

Anything short of all three is a **tactical adjustment**, not a kill: re-cut hours, focused skill sprint, change content format, fix the pitch, scope a project down. These are expected and frequent. Killing the strategy is rare and late.

---

## What is NOT a reason to pivot (read this when you're doubting)

The pivot urge almost always shows up disguised as one of these. None of them are valid signals:

- **"This week was hard / I'm confused."** That's *learning*, not failure. The confusion is the work.
- **"No inbound yet"** (in 2026). Expected. Audiences and reputation lag by *months*. Judged too early, every reputation play looks dead.
- **"A post flopped."** Normal and frequent. Judge the *trend over months*, never a single post.
- **"I saw a shinier opportunity"** — a new hot framework, someone's viral success, a fresh business idea. **This is the most dangerous trigger and exactly the thrash this plan exists to kill.** → Parking lot. Revisit only at the next checkpoint.
- **"Someone online said [my niche] is saturated/dead."** You have adversarially-verified research (`research-brief.md`). Don't re-litigate a settled, evidenced decision on a stranger's hot take.
- **"Others are further ahead."** Different start line, different hours. Irrelevant to whether *your* plan is working.

---

## Parking lot

When a shiny idea or a "maybe I should instead…" hits mid-block, **write it here and keep going.** You'll review the list — with a clear head and real data — at the next checkpoint. This is where thrash goes to wait.

```
- [ ] (date) idea / doubt / temptation:
- [ ]
- [ ]
```

---

## How to run a checkpoint (45 min)

1. Pull your execution + market numbers from the log.
2. Classify the block 🟢 / 🟡 / 🔴 against the table above.
3. If not 🟢, name the **failure mode** (1 / 2 / 3) before choosing an action — this is the step that prevents a wrong pivot.
4. Apply the pre-decided action. Update `calendar.md` if the timeline shifts.
5. Review the parking lot: adopt, park again, or discard each item — deliberately.
6. Write **one paragraph**: what you decided and why. Date it. This record is what stops future-you from quietly rewriting the strategy on a bad day.
